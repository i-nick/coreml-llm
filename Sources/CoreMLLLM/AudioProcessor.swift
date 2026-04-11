import Accelerate
import CoreML
import Foundation

/// Computes mel spectrograms and runs audio projection for the Gemma 4 audio encoder.
///
/// Pipeline: raw PCM → mel spectrogram → CoreML Conformer → Swift projection → features
///
/// The final projection (output_proj + RMSNorm + embed_proj) runs in Swift/Accelerate
/// with float32 precision because CoreML GPU runtime corrupts RMSNorm(with_scale=False).
public enum AudioProcessor {

    // MARK: - Constants (matching HF Gemma4AudioFeatureExtractor)

    static let sampleRate = 16000
    static let frameLength = 320
    static let hopLength = 160
    static let fftLength = 512
    static let numMelBins = 128
    static let numFFTBins = fftLength / 2 + 1  // 257
    static let melFloor: Float = 0.001

    // MARK: - Projection weights (loaded from .npy files)

    /// Loaded projection weights for Swift-side computation.
    public struct ProjectionWeights {
        let outputProjWeight: [Float]  // (1536, 1024) row-major
        let outputProjBias: [Float]    // (1536,)
        let embedProjWeight: [Float]   // (1536, 1536) row-major
        let inDim: Int                 // 1024
        let outDim: Int                // 1536

        /// Load projection weights from .npy files in the model directory.
        public static func load(from directory: URL) throws -> ProjectionWeights {
            let opW = try loadNpyFloat16(directory.appendingPathComponent("output_proj_weight.npy"))
            let opB = try loadNpyFloat16(directory.appendingPathComponent("output_proj_bias.npy"))
            let epW = try loadNpyFloat16(directory.appendingPathComponent("embed_proj_weight.npy"))
            return ProjectionWeights(
                outputProjWeight: opW, outputProjBias: opB,
                embedProjWeight: epW, inDim: 1024, outDim: 1536)
        }

        /// Load a float16 numpy file as [Float].
        private static func loadNpyFloat16(_ url: URL) throws -> [Float] {
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            // Numpy header: first 10 bytes + header_len at bytes 8-9
            var headerSize = 128
            if data.count > 10 {
                data.withUnsafeBytes { raw in
                    let b = raw.baseAddress!.assumingMemoryBound(to: UInt8.self)
                    headerSize = 10 + (Int(b[8]) | (Int(b[9]) << 8))
                }
            }
            let count = (data.count - headerSize) / MemoryLayout<UInt16>.stride
            var result = [Float](repeating: 0, count: count)
            data.withUnsafeBytes { raw in
                let src = raw.baseAddress!.advanced(by: headerSize)
                    .assumingMemoryBound(to: UInt16.self)
                result.withUnsafeMutableBufferPointer { dst in
                    var srcBuf = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: src),
                        height: 1, width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<UInt16>.stride)
                    var dstBuf = vImage_Buffer(
                        data: dst.baseAddress!, height: 1,
                        width: vImagePixelCount(count),
                        rowBytes: count * MemoryLayout<Float>.stride)
                    vImageConvert_Planar16FtoPlanarF(&srcBuf, &dstBuf, 0)
                }
            }
            return result
        }
    }

    // MARK: - Public API

    /// Process raw audio through the Conformer encoder + Swift projection.
    ///
    /// - Parameters:
    ///   - samples: Raw PCM audio samples (Float, mono, 16kHz)
    ///   - audioModel: CoreML Conformer model (outputs hidden_states 1024-dim)
    ///   - melFilterbank: Mel filterbank matrix (257 × 128)
    ///   - targetFrames: Number of mel frames the model expects
    ///   - projection: Projection weights for Swift-side computation
    /// - Returns: Audio features MLMultiArray (1, numTokens, 1536)
    public static func process(
        _ samples: [Float],
        with audioModel: MLModel,
        melFilterbank: [Float],
        targetFrames: Int,
        projection: ProjectionWeights
    ) throws -> MLMultiArray {
        let mel = computeMelSpectrogram(
            samples, melFilterbank: melFilterbank, targetFrames: targetFrames)

        // CoreML Conformer encoder → hidden_states (1, S, 1024)
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_features": MLFeatureValue(multiArray: mel),
        ])
        guard let hidden = try audioModel.prediction(from: input)
                .featureValue(for: "hidden_states")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }

        // Swift float32 projection: output_proj → RMSNorm → embed_proj
        return projectHiddenStates(hidden, with: projection)
    }

    /// Run output_proj + RMSNorm + embed_proj in float32 on CPU via Accelerate.
    ///
    /// This avoids a CoreML GPU runtime bug where RMSNorm(with_scale=False)
    /// produces all-zeros output, corrupting the final audio features.
    static func projectHiddenStates(
        _ hidden: MLMultiArray,
        with proj: ProjectionWeights
    ) -> MLMultiArray {
        let S = hidden.shape[1].intValue
        let inDim = proj.inDim
        let outDim = proj.outDim
        let hp = hidden.dataPointer.bindMemory(to: Float16.self, capacity: hidden.count)

        // Output: (1, S, 1536) float16
        let result = try! MLMultiArray(
            shape: [1, NSNumber(value: S), NSNumber(value: outDim)],
            dataType: .float16)
        let rp = result.dataPointer.bindMemory(to: Float16.self, capacity: S * outDim)

        var hVec = [Float](repeating: 0, count: inDim)
        var projected = [Float](repeating: 0, count: outDim)
        var features = [Float](repeating: 0, count: outDim)

        for t in 0..<S {
            // fp16 → fp32
            for i in 0..<inDim { hVec[i] = Float(hp[t * inDim + i]) }

            // output_proj: W(1536,1024) @ x(1024) + bias(1536)
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        Int32(outDim), Int32(inDim),
                        1.0, proj.outputProjWeight, Int32(inDim),
                        hVec, 1, 0.0, &projected, 1)
            for i in 0..<outDim { projected[i] += proj.outputProjBias[i] }

            // RMSNorm (float32, no learnable scale)
            var sumSq: Float = 0
            vDSP_svesq(projected, 1, &sumSq, vDSP_Length(outDim))
            let rms = sqrt(sumSq / Float(outDim) + 1e-6)
            var invRms = 1.0 / rms
            vDSP_vsmul(projected, 1, &invRms, &projected, 1, vDSP_Length(outDim))

            // embed_proj: W(1536,1536) @ x(1536)
            cblas_sgemv(CblasRowMajor, CblasNoTrans,
                        Int32(outDim), Int32(outDim),
                        1.0, proj.embedProjWeight, Int32(outDim),
                        projected, 1, 0.0, &features, 1)

            // fp32 → fp16
            for i in 0..<outDim { rp[t * outDim + i] = Float16(features[i]) }
        }

        return result
    }

    // MARK: - Mel spectrogram

    /// Compute mel spectrogram from raw audio.
    public static func computeMelSpectrogram(
        _ samples: [Float],
        melFilterbank: [Float],
        targetFrames: Int
    ) -> MLMultiArray {
        let padLeft = frameLength / 2
        var padded = [Float](repeating: 0, count: padLeft + samples.count)
        padded.replaceSubrange(padLeft..<padLeft + samples.count, with: samples)

        let unfoldSize = frameLength + 1
        let numFrames = max(0, (padded.count - unfoldSize) / hopLength + 1)
        let actualFrames = min(numFrames, targetFrames)

        var window = [Float](repeating: 0, count: frameLength)
        for n in 0..<frameLength {
            window[n] = 0.5 - 0.5 * cos(2 * .pi * Float(n) / Float(frameLength))
        }

        let log2n = vDSP_Length(log2(Double(fftLength)))
        guard let fftSetup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            fatalError("Failed to create FFT setup")
        }
        defer { vDSP_destroy_fftsetup(fftSetup) }

        var melSpec = [Float](repeating: 0, count: targetFrames * numMelBins)
        var realp = [Float](repeating: 0, count: fftLength / 2)
        var imagp = [Float](repeating: 0, count: fftLength / 2)
        var windowed = [Float](repeating: 0, count: fftLength)
        var magnitude = [Float](repeating: 0, count: numFFTBins)

        for f in 0..<actualFrames {
            let start = f * hopLength
            let frameSlice = Array(padded[start..<min(start + unfoldSize, padded.count)])
            let frame: [Float]
            if frameSlice.count >= unfoldSize {
                frame = Array(frameSlice[..<frameLength])
            } else {
                var tmp = frameSlice
                tmp.append(contentsOf: [Float](repeating: 0, count: unfoldSize - tmp.count))
                frame = Array(tmp[..<frameLength])
            }

            vDSP_vmul(frame, 1, window, 1, &windowed, 1, vDSP_Length(frameLength))
            for i in frameLength..<fftLength { windowed[i] = 0 }

            windowed.withUnsafeBufferPointer { buf in
                buf.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: fftLength / 2) { complexPtr in
                    realp.withUnsafeMutableBufferPointer { rBuf in
                        imagp.withUnsafeMutableBufferPointer { iBuf in
                            var split = DSPSplitComplex(realp: rBuf.baseAddress!, imagp: iBuf.baseAddress!)
                            vDSP_ctoz(complexPtr, 2, &split, 1, vDSP_Length(fftLength / 2))
                            vDSP_fft_zrip(fftSetup, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                        }
                    }
                }
            }

            var half: Float = 0.5
            vDSP_vsmul(realp, 1, &half, &realp, 1, vDSP_Length(fftLength / 2))
            vDSP_vsmul(imagp, 1, &half, &imagp, 1, vDSP_Length(fftLength / 2))

            magnitude[0] = abs(realp[0])
            for k in 1..<(fftLength / 2) {
                magnitude[k] = sqrt(realp[k] * realp[k] + imagp[k] * imagp[k])
            }
            magnitude[fftLength / 2] = abs(imagp[0])

            let melOffset = f * numMelBins
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        Int32(numFFTBins), Int32(numMelBins),
                        1.0, melFilterbank, Int32(numMelBins),
                        magnitude, 1, 0.0, &melSpec[melOffset], 1)
            for i in 0..<numMelBins {
                melSpec[melOffset + i] = log(melSpec[melOffset + i] + melFloor)
            }
        }

        if actualFrames < targetFrames {
            let logFloor = log(melFloor)
            for i in (actualFrames * numMelBins)..<(targetFrames * numMelBins) {
                melSpec[i] = logFloor
            }
        }

        let totalElements = targetFrames * numMelBins
        let result = try! MLMultiArray(
            shape: [1, NSNumber(value: targetFrames), NSNumber(value: numMelBins)],
            dataType: .float16)
        let dst = result.dataPointer.bindMemory(to: UInt16.self, capacity: totalElements)
        melSpec.withUnsafeBufferPointer { src in
            var srcBuf = vImage_Buffer(data: UnsafeMutableRawPointer(mutating: src.baseAddress!),
                                        height: 1, width: vImagePixelCount(totalElements),
                                        rowBytes: totalElements * MemoryLayout<Float>.stride)
            var dstBuf = vImage_Buffer(data: dst, height: 1,
                                        width: vImagePixelCount(totalElements),
                                        rowBytes: totalElements * MemoryLayout<UInt16>.stride)
            vImageConvert_PlanarFtoPlanar16F(&srcBuf, &dstBuf, 0)
        }
        return result
    }

    // MARK: - Utilities

    /// Load mel filterbank matrix from binary file (257 × 128 float32).
    public static func loadMelFilterbank(from url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        let count = numFFTBins * numMelBins
        var result = [Float](repeating: 0, count: count)
        data.withUnsafeBytes { raw in
            let src = raw.baseAddress!.assumingMemoryBound(to: Float.self)
            for i in 0..<min(count, data.count / MemoryLayout<Float>.stride) {
                result[i] = src[i]
            }
        }
        return result
    }

    /// Extract a single audio feature token from the output.
    public static func sliceFeature(_ features: MLMultiArray, at index: Int,
                                     hiddenSize: Int) -> MLMultiArray {
        let r = try! MLMultiArray(shape: [1, 1, NSNumber(value: hiddenSize)],
                                  dataType: .float16)
        let s = features.dataPointer.bindMemory(to: UInt16.self, capacity: features.count)
        let d = r.dataPointer.bindMemory(to: UInt16.self, capacity: hiddenSize)
        memcpy(d, s.advanced(by: index * hiddenSize),
               hiddenSize * MemoryLayout<UInt16>.stride)
        return r
    }
}
