import CoreML
import CoreGraphics
import Foundation

/// Processes images for Gemma 4 multimodal vision encoder.
public enum ImageProcessor {

    /// Process an image through the vision encoder CoreML model.
    ///
    /// - Parameters:
    ///   - image: Input CGImage
    ///   - visionModel: Compiled vision CoreML model
    /// - Returns: Image features MLMultiArray (1, 280, hidden_size)
    public static func process(_ image: CGImage, with visionModel: MLModel) throws -> MLMultiArray {
        let (pixelValues, positionIDs) = createPatches(from: image)

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pixelValues),
            "pixel_position_ids": MLFeatureValue(multiArray: positionIDs),
        ])

        let output = try visionModel.prediction(from: input)
        guard let features = output.featureValue(for: "image_features")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        return features
    }

    /// Extract a single image feature token from the vision output.
    public static func sliceFeature(_ features: MLMultiArray, at index: Int, hiddenSize: Int) -> MLMultiArray {
        let result = try! MLMultiArray(shape: [1, 1, NSNumber(value: hiddenSize)], dataType: .float16)
        let srcPtr = features.dataPointer.bindMemory(to: UInt16.self, capacity: features.count)
        let dstPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: hiddenSize)
        memcpy(dstPtr, srcPtr.advanced(by: index * hiddenSize), hiddenSize * MemoryLayout<UInt16>.stride)
        return result
    }

    // MARK: - Private

    private static func createPatches(from image: CGImage) -> (MLMultiArray, MLMultiArray) {
        let patchSize = 16
        let totalPatches = 2520
        let patchDim = 3 * patchSize * patchSize  // 768

        // Resize image to standard grid
        let targetSize = 896
        let w = targetSize, h = targetSize

        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var pixels = [UInt8](repeating: 0, count: w * h * 4)
        let context = CGContext(data: &pixels, width: w, height: h,
                                bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

        let pixelValues = try! MLMultiArray(
            shape: [1, NSNumber(value: totalPatches), NSNumber(value: patchDim)],
            dataType: .float32
        )
        let positionIDs = try! MLMultiArray(
            shape: [1, NSNumber(value: totalPatches), 2],
            dataType: .int32
        )

        let pvPtr = pixelValues.dataPointer.bindMemory(to: Float.self, capacity: totalPatches * patchDim)
        let pidPtr = positionIDs.dataPointer.bindMemory(to: Int32.self, capacity: totalPatches * 2)

        let patchesPerSide = w / patchSize
        var patchIdx = 0

        for py in 0..<patchesPerSide {
            for px in 0..<patchesPerSide {
                guard patchIdx < totalPatches else { break }
                var offset = patchIdx * patchDim
                for dy in 0..<patchSize {
                    for dx in 0..<patchSize {
                        let ix = px * patchSize + dx
                        let iy = py * patchSize + dy
                        let pixelOffset = (iy * w + ix) * 4
                        pvPtr[offset] = Float(pixels[pixelOffset]) / 255.0
                        pvPtr[offset + 1] = Float(pixels[pixelOffset + 1]) / 255.0
                        pvPtr[offset + 2] = Float(pixels[pixelOffset + 2]) / 255.0
                        offset += 3
                    }
                }
                pidPtr[patchIdx * 2] = Int32(px)
                pidPtr[patchIdx * 2 + 1] = Int32(py)
                patchIdx += 1
            }
        }

        // Padding
        for i in patchIdx..<totalPatches {
            pidPtr[i * 2] = -1
            pidPtr[i * 2 + 1] = -1
        }

        return (pixelValues, positionIDs)
    }
}
