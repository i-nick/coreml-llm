import Foundation
import CoreML

/// Memory-mapped embedding lookup for large vocabulary tables.
/// Reads int8 quantized embeddings from disk without loading entire table into RAM.
final class EmbeddingLookup {
    private let data: Data  // memory-mapped
    private let scales: Data
    private let vocabSize: Int
    private let dim: Int
    private let scale: Float  // embedding scale factor

    /// Load an int8 quantized embedding table.
    /// - Parameters:
    ///   - dataURL: Path to q8.bin file (vocabSize × dim, int8)
    ///   - scalesURL: Path to scales.bin file (vocabSize × 1, float16)
    ///   - vocabSize: Vocabulary size
    ///   - dim: Embedding dimension
    ///   - scale: Multiply result by this (e.g., sqrt(hidden_size) for token embeddings)
    init(dataURL: URL, scalesURL: URL, vocabSize: Int, dim: Int, scale: Float = 1.0) throws {
        self.data = try Data(contentsOf: dataURL, options: .mappedIfSafe)
        self.scales = try Data(contentsOf: scalesURL, options: .mappedIfSafe)
        self.vocabSize = vocabSize
        self.dim = dim
        self.scale = scale
    }

    /// Look up embedding for a single token and return as float16 MLMultiArray.
    func lookup(_ tokenID: Int, shape: [NSNumber]) throws -> MLMultiArray {
        let result = try MLMultiArray(shape: shape, dataType: .float16)
        let dstPtr = result.dataPointer.bindMemory(to: UInt16.self, capacity: dim)

        let rowOffset = tokenID * dim
        let scaleOffset = tokenID * 2  // float16 = 2 bytes

        data.withUnsafeBytes { rawPtr in
            let int8Ptr = rawPtr.baseAddress!.assumingMemoryBound(to: Int8.self).advanced(by: rowOffset)

            scales.withUnsafeBytes { scaleRaw in
                let scalePtr = scaleRaw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                let rowScaleBits = scalePtr[tokenID]

                // Convert float16 scale to float32
                var rowScale = float16ToFloat32(rowScaleBits) / 127.0 * scale

                for i in 0..<dim {
                    let val = Float(int8Ptr[i]) * rowScale
                    dstPtr[i] = float32ToFloat16(val)
                }
            }
        }

        return result
    }

    /// Look up and return as raw float16 array (for per-layer combined computation).
    func lookupRaw(_ tokenID: Int) -> [UInt16] {
        var result = [UInt16](repeating: 0, count: dim)
        let rowOffset = tokenID * dim

        data.withUnsafeBytes { rawPtr in
            let int8Ptr = rawPtr.baseAddress!.assumingMemoryBound(to: Int8.self).advanced(by: rowOffset)

            scales.withUnsafeBytes { scaleRaw in
                let scalePtr = scaleRaw.baseAddress!.assumingMemoryBound(to: UInt16.self)
                let rowScale = float16ToFloat32(scalePtr[tokenID]) / 127.0 * scale

                for i in 0..<dim {
                    result[i] = float32ToFloat16(Float(int8Ptr[i]) * rowScale)
                }
            }
        }

        return result
    }

    // MARK: - Float16 Conversion

    private func float16ToFloat32(_ bits: UInt16) -> Float {
        var f: Float = 0
        var h = bits
        withUnsafePointer(to: &h) { hPtr in
            withUnsafeMutablePointer(to: &f) { fPtr in
                // Use vImageConvert if available, otherwise manual
                let sign: UInt32 = UInt32(bits >> 15) << 31
                let exp = UInt32((bits >> 10) & 0x1F)
                let frac = UInt32(bits & 0x3FF)

                if exp == 0 {
                    if frac == 0 { fPtr.pointee = Float(bitPattern: sign); return }
                    // Denormalized
                    var e: UInt32 = 113
                    var f = frac
                    while f & 0x400 == 0 { f <<= 1; e -= 1 }
                    f &= 0x3FF
                    fPtr.pointee = Float(bitPattern: sign | ((e + 1) << 23) | (f << 13))
                } else if exp == 31 {
                    fPtr.pointee = Float(bitPattern: sign | 0x7F800000 | (frac << 13))
                } else {
                    fPtr.pointee = Float(bitPattern: sign | ((exp + 112) << 23) | (frac << 13))
                }
            }
        }
        return f
    }

    private func float32ToFloat16(_ value: Float) -> UInt16 {
        let bits = value.bitPattern
        let sign = UInt16((bits >> 16) & 0x8000)
        let exp = Int((bits >> 23) & 0xFF) - 127 + 15
        let frac = UInt16((bits >> 13) & 0x3FF)

        if exp <= 0 { return sign }
        if exp >= 31 { return sign | 0x7C00 }
        return sign | UInt16(exp) << 10 | frac
    }
}
