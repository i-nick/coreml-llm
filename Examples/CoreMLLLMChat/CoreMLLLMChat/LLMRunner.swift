import CoreML
import Foundation
import Tokenizers

/// Manages CoreML LLM model loading and inference.
/// Uses the same API as the CoreMLLLM Swift Package.
@Observable
final class LLMRunner {
    var isLoaded = false
    var isGenerating = false
    var loadingStatus = "Not loaded"
    var tokensPerSecond: Double = 0
    var modelName = ""
    var hasVision = false

    private var model: MLModel?
    private var visionModel: MLModel?
    private var state: MLState?
    private var tokenizer: (any Tokenizer)?
    private var contextLength = 512
    private var hiddenSize = 1536
    private var architecture = "gemma4"
    private var currentPosition = 0

    func loadModel(from url: URL) async throws {
        let folder = url.deletingLastPathComponent()

        // Config
        loadingStatus = "Reading config..."
        let configURL = folder.appendingPathComponent("model_config.json")
        if let data = try? Data(contentsOf: configURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            contextLength = json["context_length"] as? Int ?? 512
            architecture = json["architecture"] as? String ?? "gemma4"
            hiddenSize = json["hidden_size"] as? Int ?? 1536
            modelName = json["model_name"] as? String ?? "Model"
        }

        // Main model
        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = .cpuAndNeuralEngine

        // Load main model (.mlmodelc = pre-compiled, .mlpackage = needs compile)
        let modelcURL = folder.appendingPathComponent("model.mlmodelc")
        if FileManager.default.fileExists(atPath: modelcURL.path) {
            loadingStatus = "Loading model..."
            model = try MLModel(contentsOf: modelcURL, configuration: mlConfig)
        } else {
            loadingStatus = "Compiling model..."
            let compiled = try await MLModel.compileModel(at: url)
            model = try MLModel(contentsOf: compiled, configuration: mlConfig)
        }
        state = model?.makeState()

        // Vision model
        let visionCompiledURL = folder.appendingPathComponent("vision.mlmodelc")
        let visionPackageURL = folder.appendingPathComponent("vision.mlpackage")
        if FileManager.default.fileExists(atPath: visionCompiledURL.path) {
            loadingStatus = "Loading vision..."
            visionModel = try MLModel(contentsOf: visionCompiledURL, configuration: mlConfig)
            hasVision = true
        } else if FileManager.default.fileExists(atPath: visionPackageURL.path) {
            loadingStatus = "Compiling vision..."
            let compiledV = try await MLModel.compileModel(at: visionPackageURL)
            visionModel = try MLModel(contentsOf: compiledV, configuration: mlConfig)
            hasVision = true
        }

        // Tokenizer
        loadingStatus = "Loading tokenizer..."
        let tokDir = folder.appendingPathComponent("hf_model")
        tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)

        isLoaded = true
        currentPosition = 0
        loadingStatus = "Ready"
    }

    func generate(messages: [ChatMessage], image: CGImage? = nil) async throws -> AsyncStream<String> {
        guard model != nil, state != nil, tokenizer != nil else {
            throw NSError(domain: "LLMRunner", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not loaded"])
        }

        isGenerating = true
        let prompt = buildPrompt(messages: messages, hasImage: image != nil)
        let tokenIDs = tokenizer!.encode(text: prompt)

        // Vision
        var imageFeatures: MLMultiArray?
        if let image, let vm = visionModel {
            imageFeatures = try processImage(image, with: vm)
        }

        state = model!.makeState()
        currentPosition = 0

        return AsyncStream { continuation in
            Task {
                defer { self.isGenerating = false }
                do {
                    let IMAGE_TOKEN_ID = 258880
                    var imageIdx = 0
                    var nextID = 0

                    // Prefill
                    for (step, tid) in tokenIDs.enumerated() {
                        if tid == IMAGE_TOKEN_ID, let feats = imageFeatures, imageIdx < 256 {
                            let imgEmb = self.sliceFeature(feats, at: imageIdx)
                            nextID = try self.predict(tokenID: 0, position: step, imageEmbedding: imgEmb)
                            imageIdx += 1
                        } else {
                            nextID = try self.predict(tokenID: tid, position: step)
                        }
                        self.currentPosition = step + 1
                    }

                    // Decode
                    let startTime = CFAbsoluteTimeGetCurrent()
                    var tokenCount = 0
                    let eosIDs: Set<Int> = [1, 106, 151645]

                    for _ in 0..<256 {
                        if eosIDs.contains(nextID) { break }
                        let text = self.tokenizer!.decode(tokens: [nextID])
                        continuation.yield(text)
                        tokenCount += 1
                        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                        if elapsed > 0 { self.tokensPerSecond = Double(tokenCount) / elapsed }
                        nextID = try self.predict(tokenID: nextID, position: self.currentPosition)
                        self.currentPosition += 1
                    }
                } catch {}
                continuation.finish()
            }
        }
    }

    func resetConversation() {
        state = model?.makeState()
        currentPosition = 0
    }

    // MARK: - Private

    private func predict(tokenID: Int, position: Int, imageEmbedding: MLMultiArray? = nil) throws -> Int {
        guard let model, let state else { throw NSError(domain: "", code: 0) }
        let ctx = contextLength, hs = hiddenSize

        let ids = try MLMultiArray(shape: [1, 1], dataType: .int32)
        ids[[0, 0] as [NSNumber]] = NSNumber(value: Int32(tokenID))
        let pos = try MLMultiArray(shape: [1], dataType: .int32)
        pos[0] = NSNumber(value: Int32(position))
        let mask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: ctx)], dataType: .float16)
        let mp = mask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        for i in 0..<ctx { mp[i] = i <= position ? 0 : 0xFC00 }
        let umask = try MLMultiArray(shape: [1, 1, NSNumber(value: ctx), 1], dataType: .float16)
        let up = umask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        memset(up, 0, ctx * MemoryLayout<UInt16>.stride)
        up[position] = 0x3C00

        let imgEmb: MLMultiArray
        if let imageEmbedding {
            imgEmb = imageEmbedding
        } else {
            imgEmb = try MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16)
            memset(imgEmb.dataPointer, 0, hs * MemoryLayout<UInt16>.stride)
        }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "input_ids": MLFeatureValue(multiArray: ids),
            "position_ids": MLFeatureValue(multiArray: pos),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
            "image_embedding": MLFeatureValue(multiArray: imgEmb),
        ])
        let output = try model.prediction(from: input, using: state)
        return output.featureValue(for: "token_id")!.multiArrayValue![0].intValue
    }

    private func processImage(_ image: CGImage, with visionModel: MLModel) throws -> MLMultiArray {
        let ps = 16, total = 2520, pd = 768
        let sz = 896
        var pixels = [UInt8](repeating: 0, count: sz * sz * 4)
        let ctx = CGContext(data: &pixels, width: sz, height: sz, bitsPerComponent: 8,
                            bytesPerRow: sz * 4, space: CGColorSpaceCreateDeviceRGB(),
                            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: sz, height: sz))

        let pv = try MLMultiArray(shape: [1, NSNumber(value: total), NSNumber(value: pd)], dataType: .float32)
        let pid = try MLMultiArray(shape: [1, NSNumber(value: total), 2], dataType: .int32)
        let pvp = pv.dataPointer.bindMemory(to: Float.self, capacity: total * pd)
        let pidp = pid.dataPointer.bindMemory(to: Int32.self, capacity: total * 2)

        var pi = 0
        let pps = sz / ps
        for py in 0..<pps {
            for px in 0..<pps {
                guard pi < total else { break }
                var o = pi * pd
                for dy in 0..<ps { for dx in 0..<ps {
                    let po = ((py * ps + dy) * sz + (px * ps + dx)) * 4
                    pvp[o] = Float(pixels[po]) / 255; pvp[o+1] = Float(pixels[po+1]) / 255; pvp[o+2] = Float(pixels[po+2]) / 255; o += 3
                }}
                pidp[pi * 2] = Int32(px); pidp[pi * 2 + 1] = Int32(py); pi += 1
            }
        }
        for i in pi..<total { pidp[i * 2] = -1; pidp[i * 2 + 1] = -1 }

        let input = try MLDictionaryFeatureProvider(dictionary: [
            "pixel_values": MLFeatureValue(multiArray: pv),
            "pixel_position_ids": MLFeatureValue(multiArray: pid),
        ])
        return try visionModel.prediction(from: input).featureValue(for: "image_features")!.multiArrayValue!
    }

    private func sliceFeature(_ features: MLMultiArray, at index: Int) -> MLMultiArray {
        let hs = hiddenSize
        let r = try! MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16)
        let s = features.dataPointer.bindMemory(to: UInt16.self, capacity: features.count)
        let d = r.dataPointer.bindMemory(to: UInt16.self, capacity: hs)
        memcpy(d, s.advanced(by: index * hs), hs * MemoryLayout<UInt16>.stride)
        return r
    }

    private func buildPrompt(messages: [ChatMessage], hasImage: Bool) -> String {
        if architecture.hasPrefix("qwen") {
            var p = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            for m in messages {
                if m.role == .user { p += "<|im_start|>user\n\(m.content)<|im_end|>\n" }
                else if m.role == .assistant { p += "<|im_start|>assistant\n\(m.content)<|im_end|>\n" }
            }
            return p + "<|im_start|>assistant\n"
        }
        var p = "<bos>"
        for m in messages {
            if m.role == .user {
                if hasImage {
                    p += "<|turn>user\n\n\n\(String(repeating: "<|image|>", count: 256))\n\n\(m.content)<turn|>\n"
                } else { p += "<|turn>user\n\(m.content)<turn|>\n" }
            } else if m.role == .assistant { p += "<|turn>model\n\(m.content)<turn|>\n" }
        }
        return p + "<|turn>model\n"
    }
}
