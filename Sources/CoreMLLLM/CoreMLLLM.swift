import CoreML
import Foundation
import Tokenizers

/// On-device LLM inference using CoreML with ANE+GPU optimization.
///
/// Supports text generation and multimodal image understanding (Gemma 4).
///
/// ```swift
/// let llm = try await CoreMLLLM.load(from: modelDirectory)
/// let answer = try await llm.generate("What is the capital of France?")
/// // → "The capital of France is **Paris**."
///
/// // With image (Gemma 4 multimodal)
/// let caption = try await llm.generate("Describe this image", image: cgImage)
/// // → "A solid red square centered on a white background."
/// ```
public final class CoreMLLLM: @unchecked Sendable {
    private let model: MLModel
    private let visionModel: MLModel?
    private let tokenizer: any Tokenizer
    private var state: MLState
    private let config: ModelConfig
    private var currentPosition = 0

    private init(model: MLModel, visionModel: MLModel?, tokenizer: any Tokenizer,
                 state: MLState, config: ModelConfig) {
        self.model = model
        self.visionModel = visionModel
        self.tokenizer = tokenizer
        self.state = state
        self.config = config
    }

    // MARK: - Loading

    /// Load a model from a local directory containing model.mlpackage and model_config.json.
    ///
    /// - Parameters:
    ///   - directory: URL to the folder with model files
    ///   - computeUnits: CoreML compute units (default: ANE + CPU)
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> CoreMLLLM {
        let config = try ModelConfig.load(from: directory)

        let mlConfig = MLModelConfiguration()
        mlConfig.computeUnits = computeUnits

        // Compile and load main model
        let modelURL = directory.appendingPathComponent("model.mlpackage")
        let compiledModel = try await MLModel.compileModel(at: modelURL)
        let model = try MLModel(contentsOf: compiledModel, configuration: mlConfig)

        // Vision model (optional)
        var visionModel: MLModel?
        let visionURL = directory.appendingPathComponent("vision.mlpackage")
        if FileManager.default.fileExists(atPath: visionURL.path) {
            let compiledVision = try await MLModel.compileModel(at: visionURL)
            visionModel = try MLModel(contentsOf: compiledVision, configuration: mlConfig)
        }

        // Tokenizer
        let tokenizerDir = directory.appendingPathComponent("hf_model")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        let state = model.makeState()
        return CoreMLLLM(model: model, visionModel: visionModel, tokenizer: tokenizer,
                         state: state, config: config)
    }

    /// Whether this model supports image input.
    public var supportsVision: Bool { visionModel != nil }

    /// Model name from config.
    public var modelName: String { config.modelName }

    // MARK: - Generation

    /// Generate a complete response.
    ///
    /// - Parameters:
    ///   - prompt: Text prompt
    ///   - image: Optional image for multimodal models (Gemma 4)
    ///   - maxTokens: Maximum tokens to generate (default: 256)
    /// - Returns: Generated text
    public func generate(_ prompt: String, image: CGImage? = nil, maxTokens: Int = 256) async throws -> String {
        var result = ""
        for await token in try await stream(prompt, image: image, maxTokens: maxTokens) {
            result += token
        }
        return result
    }

    /// Stream tokens as they're generated.
    ///
    /// - Parameters:
    ///   - prompt: Text prompt
    ///   - image: Optional image for multimodal models
    ///   - maxTokens: Maximum tokens to generate
    /// - Returns: AsyncStream of token strings
    public func stream(_ prompt: String, image: CGImage? = nil, maxTokens: Int = 256) async throws -> AsyncStream<String> {
        let chatPrompt = buildPrompt(prompt, hasImage: image != nil)
        let tokenIDs = tokenizer.encode(text: chatPrompt)

        // Process image if provided
        let imageFeatures: MLMultiArray? = if let image, let vm = visionModel {
            try ImageProcessor.process(image, with: vm)
        } else {
            nil
        }

        reset()

        // Capture everything before entering the async context
        let mutableSelf = self
        let features = imageFeatures
        let tokens = tokenIDs

        return AsyncStream { continuation in
            Task {
                let unsafeSelf = mutableSelf
                let capturedFeatures = features
                let capturedTokenIDs = tokens
                do {
                    // Prefill
                    let IMAGE_TOKEN_ID = 258880
                    let PAD_ID = 0
                    var imageIdx = 0
                    var nextID = 0

                    for (step, tid) in capturedTokenIDs.enumerated() {
                        if tid == IMAGE_TOKEN_ID, let feats = capturedFeatures, imageIdx < 256 {
                            let imgEmb = ImageProcessor.sliceFeature(feats, at: imageIdx, hiddenSize: unsafeSelf.config.hiddenSize)
                            nextID = try unsafeSelf.predict(tokenID: PAD_ID, position: step, imageEmbedding: imgEmb)
                            imageIdx += 1
                        } else {
                            nextID = try unsafeSelf.predict(tokenID: tid, position: step)
                        }
                        unsafeSelf.currentPosition = step + 1
                    }

                    // Decode
                    let eosIDs: Set<Int> = [1, 106, 151645]
                    for _ in 0..<maxTokens {
                        if eosIDs.contains(nextID) { break }

                        let text = unsafeSelf.tokenizer.decode(tokens: [nextID])
                        continuation.yield(text)

                        nextID = try unsafeSelf.predict(tokenID: nextID, position: unsafeSelf.currentPosition)
                        unsafeSelf.currentPosition += 1
                    }
                } catch {}
                continuation.finish()
            }
        }
    }

    /// Reset conversation state (clears KV cache).
    public func reset() {
        state = model.makeState()
        currentPosition = 0
    }

    // MARK: - Private

    private func predict(tokenID: Int, position: Int, imageEmbedding: MLMultiArray? = nil) throws -> Int {
        let ctx = config.contextLength
        let hs = config.hiddenSize

        let inputIDs = try MLMultiArray(shape: [1, 1], dataType: .int32)
        inputIDs[[0, 0] as [NSNumber]] = NSNumber(value: Int32(tokenID))

        let positionIDs = try MLMultiArray(shape: [1], dataType: .int32)
        positionIDs[0] = NSNumber(value: Int32(position))

        let causalMask = try MLMultiArray(shape: [1, 1, 1, NSNumber(value: ctx)], dataType: .float16)
        let maskPtr = causalMask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        for i in 0..<ctx { maskPtr[i] = i <= position ? 0 : 0xFC00 }

        let updateMask = try MLMultiArray(shape: [1, 1, NSNumber(value: ctx), 1], dataType: .float16)
        let updatePtr = updateMask.dataPointer.bindMemory(to: UInt16.self, capacity: ctx)
        memset(updatePtr, 0, ctx * MemoryLayout<UInt16>.stride)
        updatePtr[position] = 0x3C00

        var dict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: inputIDs),
            "position_ids": MLFeatureValue(multiArray: positionIDs),
            "causal_mask": MLFeatureValue(multiArray: causalMask),
            "update_mask": MLFeatureValue(multiArray: updateMask),
        ]

        // Image embedding (zeros for text, vision features for image tokens)
        let imgEmb: MLMultiArray
        if let imageEmbedding {
            imgEmb = imageEmbedding
        } else {
            imgEmb = try MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16)
            memset(imgEmb.dataPointer, 0, hs * MemoryLayout<UInt16>.stride)
        }
        dict["image_embedding"] = MLFeatureValue(multiArray: imgEmb)

        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict), using: state)

        guard let tokenID = output.featureValue(for: "token_id")?.multiArrayValue else {
            throw CoreMLLLMError.predictionFailed
        }
        return tokenID[0].intValue
    }

    private func buildPrompt(_ text: String, hasImage: Bool) -> String {
        if config.architecture.hasPrefix("qwen") {
            return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        }
        // Gemma format
        if hasImage {
            let imageTokens = String(repeating: "<|image|>", count: 256)
            return "<bos><|turn>user\n\n\n\(imageTokens)\n\n\(text)<turn|>\n<|turn>model\n"
        }
        return "<bos><|turn>user\n\(text)<turn|>\n<|turn>model\n"
    }
}

// MARK: - Supporting Types

public enum CoreMLLLMError: LocalizedError {
    case configNotFound
    case predictionFailed

    public var errorDescription: String? {
        switch self {
        case .configNotFound: return "model_config.json not found"
        case .predictionFailed: return "Model prediction failed"
        }
    }
}
