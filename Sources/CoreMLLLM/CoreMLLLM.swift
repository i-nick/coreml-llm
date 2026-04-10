import CoreML
import Foundation
import Tokenizers

/// On-device LLM inference using CoreML with ANE optimization.
///
/// Supports both monolithic models (single .mlpackage) and chunked SWA models
/// (Gemma 4 E2B with 4 decode + 4 prefill chunks + external embeddings).
///
/// ```swift
/// let llm = try await CoreMLLLM.load(from: modelDirectory)
/// let answer = try await llm.generate("What is the capital of France?")
/// // → "The capital of France is **Paris**."
///
/// for await token in try await llm.stream("Tell me a story") {
///     print(token, terminator: "")
/// }
/// ```
public final class CoreMLLLM: @unchecked Sendable {
    private let tokenizer: any Tokenizer
    private let config: ModelConfig

    // Engine: exactly one of these is non-nil.
    private var chunkedEngine: ChunkedEngine?
    private var monolithicModel: MLModel?
    private var monolithicState: MLState?

    // Vision (lazy loaded to save memory)
    private var visionModel: MLModel?
    private var visionModelURL: URL?
    private var visionConfig: MLModelConfiguration?

    // Audio (lazy loaded to save memory)
    private var audioModel: MLModel?
    private var audioModelURL: URL?
    private var audioConfig: MLModelConfiguration?
    private var melFilterbank: [Float]?
    private var audioMelFrames: Int = 200
    private var audioNumTokens: Int = 50

    private init(config: ModelConfig, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    // MARK: - Public API

    /// Load a model from a local directory.
    ///
    /// Auto-detects layout:
    /// - If `chunk1.mlmodelc` exists → chunked SWA engine (Gemma 4 E2B)
    /// - Otherwise → monolithic model (`model.mlpackage` / `model.mlmodelc`)
    ///
    /// - Parameters:
    ///   - directory: Folder containing model files, embeddings, config
    ///   - computeUnits: CoreML compute units (default: `.cpuAndNeuralEngine`)
    public static func load(
        from directory: URL,
        computeUnits: MLComputeUnits = .cpuAndNeuralEngine
    ) async throws -> CoreMLLLM {
        let config = try ModelConfig.load(from: directory)

        // Tokenizer
        let tokDir = directory.appendingPathComponent("hf_model")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)

        let llm = CoreMLLLM(config: config, tokenizer: tokenizer)

        // Auto-detect: chunked or monolithic
        let isChunked = FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("chunk1.mlmodelc").path)
            || FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("chunk1.mlpackage").path)

        if isChunked {
            llm.chunkedEngine = try await ChunkedEngine.load(
                from: directory, config: config, computeUnits: computeUnits)
        } else {
            let mlConfig = MLModelConfiguration()
            mlConfig.computeUnits = computeUnits
            let modelURL = directory.appendingPathComponent("model.mlmodelc")
            if FileManager.default.fileExists(atPath: modelURL.path) {
                llm.monolithicModel = try MLModel(contentsOf: modelURL, configuration: mlConfig)
            } else {
                let pkgURL = directory.appendingPathComponent("model.mlpackage")
                let compiled = try await MLModel.compileModel(at: pkgURL)
                llm.monolithicModel = try MLModel(contentsOf: compiled, configuration: mlConfig)
            }
            llm.monolithicState = llm.monolithicModel?.makeState()
        }

        // Vision model (optional, lazy loaded on first image)
        let visionCompiled = directory.appendingPathComponent("vision.mlmodelc")
        let visionPkg = directory.appendingPathComponent("vision.mlpackage")
        if FileManager.default.fileExists(atPath: visionCompiled.path) {
            llm.visionModelURL = visionCompiled
        } else if FileManager.default.fileExists(atPath: visionPkg.path) {
            llm.visionModelURL = visionPkg
        }
        if llm.visionModelURL != nil {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndGPU
            llm.visionConfig = cfg
        }

        // Audio model (optional, lazy loaded on first audio)
        let audioCompiled = directory.appendingPathComponent("audio.mlmodelc")
        let audioPkg = directory.appendingPathComponent("audio.mlpackage")
        if FileManager.default.fileExists(atPath: audioCompiled.path) {
            llm.audioModelURL = audioCompiled
        } else if FileManager.default.fileExists(atPath: audioPkg.path) {
            llm.audioModelURL = audioPkg
        }
        if llm.audioModelURL != nil {
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .cpuAndGPU
            llm.audioConfig = cfg

            // Load mel filterbank
            let melURL = directory.appendingPathComponent("mel_filterbank.bin")
            if FileManager.default.fileExists(atPath: melURL.path) {
                llm.melFilterbank = try? AudioProcessor.loadMelFilterbank(from: melURL)
            }

            // Load audio config for frame/token counts
            let audioConfURL = directory.appendingPathComponent("audio_config.json")
            if let data = try? Data(contentsOf: audioConfURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                llm.audioMelFrames = json["mel_frames"] as? Int ?? 200
                llm.audioNumTokens = json["num_tokens"] as? Int ?? 50
            }
        }

        return llm
    }

    /// Whether this model supports image input.
    public var supportsVision: Bool { visionModelURL != nil }

    /// Whether this model supports audio input.
    public var supportsAudio: Bool { audioModelURL != nil && melFilterbank != nil }

    /// Model name from config.
    public var modelName: String { config.modelName }

    /// Generate a complete response.
    public func generate(_ prompt: String, image: CGImage? = nil,
                         audio: [Float]? = nil,
                         maxTokens: Int = 256) async throws -> String {
        var result = ""
        for await token in try await stream(prompt, image: image, audio: audio,
                                             maxTokens: maxTokens) {
            result += token
        }
        return result
    }

    /// Stream tokens as they're generated.
    public func stream(_ prompt: String, image: CGImage? = nil,
                       audio: [Float]? = nil,
                       maxTokens: Int = 256) async throws -> AsyncStream<String> {
        let chatPrompt = buildPrompt(prompt, hasImage: image != nil,
                                     hasAudio: audio != nil)
        let tokenIDs = tokenizer.encode(text: chatPrompt)

        var imageFeatures: MLMultiArray?
        if let image {
            imageFeatures = try processImage(image)
        }

        var audioFeatures: MLMultiArray?
        if let audio {
            audioFeatures = try processAudio(audio)
        }

        reset()

        let mutableSelf = self
        let imgFeats = imageFeatures
        let audFeats = audioFeatures
        let tokens = tokenIDs
        let audTokenCount = mutableSelf.audioNumTokens

        return AsyncStream { continuation in
            Task {
                do {
                    let IMAGE_TOKEN_ID = 258880
                    let AUDIO_TOKEN_ID = 258881
                    var imageIdx = 0
                    var audioIdx = 0
                    var nextID = 0

                    // Helper: resolve multimodal embedding for a token
                    func multimodalEmbedding(for tid: Int) -> MLMultiArray? {
                        if tid == IMAGE_TOKEN_ID, let f = imgFeats, imageIdx < 256 {
                            let emb = engine?.sliceFeature(f, at: imageIdx)
                                ?? ImageProcessor.sliceFeature(f, at: imageIdx,
                                    hiddenSize: mutableSelf.config.hiddenSize)
                            imageIdx += 1
                            return emb
                        }
                        if tid == AUDIO_TOKEN_ID, let f = audFeats, audioIdx < audTokenCount {
                            let emb = engine?.sliceFeature(f, at: audioIdx)
                                ?? AudioProcessor.sliceFeature(f, at: audioIdx,
                                    hiddenSize: mutableSelf.config.hiddenSize)
                            audioIdx += 1
                            return emb
                        }
                        return nil
                    }

                    let engine = mutableSelf.chunkedEngine

                    if let engine {
                        // Chunked path: hybrid prefill + decode
                        let prefillLen = min(tokens.count, engine.prefillN)
                        let useHybrid = engine.hasPrefill && prefillLen > 0

                        if useHybrid {
                            try autoreleasepool {
                                let batch = Array(tokens[0..<prefillLen])
                                // Prefill with image features (audio uses per-token path)
                                nextID = try engine.runPrefill(tokenIDs: batch,
                                                               imageFeatures: imgFeats)
                            }
                            imageIdx = tokens[0..<prefillLen].filter { $0 == IMAGE_TOKEN_ID }.count
                            audioIdx = tokens[0..<prefillLen].filter { $0 == AUDIO_TOKEN_ID }.count
                            engine.currentPosition = prefillLen

                            // Per-token decode for remaining prompt tokens
                            for step in prefillLen..<tokens.count {
                                let tid = tokens[step]
                                try autoreleasepool {
                                    if let emb = multimodalEmbedding(for: tid) {
                                        nextID = try engine.predictStep(tokenID: 0, position: step,
                                                                         imageEmbedding: emb)
                                    } else {
                                        nextID = try engine.predictStep(tokenID: tid, position: step)
                                    }
                                }
                                engine.currentPosition = step + 1
                            }
                        } else {
                            for (step, tid) in tokens.enumerated() {
                                try autoreleasepool {
                                    if let emb = multimodalEmbedding(for: tid) {
                                        nextID = try engine.predictStep(tokenID: 0, position: step,
                                                                         imageEmbedding: emb)
                                    } else {
                                        nextID = try engine.predictStep(tokenID: tid, position: step)
                                    }
                                }
                                engine.currentPosition = step + 1
                            }
                        }

                        // Decode loop
                        let eosIDs: Set<Int> = [1, 106, 151645]
                        for _ in 0..<maxTokens {
                            if eosIDs.contains(nextID) { break }
                            if engine.currentPosition >= mutableSelf.config.contextLength { break }
                            let text = mutableSelf.tokenizer.decode(tokens: [nextID])
                            continuation.yield(text)
                            try autoreleasepool {
                                nextID = try engine.predictStep(tokenID: nextID,
                                                                 position: engine.currentPosition)
                            }
                            engine.currentPosition += 1
                        }
                    } else {
                        // Monolithic path
                        for (step, tid) in tokens.enumerated() {
                            if let emb = multimodalEmbedding(for: tid) {
                                nextID = try mutableSelf.predictMonolithic(
                                    tokenID: 0, position: step, imageEmbedding: emb)
                            } else {
                                nextID = try mutableSelf.predictMonolithic(
                                    tokenID: tid, position: step)
                            }
                        }
                        let eosIDs: Set<Int> = [1, 106, 151645]
                        var pos = tokens.count
                        for _ in 0..<maxTokens {
                            if eosIDs.contains(nextID) { break }
                            let text = mutableSelf.tokenizer.decode(tokens: [nextID])
                            continuation.yield(text)
                            nextID = try mutableSelf.predictMonolithic(
                                tokenID: nextID, position: pos)
                            pos += 1
                        }
                    }
                } catch {}
                continuation.finish()
            }
        }
    }

    /// Reset conversation state (clears KV cache).
    public func reset() {
        if let engine = chunkedEngine {
            engine.reset()
        } else {
            monolithicState = monolithicModel?.makeState()
        }
    }

    // MARK: - Private: monolithic prediction

    private func predictMonolithic(tokenID: Int, position: Int,
                                    imageEmbedding: MLMultiArray? = nil) throws -> Int {
        guard let model = monolithicModel, let state = monolithicState else {
            throw CoreMLLLMError.predictionFailed
        }
        let ctx = config.contextLength
        let hs = config.hiddenSize

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
        up[min(position, ctx - 1)] = 0x3C00

        var dict: [String: MLFeatureValue] = [
            "input_ids": MLFeatureValue(multiArray: ids),
            "position_ids": MLFeatureValue(multiArray: pos),
            "causal_mask": MLFeatureValue(multiArray: mask),
            "update_mask": MLFeatureValue(multiArray: umask),
        ]

        let inputNames = model.modelDescription.inputDescriptionsByName
        if inputNames["per_layer_combined"] != nil, let engine = chunkedEngine {
            let emb = try engine.computePerLayerCombined(tokenID: tokenID,
                embedding: try MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16))
            dict["per_layer_combined"] = MLFeatureValue(multiArray: emb)
        }
        if inputNames["image_embedding"] != nil {
            let imgEmb: MLMultiArray
            if let imageEmbedding { imgEmb = imageEmbedding }
            else {
                imgEmb = try MLMultiArray(shape: [1, 1, NSNumber(value: hs)], dataType: .float16)
                memset(imgEmb.dataPointer, 0, hs * MemoryLayout<UInt16>.stride)
            }
            dict["image_embedding"] = MLFeatureValue(multiArray: imgEmb)
        }

        let output = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: dict),
                                           using: state)
        return output.featureValue(for: "token_id")!.multiArrayValue![0].intValue
    }

    // MARK: - Private: vision

    private func processImage(_ image: CGImage) throws -> MLMultiArray {
        if visionModel == nil, let url = visionModelURL, let cfg = visionConfig {
            visionModel = try MLModel(contentsOf: url, configuration: cfg)
        }
        guard let vm = visionModel else { throw CoreMLLLMError.visionNotAvailable }
        return try ImageProcessor.process(image, with: vm)
    }

    // MARK: - Private: audio

    private func processAudio(_ samples: [Float]) throws -> MLMultiArray {
        if audioModel == nil, let url = audioModelURL, let cfg = audioConfig {
            audioModel = try MLModel(contentsOf: url, configuration: cfg)
        }
        guard let am = audioModel else { throw CoreMLLLMError.audioNotAvailable }
        guard let mel = melFilterbank else { throw CoreMLLLMError.audioNotAvailable }
        return try AudioProcessor.process(samples, with: am,
                                           melFilterbank: mel,
                                           targetFrames: audioMelFrames)
    }

    // MARK: - Private: prompt building

    private func buildPrompt(_ text: String, hasImage: Bool,
                              hasAudio: Bool = false) -> String {
        if config.architecture.hasPrefix("qwen") {
            return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\(text)<|im_end|>\n<|im_start|>assistant\n"
        }
        var mediaPrefix = ""
        if hasImage {
            let imageTokens = String(repeating: "<|image|>", count: 256)
            mediaPrefix += "<|image>\(imageTokens)<image|>\n"
        }
        if hasAudio {
            let audioTokens = String(repeating: "<|audio|>", count: audioNumTokens)
            mediaPrefix += "<|audio>\(audioTokens)<audio|>\n"
        }
        return "<bos><|turn>user\n\(mediaPrefix)\(text)<turn|>\n<|turn>model\n"
    }
}

// MARK: - Error types

public enum CoreMLLLMError: LocalizedError {
    case configNotFound
    case predictionFailed
    case modelNotFound(String)
    case prefillNotAvailable
    case visionNotAvailable
    case audioNotAvailable

    public var errorDescription: String? {
        switch self {
        case .configNotFound: return "model_config.json not found"
        case .predictionFailed: return "Model prediction failed"
        case .modelNotFound(let name): return "Model file not found: \(name)"
        case .prefillNotAvailable: return "Prefill chunks not loaded"
        case .visionNotAvailable: return "Vision model not available"
        case .audioNotAvailable: return "Audio model not available"
        }
    }
}
