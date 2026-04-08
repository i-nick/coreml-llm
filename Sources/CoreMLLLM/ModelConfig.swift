import Foundation

/// Model configuration loaded from model_config.json.
public struct ModelConfig: Sendable {
    public let modelName: String
    public let architecture: String
    public let hiddenSize: Int
    public let contextLength: Int
    public let vocabSize: Int
    public let bosTokenId: Int
    public let eosTokenId: Int

    static func load(from directory: URL) throws -> ModelConfig {
        let url = directory.appendingPathComponent("model_config.json")
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw CoreMLLLMError.configNotFound
        }
        return ModelConfig(
            modelName: json["model_name"] as? String ?? "Model",
            architecture: json["architecture"] as? String ?? "unknown",
            hiddenSize: json["hidden_size"] as? Int ?? 1536,
            contextLength: json["context_length"] as? Int ?? 512,
            vocabSize: json["vocab_size"] as? Int ?? 262144,
            bosTokenId: json["bos_token_id"] as? Int ?? 2,
            eosTokenId: json["eos_token_id"] as? Int ?? 1
        )
    }
}
