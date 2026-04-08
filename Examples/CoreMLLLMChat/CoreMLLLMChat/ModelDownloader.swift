import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Downloads and caches CoreML models from GitHub Releases.
@Observable
final class ModelDownloader {
    var isDownloading = false
    var progress: Double = 0
    var status = ""
    var availableModels: [ModelInfo] = ModelInfo.defaults

    private let fileManager = FileManager.default

    struct ModelInfo: Identifiable {
        let id: String
        let name: String
        let size: String
        let downloadURL: String
        let folderName: String

        static let defaults: [ModelInfo] = [
            ModelInfo(
                id: "gemma4-e2b",
                name: "Gemma 4 E2B (Multimodal)",
                size: "2.7 GB",
                downloadURL: "https://huggingface.co/mlboydaisuke/gemma-4-E2B-coreml/resolve/main",
                folderName: "gemma4-e2b"
            ),
            ModelInfo(
                id: "qwen2.5-0.5b",
                name: "Qwen2.5 0.5B (Text)",
                size: "309 MB",
                downloadURL: "https://github.com/john-rocky/CoreML-LLM/releases/download/v0.1.0/qwen2.5-0.5b-coreml.zip",
                folderName: "qwen2.5-0.5b"
            ),
        ]
    }

    func isDownloaded(_ model: ModelInfo) -> Bool {
        localModelURL(for: model) != nil
    }

    func localModelURL(for model: ModelInfo) -> URL? {
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        let pkg = dir.appendingPathComponent("model.mlpackage")
        return fileManager.fileExists(atPath: pkg.path) ? pkg : nil
    }

    func download(_ model: ModelInfo) async throws -> URL {
        if let existing = localModelURL(for: model) { return existing }

        isDownloading = true
        progress = 0
        status = "Downloading \(model.name)..."
        defer { isDownloading = false }

        let destDir = modelsDirectory.appendingPathComponent(model.folderName)
        try? fileManager.removeItem(at: destDir)
        try fileManager.createDirectory(at: destDir, withIntermediateDirectories: true)

        if model.downloadURL.contains("huggingface.co") {
            // HuggingFace: download individual files
            try await downloadFromHuggingFace(model, to: destDir)
        } else {
            // GitHub Releases: download ZIP
            guard let url = URL(string: model.downloadURL) else {
                throw DownloadError.invalidURL
            }
            let tempZip = try await downloadFile(url)
            status = "Extracting..."
            try unzipFile(tempZip, to: destDir)
            try? fileManager.removeItem(at: tempZip)
        }

        guard let result = localModelURL(for: model) else {
            throw DownloadError.extractionFailed
        }

        status = "Ready"
        progress = 1.0
        return result
    }

    private func downloadFromHuggingFace(_ model: ModelInfo, to destDir: URL) async throws {
        let base = model.downloadURL
        let files: [(String, String)] = [
            ("model.mlpackage/Manifest.json", "model.mlpackage/Manifest.json"),
            ("model.mlpackage/Data/com.apple.CoreML/model.mlmodel", "model.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
            ("model.mlpackage/Data/com.apple.CoreML/weights/weight.bin", "model.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
            ("model_config.json", "model_config.json"),
            ("hf_model/tokenizer.json", "hf_model/tokenizer.json"),
        ]

        // Add vision files for multimodal models
        let visionFiles: [(String, String)] = model.id.contains("gemma") ? [
            ("vision.mlpackage/Manifest.json", "vision.mlpackage/Manifest.json"),
            ("vision.mlpackage/Data/com.apple.CoreML/model.mlmodel", "vision.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
            ("vision.mlpackage/Data/com.apple.CoreML/weights/weight.bin", "vision.mlpackage/Data/com.apple.CoreML/weights/weight.bin"),
        ] : []

        let allFiles = files + visionFiles
        for (i, (remotePath, localPath)) in allFiles.enumerated() {
            let fileName = (localPath as NSString).lastPathComponent
            status = "Downloading \(fileName)... (\(i+1)/\(allFiles.count))"

            guard let url = URL(string: "\(base)/\(remotePath)") else { continue }
            let destFile = destDir.appendingPathComponent(localPath)
            try fileManager.createDirectory(at: destFile.deletingLastPathComponent(), withIntermediateDirectories: true)

            let tempFile = try await downloadFile(url)
            try? fileManager.removeItem(at: destFile)
            try fileManager.moveItem(at: tempFile, to: destFile)

            progress = Double(i + 1) / Double(allFiles.count)
        }
    }

    func delete(_ model: ModelInfo) throws {
        let dir = modelsDirectory.appendingPathComponent(model.folderName)
        if fileManager.fileExists(atPath: dir.path) {
            try fileManager.removeItem(at: dir)
        }
    }

    // MARK: - Private

    private var modelsDirectory: URL {
        let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("Models")
    }

    private func downloadFile(_ url: URL) async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            let session = URLSession(configuration: .default, delegate: ProgressTracker { [weak self] p in
                Task { @MainActor in self?.progress = p }
            }, delegateQueue: nil)

            let task = session.downloadTask(with: url) { tempURL, _, error in
                if let error { continuation.resume(throwing: error); return }
                guard let tempURL else { continuation.resume(throwing: DownloadError.extractionFailed); return }
                let dest = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".zip")
                do {
                    try FileManager.default.moveItem(at: tempURL, to: dest)
                    continuation.resume(returning: dest)
                } catch { continuation.resume(throwing: error) }
            }
            task.resume()
        }
    }

    private func unzipFile(_ zipURL: URL, to destDir: URL) throws {
        // Use /usr/bin/ditto (available on both macOS and iOS simulators)
        // For real iOS devices, we use a minimal Swift ZIP implementation
        #if targetEnvironment(simulator) || os(macOS)
        let proc = Process()
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/ditto")
        proc.arguments = ["-xk", zipURL.path, destDir.path]
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = FileHandle.nullDevice
        try proc.run()
        proc.waitUntilExit()
        #else
        // On-device: use Foundation's built-in ZIP reading
        // ZIP files are just PKZip format — read central directory and extract
        try extractZipNative(from: zipURL, to: destDir)
        #endif
    }

    #if !targetEnvironment(simulator) && !os(macOS)
    private func extractZipNative(from zipURL: URL, to destDir: URL) throws {
        // Use Apple's Archive framework for ZIP extraction (iOS 16+)
        // Fallback: manual ZIP parsing
        let data = try Data(contentsOf: zipURL)

        // Find End of Central Directory record
        guard data.count > 22 else { throw DownloadError.extractionFailed }

        var eocdOffset = data.count - 22
        while eocdOffset >= 0 {
            if data[eocdOffset] == 0x50 && data[eocdOffset+1] == 0x4B &&
               data[eocdOffset+2] == 0x05 && data[eocdOffset+3] == 0x06 {
                break
            }
            eocdOffset -= 1
        }
        guard eocdOffset >= 0 else { throw DownloadError.extractionFailed }

        // Parse central directory
        let cdOffset = Int(data[eocdOffset+16..<eocdOffset+20].withUnsafeBytes { $0.load(as: UInt32.self) })
        let cdCount = Int(data[eocdOffset+10..<eocdOffset+12].withUnsafeBytes { $0.load(as: UInt16.self) })

        var pos = cdOffset
        for _ in 0..<cdCount {
            guard data[pos] == 0x50, data[pos+1] == 0x4B, data[pos+2] == 0x01, data[pos+3] == 0x02 else { break }

            let method = Int(data[pos+10..<pos+12].withUnsafeBytes { $0.load(as: UInt16.self) })
            let compSize = Int(data[pos+20..<pos+24].withUnsafeBytes { $0.load(as: UInt32.self) })
            let uncompSize = Int(data[pos+24..<pos+28].withUnsafeBytes { $0.load(as: UInt32.self) })
            let nameLen = Int(data[pos+28..<pos+30].withUnsafeBytes { $0.load(as: UInt16.self) })
            let extraLen = Int(data[pos+30..<pos+32].withUnsafeBytes { $0.load(as: UInt16.self) })
            let commentLen = Int(data[pos+32..<pos+34].withUnsafeBytes { $0.load(as: UInt16.self) })
            let localOffset = Int(data[pos+42..<pos+46].withUnsafeBytes { $0.load(as: UInt32.self) })

            let nameData = data[pos+46..<pos+46+nameLen]
            let name = String(data: nameData, encoding: .utf8) ?? ""

            let destPath = destDir.appendingPathComponent(name)

            if name.hasSuffix("/") {
                try fileManager.createDirectory(at: destPath, withIntermediateDirectories: true)
            } else {
                try fileManager.createDirectory(at: destPath.deletingLastPathComponent(), withIntermediateDirectories: true)

                // Read local file header to find data offset
                let localNameLen = Int(data[localOffset+26..<localOffset+28].withUnsafeBytes { $0.load(as: UInt16.self) })
                let localExtraLen = Int(data[localOffset+28..<localOffset+30].withUnsafeBytes { $0.load(as: UInt16.self) })
                let dataStart = localOffset + 30 + localNameLen + localExtraLen

                if method == 0 {
                    // Stored (no compression)
                    let fileData = data[dataStart..<dataStart+uncompSize]
                    try Data(fileData).write(to: destPath)
                } else {
                    // Compressed — for our models, we use zip -0 (stored), so this shouldn't happen
                    throw DownloadError.extractionFailed
                }
            }

            pos += 46 + nameLen + extraLen + commentLen
        }
    }
    #endif
}

private final class ProgressTracker: NSObject, URLSessionDownloadDelegate {
    let onProgress: (Double) -> Void
    init(onProgress: @escaping (Double) -> Void) { self.onProgress = onProgress }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        guard totalBytesExpectedToWrite > 0 else { return }
        onProgress(Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
    }
}

enum DownloadError: LocalizedError {
    case invalidURL, extractionFailed
    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid download URL"
        case .extractionFailed: return "Failed to extract model"
        }
    }
}
