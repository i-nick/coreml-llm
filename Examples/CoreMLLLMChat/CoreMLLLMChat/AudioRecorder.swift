import AVFoundation
import Foundation

/// Simple audio recorder that captures mono 16kHz PCM for the Gemma 4 audio encoder.
@Observable
final class AudioRecorder {
    var isRecording = false
    var duration: TimeInterval = 0

    private var engine: AVAudioEngine?
    private var samples: [Float] = []
    private var timer: Timer?
    private let sampleRate: Double = 16000

    /// Maximum recording duration in seconds (model supports ~2 sec).
    let maxDuration: TimeInterval = 2.0

    /// Start recording from the microphone.
    func start() throws {
        #if os(iOS)
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .measurement)
        try session.setActive(true)
        #endif

        samples = []
        duration = 0

        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Target format: mono 16kHz float32
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else { return }

        // Install a converter tap
        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else { return }

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) {
            [weak self] buffer, _ in
            guard let self else { return }

            let frameCount = AVAudioFrameCount(
                Double(buffer.frameLength) * self.sampleRate / inputFormat.sampleRate)
            guard let convertedBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat, frameCapacity: frameCount
            ) else { return }

            var error: NSError?
            let status = converter.convert(to: convertedBuffer, error: &error) { _, outStatus in
                outStatus.pointee = .haveData
                return buffer
            }
            guard status != .error, let channelData = convertedBuffer.floatChannelData else { return }

            let count = Int(convertedBuffer.frameLength)
            let ptr = channelData[0]
            let newSamples = Array(UnsafeBufferPointer(start: ptr, count: count))
            DispatchQueue.main.async {
                self.samples.append(contentsOf: newSamples)
                self.duration = Double(self.samples.count) / self.sampleRate
                if self.duration >= self.maxDuration {
                    _ = self.stop()
                }
            }
        }

        try engine.start()
        self.engine = engine
        isRecording = true
    }

    /// Stop recording and return captured samples.
    func stop() -> [Float] {
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        isRecording = false
        timer?.invalidate()
        timer = nil

        // Truncate to maxDuration
        let maxSamples = Int(maxDuration * sampleRate)
        let result = Array(samples.prefix(maxSamples))
        return result
    }
}
