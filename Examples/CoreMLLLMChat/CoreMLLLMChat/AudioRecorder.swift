import AVFoundation
import Foundation

/// Simple audio recorder that captures mono 16kHz PCM for the Gemma 4 audio encoder.
@Observable
final class AudioRecorder {
    var isRecording = false
    var duration: TimeInterval = 0

    /// Recorded samples, set when recording finishes (auto-stop or manual stop).
    var recordedSamples: [Float]?

    private var engine: AVAudioEngine?
    private var samples: [Float] = []
    private let sampleRate: Double = 16000

    /// Maximum recording duration in seconds.
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
        recordedSamples = nil

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

        guard let converter = AVAudioConverter(from: inputFormat, to: targetFormat) else { return }

        inputNode.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) {
            [weak self] buffer, _ in
            guard let self, self.isRecording else { return }

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
            let newSamples = Array(UnsafeBufferPointer(start: channelData[0], count: count))
            DispatchQueue.main.async {
                self.samples.append(contentsOf: newSamples)
                self.duration = Double(self.samples.count) / self.sampleRate
                if self.duration >= self.maxDuration {
                    self.stop()
                }
            }
        }

        try engine.start()
        self.engine = engine
        isRecording = true
    }

    /// Stop recording. Samples are stored in `recordedSamples`.
    func stop() {
        guard isRecording else { return }
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        isRecording = false

        let maxSamples = Int(maxDuration * sampleRate)
        recordedSamples = Array(samples.prefix(maxSamples))
    }

    /// Clear recorded audio.
    func clear() {
        recordedSamples = nil
        samples = []
        duration = 0
    }
}
