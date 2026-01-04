import AVFoundation
import Combine
import Foundation

/// ViewModel managing real-time speaker diarization state.
/// Uses FluidAudio's SortformerDiarizer with gradientDescent configuration.
@MainActor
final class DiarizerViewModel: ObservableObject {
    
    // MARK: - Published State
    
    /// Whether diarization is currently active
    @Published private(set) var isRecording = false
    
    /// Whether models are loaded and ready
    @Published private(set) var isReady = false
    
    /// Loading status message
    @Published private(set) var statusMessage = "Initializing..."
    
    /// Timeline with diarization history, segments, and predictions
    @Published private(set) var timeline: SortformerTimeline?
    
    /// Speaker cache predictions for visualization
    @Published private(set) var spkcachePreds: [Float]?
    
    /// Trigger for UI updates (incremented when timeline changes)
    @Published private(set) var updateTrigger = 0
    
    /// All recorded audio samples for playback (16kHz mono)
    private(set) var recordedAudio: [Float] = []
    
    // MARK: - Private Properties
    
    private var diarizer: SortformerDiarizer?
    private var audioEngine: AVAudioEngine?
    private var processingTask: Task<Void, Never>?
    private var audioPlayer: AVAudioPlayer?
    private var audioConverter: AudioConverter
    
    private let config = SortformerConfig.default
    private let sampleRate: Double = 16000.0
    
    // Audio buffer for accumulating samples between processing
    private var audioBuffer: [Float] = []
    private let audioBufferLock = NSLock()
    
    // MARK: - Initialization
    
    init() {
        self.audioConverter = AudioConverter()
        Task {
            await loadModels()
        }
    }
    
    // MARK: - Public Methods
    
    /// Start diarization - resets state and begins microphone capture
    func startDiarization() {
        guard isReady, !isRecording else { return }
        
        // Reset state
        audioBuffer = []
        recordedAudio = []
        diarizer?.reset()
        timeline = diarizer?.timeline
        
        // Start audio capture
        do {
            try setupAudioEngine()
            isRecording = true
            statusMessage = "Recording..."
            
            // Start processing loop
            processingTask = Task {
                await processingLoop()
            }
        } catch {
            statusMessage = "Audio error: \(error.localizedDescription)"
        }
    }
    
    /// Stop diarization
    func stopDiarization() {
        guard isRecording else { return }
        
        processingTask?.cancel()
        processingTask = nil
        
        audioEngine?.stop()
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine = nil
        
        // Finalize timeline
        timeline?.finalize()
        updateTrigger += 1
        
        isRecording = false
        statusMessage = "Ready - Click segments to play"
    }
    
    /// Play a segment of recorded audio
    func playSegment(startTime: Float, endTime: Float) {
        guard !isRecording else { return }
        
        let startSample = Int(startTime * Float(sampleRate))
        let endSample = min(Int(endTime * Float(sampleRate)), recordedAudio.count)
        
        guard startSample < endSample && startSample >= 0 else {
            print("Invalid segment range: \(startTime) - \(endTime)")
            return
        }
        
        let segmentSamples = Array(recordedAudio[startSample..<endSample])
        playAudio(samples: segmentSamples)
    }
    
    /// Play audio samples using AVAudioPlayer
    private func playAudio(samples: [Float]) {
        // Convert Float samples to 16-bit PCM WAV data
        guard let wavData = createWAVData(from: samples, sampleRate: Int(sampleRate)) else {
            print("Failed to create WAV data")
            return
        }
        
        do {
            audioPlayer?.stop()
            audioPlayer = try AVAudioPlayer(data: wavData)
            audioPlayer?.play()
            statusMessage = "Playing..."
            
            // Reset status after playback (approximate)
            let duration = Double(samples.count) / sampleRate
            Task {
                try? await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
                await MainActor.run {
                    if !isRecording {
                        statusMessage = "Ready - Click segments to play"
                    }
                }
            }
        } catch {
            print("Failed to play audio: \(error)")
            statusMessage = "Playback error"
        }
    }
    
    /// Create WAV file data from Float samples
    private func createWAVData(from samples: [Float], sampleRate: Int) -> Data? {
        let bytesPerSample = 2  // 16-bit
        let numChannels = 1
        let dataSize = samples.count * bytesPerSample
        let fileSize = 44 + dataSize  // WAV header is 44 bytes
        
        var data = Data()
        
        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(fileSize - 8).littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)
        
        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })  // Chunk size
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })   // PCM format
        data.append(contentsOf: withUnsafeBytes(of: UInt16(numChannels).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        let byteRate = sampleRate * numChannels * bytesPerSample
        data.append(contentsOf: withUnsafeBytes(of: UInt32(byteRate).littleEndian) { Array($0) })
        let blockAlign = numChannels * bytesPerSample
        data.append(contentsOf: withUnsafeBytes(of: UInt16(blockAlign).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(bytesPerSample * 8).littleEndian) { Array($0) })
        
        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(dataSize).littleEndian) { Array($0) })
        
        // Convert float samples to 16-bit PCM
        for sample in samples {
            let clamped = max(-1.0, min(1.0, sample))
            let intSample = Int16(clamped * 32767.0)
            data.append(contentsOf: withUnsafeBytes(of: intSample.littleEndian) { Array($0) })
        }
        
        return data
    }
    
    // MARK: - Private Methods
    
    private func loadModels() async {
        statusMessage = "Loading models from HuggingFace..."
        
        do {
            let newDiarizer = SortformerDiarizer(config: config)
            let models = try await SortformerModels.loadFromHuggingFace(config: config)
            newDiarizer.initialize(models: models)
            
            self.diarizer = newDiarizer
            self.timeline = newDiarizer.timeline
            self.isReady = true
            self.statusMessage = "Ready"
        } catch {
            statusMessage = "Failed to load models: \(error.localizedDescription)"
        }
    }
    
    private func setupAudioEngine() throws {
        let engine = AVAudioEngine()
        let inputNode = engine.inputNode
        
        // Get native format and create converter if needed
        let nativeFormat = inputNode.inputFormat(forBus: 0)
        
        // Install tap at native format
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: nativeFormat) { [weak self] buffer, _ in
            try? self?.handleAudioBuffer(buffer, format: nativeFormat)
        }
        
        try engine.start()
        self.audioEngine = engine
    }
    
    private func handleAudioBuffer(_ buffer: AVAudioPCMBuffer, format: AVAudioFormat) throws {
        guard let channelData = buffer.floatChannelData else { return }
        
        let frameCount = Int(buffer.frameLength)
        let samples: [Float]
        
        // Convert to mono if needed
        if format.channelCount > 1 {
            samples = (0..<frameCount).map { i in
                var sum: Float = 0
                for ch in 0..<Int(format.channelCount) {
                    sum += channelData[ch][i]
                }
                return sum / Float(format.channelCount)
            }
        } else {
            samples = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        }
        
        // Resample if not 16kHz
        let inputSampleRate = format.sampleRate
        let resampledSamples: [Float]
        if abs(inputSampleRate - sampleRate) > 1.0 {
            resampledSamples = try resample(samples, from: inputSampleRate)
        } else {
            resampledSamples = samples
        }
        
        // Add to buffer and store for playback
        audioBufferLock.lock()
        audioBuffer.append(contentsOf: resampledSamples)
        recordedAudio.append(contentsOf: resampledSamples)
        audioBufferLock.unlock()
    }
    
    private func resample(_ samples: [Float], from sourceSampleRate: Double) throws -> [Float] {
        let output = try audioConverter.resample(samples, from: sourceSampleRate)
        return output
    }
    
    private func processingLoop() async {
        while !Task.isCancelled {
            // Get samples from buffer
            audioBufferLock.lock()
            let samples = audioBuffer
            audioBuffer = []
            audioBufferLock.unlock()
            
            if !samples.isEmpty {
                await processAudioChunk(samples)
            }
            
            // Process at ~10Hz
            try? await Task.sleep(nanoseconds: 100_000_000)
        }
    }
    
    private func processAudioChunk(_ samples: [Float]) async {
        guard let diarizer = diarizer else { return }
        
        diarizer.addAudio(samples)
        
        do {
            if let _ = try diarizer.process() {
                // Timeline is updated automatically by diarizer
                // Trigger UI refresh
                updateTrigger += 1
                spkcachePreds = diarizer.state.spkcachePreds
            }
        } catch {
            statusMessage = "Processing error: \(error.localizedDescription)"
            print("Diarizer processing error: \(error)")
        }
    }
}
