import AVFoundation
import Combine
import Foundation

/// ViewModel managing real-time speaker diarization state.
/// Uses FluidAudio's SortformerDiarizer with default configuration.
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
    
    /// FIFO queue predictions for visualization
    @Published private(set) var fifoPreds: [Float]?
    
    /// Trigger for UI updates (incremented when timeline changes)
    @Published private(set) var updateTrigger = 0
    
    /// Embedding graph model for visualization
    let embeddingGraphModel = EmbeddingGraphModel()
    
    /// Right context frames for FIFO alignment
    var chunkRightContext: Int {
        globalConfig.chunkRightContext
    }
    
    /// Left context frames for FIFO alignment  
    var chunkLeftContext: Int {
        globalConfig.chunkLeftContext
    }
    
    /// Segment annotations - maps "startFrame-endFrame-speaker" to custom label
    @Published var segmentAnnotations: [String: String] = [:]
    
    /// All recorded audio samples for playback (16kHz mono)
    private(set) var recordedAudio: [Float] = []
    
    // MARK: - Private Properties
    
    private var diarizer: SortformerDiarizer?
    private var embeddingManager: EmbeddingManager?
    private var audioEngine: AVAudioEngine?
    private var processingTask: Task<Void, Never>?
    private var audioPlayer: AVAudioPlayer?
    private var audioConverter: AudioConverter
    
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
        embeddingManager?.resetAudioBuffer()
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
        
        // Process any remaining audio in buffer
        audioBufferLock.lock()
        let remainingSamples = audioBuffer
        audioBuffer = []
        audioBufferLock.unlock()
        
        if !remainingSamples.isEmpty {
            diarizer?.addAudio(remainingSamples)
            embeddingManager?.appendAudio(remainingSamples)
            // Process until no more chunks
            while let _ = try? diarizer?.process() {
                // Keep processing
            }
        }
        
        // Finalize timeline
        timeline?.finalize()
        updateTrigger += 1
        
        // Extract any remaining embeddings
        if let tl = timeline, let embMgr = embeddingManager {
            embMgr.processRequests(from: tl) { [weak self] count in
                guard let self = self else { return }
                print("[Embeddings] Final extraction: \(count) embeddings")
                self.updateGraph()
            }
        }
        
        updateGraph()
        
        isRecording = false
        statusMessage = "Ready - Click segments to play"
    }
    
    /// Save recorded audio to WAV file
    func saveRecording(to url: URL) {
        guard !recordedAudio.isEmpty else {
            statusMessage = "No recording to save"
            return
        }
        
        guard let wavData = createWAVData(from: recordedAudio, sampleRate: Int(sampleRate)) else {
            statusMessage = "Failed to create WAV data"
            return
        }
        
        do {
            try wavData.write(to: url)
            statusMessage = "Saved to \(url.lastPathComponent)"
        } catch {
            statusMessage = "Save failed: \(error.localizedDescription)"
        }
    }
    
    /// Load and process an audio file
    func loadAudioFile(from url: URL) async {
        guard isReady, !isRecording else { return }
        guard let diarizer = diarizer else { return }
        
        statusMessage = "Loading audio file..."
        
        // Start security-scoped access for sandboxed apps
        let didStartAccessing = url.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                url.stopAccessingSecurityScopedResource()
            }
        }
        
        do {
            // Load audio file
            let audioFile = try AVAudioFile(forReading: url)
            let format = audioFile.processingFormat
            let frameCount = AVAudioFrameCount(audioFile.length)
            
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                statusMessage = "Failed to create audio buffer"
                return
            }
            
            try audioFile.read(into: buffer)
            
            guard let channelData = buffer.floatChannelData else {
                statusMessage = "Failed to read audio data"
                return
            }
            
            // Convert to mono
            let samples: [Float]
            if format.channelCount > 1 {
                samples = (0..<Int(frameCount)).map { i in
                    var sum: Float = 0
                    for ch in 0..<Int(format.channelCount) {
                        sum += channelData[ch][i]
                    }
                    return sum / Float(format.channelCount)
                }
            } else {
                samples = Array(UnsafeBufferPointer(start: channelData[0], count: Int(frameCount)))
            }
            
            // Resample to 16kHz if needed
            let resampledSamples: [Float]
            if abs(format.sampleRate - sampleRate) > 1.0 {
                resampledSamples = try audioConverter.resample(samples, from: format.sampleRate)
            } else {
                resampledSamples = samples
            }
            
            // Store for playback
            recordedAudio = resampledSamples
            
            // Reset and feed audio to embedding manager
            embeddingManager?.resetAudioBuffer()
            embeddingManager?.appendAudio(resampledSamples)
            
            statusMessage = "Processing \(url.lastPathComponent)..."
            
            // Process complete audio
            timeline = try diarizer.processComplete(resampledSamples)
            spkcachePreds = diarizer.state.spkcachePreds  // Update speaker cache display
            fifoPreds = diarizer.state.fifoPreds  // Update FIFO queue display
            updateTrigger += 1
            
            // Extract embeddings for all segments
            if let tl = timeline, let embMgr = embeddingManager {
                statusMessage = "Extracting embeddings..."
                embMgr.processRequests(from: tl) { [weak self] count in
                    guard let self = self else { return }
                    print("[Embeddings] Extracted \(count) embeddings")
                    self.updateGraph()
                }
            }
            
            updateGraph()
            
            let duration = Float(resampledSamples.count) / Float(sampleRate)
            statusMessage = String(format: "Processed %.1fs - Click segments to play", duration)
            
            // Print segments to console
            printSegments()
            
        } catch {
            statusMessage = "Load failed: \(error.localizedDescription)"
            print("Audio load error: \(error)")
        }
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
            let newDiarizer = SortformerDiarizer(config: globalConfig)
            let models = try await SortformerModels.loadFromHuggingFace(config: globalConfig)
            newDiarizer.initialize(models: models)
            
            // Initialize embedding manager
            let newEmbeddingManager = EmbeddingManager(
                frameDurationSeconds: globalConfig.frameDurationSeconds
            )
            try newEmbeddingManager.initialize()
            
            self.diarizer = newDiarizer
            self.embeddingManager = newEmbeddingManager
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
            resampledSamples = try audioConverter.resample(samples, from: inputSampleRate)
        } else {
            resampledSamples = samples
        }
        
        // Add to buffer and store for playback
        audioBufferLock.lock()
        audioBuffer.append(contentsOf: resampledSamples)
        recordedAudio.append(contentsOf: resampledSamples)
        audioBufferLock.unlock()
        
        // Also feed to embedding manager for extraction
        embeddingManager?.appendAudio(resampledSamples)
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
                fifoPreds = diarizer.state.fifoPreds
                
                // Process embedding extraction requests
                if let tl = timeline, let embMgr = embeddingManager {
                    embMgr.processRequests(from: tl) { [weak self] count in
                        guard let self = self, count > 0 else { return }
                        // Update graph when new embeddings are extracted
                        self.updateGraph()
                    }
                }
                
                // Update graph occasionally (every 5 updates or if small) to save perf
                if updateTrigger % 5 == 0 || (timeline?.embeddingSegments.count ?? 0) < 50 {
                   updateGraph()
                }
            }
        } catch {
            statusMessage = "Processing error: \(error.localizedDescription)"
            print("Diarizer processing error: \(error)")
        }
    }
    
    /// Update the embedding graph model from the current timeline
    private func updateGraph() {
        guard let tl = timeline else { return }
        
        #if DEBUG
        let segCount = tl.embeddingSegments.count
        let embCount = tl.embeddingSegments.reduce(0) { $0 + $1.embeddings.count }
        if segCount > 0 || embCount > 0 {
            print("[Graph] Updating with \(segCount) segments, \(embCount) total embeddings")
        }
        #endif
        
        // Run on main actor (already on main actor)
        embeddingGraphModel.update(from: tl.embeddingSegments)
    }
    
    /// Print all segments in CSV format: speaker_id,start_time,end_time
    private func printSegments() {
        guard let tl = timeline else { return }
        
        print("\n--- Diarization Segments ---")
        print("speaker_id,start_time,end_time")
        
        // Collect all segments and sort by start time
        let allSegments = tl.segments.flatMap { $0 }.sorted { $0.startTime < $1.startTime }
        
        for segment in allSegments {
            let startStr = String(format: "%.2f", segment.startTime)
            let endStr = String(format: "%.2f", segment.endTime)
            print("\(segment.speakerIndex),\(startStr),\(endStr)")
        }
        
        print("--- End Segments (\(allSegments.count) total) ---\n")
    }
    
    // MARK: - Segment Annotation
    
    /// Generate a unique key for a segment
    static func segmentKey(_ segment: SortformerSegment) -> String {
        return "\(segment.startFrame)-\(segment.endFrame)-\(segment.speakerIndex)"
    }
    
    /// Set annotation for a segment
    func setAnnotation(for segment: SortformerSegment, label: String) {
        let key = Self.segmentKey(segment)
        if label.isEmpty {
            segmentAnnotations.removeValue(forKey: key)
        } else {
            segmentAnnotations[key] = label
        }
    }
    
    /// Get annotation for a segment (nil if not annotated)
    func getAnnotation(for segment: SortformerSegment) -> String? {
        let key = Self.segmentKey(segment)
        return segmentAnnotations[key]
    }
    
    /// Export segments to a text file
    func exportSegments(to url: URL) {
        guard let tl = timeline else { return }
        
        var output = "speaker_id,start_time,end_time\n"
        
        // Collect all segments and sort by start time
        let allSegments = tl.segments.flatMap { $0 }.sorted { $0.startTime < $1.startTime }
        
        for segment in allSegments {
            // Use annotation if available, otherwise use original speaker index
            let label = getAnnotation(for: segment) ?? String(segment.speakerIndex)
            let startStr = String(format: "%.2f", segment.startTime)
            let endStr = String(format: "%.2f", segment.endTime)
            output += "\(label),\(startStr),\(endStr)\n"
        }
        
        do {
            try output.write(to: url, atomically: true, encoding: .utf8)
            statusMessage = "Exported \(allSegments.count) segments to \(url.lastPathComponent)"
        } catch {
            statusMessage = "Export failed: \(error.localizedDescription)"
        }
    }
}
