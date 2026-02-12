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
    
    /// Progress for file processing (0.0 to 1.0), nil when not processing
    @Published private(set) var fileProcessingProgress: Double? = nil
    
    /// Progress for clustering (0.0 to 1.0), nil when not clustering
    @Published private(set) var clusteringProgress: Double? = nil
    
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

    /// Latest live AHC dendrogram snapshot from C++ clustering
    @Published private(set) var dendrogramModel: AHCDendrogramModel = .empty
    
    /// All recorded audio samples for playback (16kHz mono)
    private(set) var recordedAudio: [Float] = []
    
    // MARK: - Private Properties
    
    private var diarizer: SortformerDiarizer?
    private var audioEngine: AVAudioEngine?
    private var processingTask: Task<Void, Never>?
    private var audioPlayer: AVAudioPlayer?
    private var audioConverter: AudioConverter
    private let ahcBridge = AHCSpeakerForestBridge(
        numRepresentatives: 24,
        minEmbeddings: 600,
        maxEmbeddings: 1500
    )
    private var streamedFinalizedEmbeddingSegmentCount = 0
    private let clusteringEmbeddingDimensions = 192
    
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
        resetDendrogramState()
        
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
            // Process until no more chunks
            while let _ = try? diarizer?.process() {
                // Keep processing
            }
        }
        
        // Finalize timeline
        try? timeline?.finalize()
        updateTrigger += 1

        updateDendrogram()
        
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

        resetDendrogramState()
        
        fileProcessingProgress = 0.0
        statusMessage = "Loading audio file..."
        
        // Allow UI to update
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        
        // Start security-scoped access for sandboxed apps
        let didStartAccessing = url.startAccessingSecurityScopedResource()
        defer {
            if didStartAccessing {
                url.stopAccessingSecurityScopedResource()
            }
            fileProcessingProgress = nil
        }
        
        do {
            // Load audio file
            fileProcessingProgress = 0.1
            
            // Allow UI to update
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            
            let audioFile = try AVAudioFile(forReading: url)
            let format = audioFile.processingFormat
            let frameCount = AVAudioFrameCount(audioFile.length)
            
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                statusMessage = "Failed to create audio buffer"
                return
            }
            
            fileProcessingProgress = 0.2
            statusMessage = "Reading audio data..."
            
            // Allow UI to update
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            
            try audioFile.read(into: buffer)
            
            guard let channelData = buffer.floatChannelData else {
                statusMessage = "Failed to read audio data"
                return
            }
            
            // Convert to mono
            fileProcessingProgress = 0.3
            statusMessage = "Converting to mono..."
            
            // Allow UI to update
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            
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
            fileProcessingProgress = 0.4
            statusMessage = "Resampling audio..."
            
            // Allow UI to update
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            
            let resampledSamples: [Float]
            if abs(format.sampleRate - sampleRate) > 1.0 {
                resampledSamples = try audioConverter.resample(samples, from: format.sampleRate)
            } else {
                resampledSamples = samples
            }
            
            // Store for playback
            recordedAudio = resampledSamples
            fileProcessingProgress = 0.5
            statusMessage = "Processing \(url.lastPathComponent)..."
            
            // Allow UI to update before heavy processing
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            
            // Process complete audio
            timeline = try diarizer.processComplete(resampledSamples)
            fileProcessingProgress = 0.9
            statusMessage = "Finalizing..."
            
            // Allow UI to update
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            
            try? timeline?.finalize()  // Finalize all tentative predictions and segments
            spkcachePreds = diarizer.state.spkcachePreds  // Update speaker cache display
            fifoPreds = diarizer.state.fifoPreds  // Update FIFO queue display
            updateTrigger += 1

            updateDendrogram()
            
            fileProcessingProgress = 1.0
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
            self.timeline = newDiarizer.timeline
            resetDendrogramState()
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
            guard let _ = try diarizer.process() else {
                return
            }
            
            // Timeline is updated automatically by diarizer
            // Trigger UI refresh
            updateTrigger += 1
            spkcachePreds = diarizer.state.spkcachePreds
            fifoPreds = diarizer.state.fifoPreds
            updateDendrogram()
            
            // Don't update graph during streaming - it causes lag and isn't visible anyway
            // Graph will be updated when clustering is triggered
        } catch {
            statusMessage = "Processing error: \(error.localizedDescription)"
            print("Diarizer processing error: \(error)")
        }
    }

    private func resetDendrogramState() {
        ahcBridge.reset()
        streamedFinalizedEmbeddingSegmentCount = 0
        dendrogramModel = .empty
    }

    private func makeBridgeSegments(from segments: [EmbeddingSegment]) -> [AHCEmbeddingSegmentInput] {
        var converted: [AHCEmbeddingSegmentInput] = []
        converted.reserveCapacity(segments.count)

        for segment in segments {
            var embeddings: [AHCEmbeddingSample] = []
            embeddings.reserveCapacity(segment.embeddings.count)

            for embedding in segment.embeddings where embedding.embedding.count == clusteringEmbeddingDimensions {
                let vectorData = embedding.embedding.withUnsafeBufferPointer { ptr -> Data in
                    guard let base = ptr.baseAddress else { return Data() }
                    return Data(bytes: base, count: ptr.count * MemoryLayout<Float>.size)
                }
                guard !vectorData.isEmpty else { continue }

                embeddings.append(
                    AHCEmbeddingSample(
                        identifier: embedding.id,
                        speakerIndex: segment.speakerIndex,
                        weight: Float(embedding.length),
                        vectorData: vectorData
                    )
                )
            }

            guard !embeddings.isEmpty else { continue }
            converted.append(
                AHCEmbeddingSegmentInput(
                    speakerIndex: segment.speakerIndex,
                    embeddings: embeddings
                )
            )
        }

        return converted
    }

    private func updateDendrogram() {
        guard let tl = timeline else {
            dendrogramModel = .empty
            streamedFinalizedEmbeddingSegmentCount = 0
            return
        }

        if tl.embeddingSegments.count < streamedFinalizedEmbeddingSegmentCount {
            _ = replayFullDendrogramState(from: tl)
        }

        let newFinalized = Array(tl.embeddingSegments.dropFirst(streamedFinalizedEmbeddingSegmentCount))

        let finalizedSegments = makeBridgeSegments(from: newFinalized)
        let tentativeSegments = makeBridgeSegments(from: tl.tentativeEmbeddingSegments)
        let streamedIncremental = ahcBridge.stream(
            withFinalizedSegments: finalizedSegments,
            tentativeSegments: tentativeSegments
        )
        if streamedIncremental {
            streamedFinalizedEmbeddingSegmentCount = tl.embeddingSegments.count
        } else {
            print("[Dendrogram] Incremental stream failed; replaying full clustering state")
            _ = replayFullDendrogramState(from: tl)
        }

        var snapshot = ahcBridge.dendrogramSnapshot()
        if snapshotHasOutOfRangeMergeDistance(snapshot) {
            print("[Dendrogram] Out-of-range merge distance detected; replaying full clustering state")
            _ = replayFullDendrogramState(from: tl)
            snapshot = ahcBridge.dendrogramSnapshot()
        }
        dendrogramModel = AHCDendrogramModel(snapshot: snapshot)
    }

    @discardableResult
    private func replayFullDendrogramState(from timeline: SortformerTimeline) -> Bool {
        ahcBridge.reset()
        let finalizedSegments = makeBridgeSegments(from: timeline.embeddingSegments)
        let tentativeSegments = makeBridgeSegments(from: timeline.tentativeEmbeddingSegments)
        let success = ahcBridge.stream(
            withFinalizedSegments: finalizedSegments,
            tentativeSegments: tentativeSegments
        )
        if success {
            print("[Dendrogram] Full replay succeeded (\(timeline.embeddingSegments.count) finalized segments)")
        } else {
            print("[Dendrogram] Full replay failed")
        }
        streamedFinalizedEmbeddingSegmentCount = success ? timeline.embeddingSegments.count : 0
        return success
    }

    private func snapshotHasOutOfRangeMergeDistance(_ snapshot: AHCDendrogramSnapshot) -> Bool {
        let epsilon: Float = 1e-3
        return snapshot.nodes.contains { node in
            node.mergeDistance.isFinite && (node.mergeDistance < -epsilon || node.mergeDistance > 2 + epsilon)
        }
    }
    
    /// Update the embedding graph model from the current timeline
    private func updateGraph() {
        guard let tl = timeline else { return }
        
        // Combine finalized and tentative embedding segments
        let allEmbeddingSegments = tl.embeddingSegments + tl.tentativeEmbeddingSegments
        
        #if DEBUG
        let segCount = allEmbeddingSegments.count
        let embCount = allEmbeddingSegments.reduce(0) { $0 + $1.embeddings.count }
        print("[Graph] Updating with \(segCount) segments (\(tl.embeddingSegments.count) finalized, \(tl.tentativeEmbeddingSegments.count) tentative), \(embCount) total embeddings")
        #endif
        
        // Run on main actor (already on main actor)
        embeddingGraphModel.update(from: allEmbeddingSegments)
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
    
    // MARK: - Spectral Clustering
    
    /// Perform clustering on the current embeddings and re-annotate segments
    /// - Parameters:
    ///   - method: Clustering method to use
    ///   - numClusters: Number of clusters (nil = auto-detect using eigengap heuristic)
    /// - Returns: True if clustering was successful
    @discardableResult
    func performClustering(method: ClusteringMethod = .constrainedAHC, numClusters: Int? = nil) async -> Bool {
        guard let tl = timeline else {
            statusMessage = "No timeline data for clustering"
            return false
        }
        
        clusteringProgress = 0.0
        statusMessage = "Building affinity matrix..."
        
        // Allow UI to update
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        
        // First update the graph model with latest embeddings
        clusteringProgress = 0.2
        statusMessage = "Updating embeddings..."
        
        // Allow UI to update
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        
        updateGraph()
        
        clusteringProgress = 0.4
        statusMessage = "Computing \(method.rawValue) clustering..."
        
        // Allow UI to update before heavy computation
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        
        // Run clustering (both this and graphModel are @MainActor, uses Task.yield internally)
        let success = await embeddingGraphModel.performClustering(method: method, numClusters: numClusters)
        
        clusteringProgress = 0.8
        statusMessage = "Annotating segments..."
        
        // Allow UI to update
        try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        
        if success {
            let clusterCount = Set(embeddingGraphModel.nodes.compactMap { $0.clusterLabel }).count
            
            // Re-annotate segments based on cluster labels
            annotateSegmentsFromClusters(timeline: tl)
            
            clusteringProgress = 1.0
            statusMessage = "Clustering complete - \(clusterCount) clusters found"
        } else {
            statusMessage = "Clustering failed - not enough embeddings"
        }
        
        clusteringProgress = nil
        return success
    }
    
    /// Annotate segments based on cluster assignments from the embedding graph
    /// Uses voting: each embedding votes for its cluster, segment gets majority label
    private func annotateSegmentsFromClusters(timeline tl: SortformerTimeline) {
        // Build a map from embedding ID to cluster label
        var embeddingToCluster: [UUID: Int] = [:]
        for node in embeddingGraphModel.nodes {
            if let clusterLabel = node.clusterLabel {
                embeddingToCluster[node.embeddingId] = clusterLabel
            }
        }
        
        guard !embeddingToCluster.isEmpty else { return }
        
        // Clear existing annotations (only cluster-generated ones)
        // We'll use "Cluster X" format to distinguish from user-provided annotations
        let clusterPrefix = "Cluster "
        for key in segmentAnnotations.keys {
            if let value = segmentAnnotations[key], value.hasPrefix(clusterPrefix) {
                segmentAnnotations.removeValue(forKey: key)
            }
        }
        
        // For each segment, find matching embeddings and vote on cluster
        let allSegments = tl.segments.flatMap { $0 }
        
        for segment in allSegments {
            // Find embeddings that overlap with this segment
            var clusterVotes: [Int: Int] = [:]
            
            for node in embeddingGraphModel.nodes {
                // Check if this embedding overlaps with the segment
                let nodeStart = node.startFrame
                let nodeEnd = node.endFrame
                
                // Check overlap: segments overlap if one starts before other ends
                if nodeStart < segment.endFrame && nodeEnd > segment.startFrame {
                    // Same speaker?
                    if node.speakerIndex == segment.speakerIndex {
                        if let cluster = node.clusterLabel {
                            clusterVotes[cluster, default: 0] += 1
                        }
                    }
                }
            }
            
            // Assign the majority cluster label
            if let (bestCluster, _) = clusterVotes.max(by: { $0.value < $1.value }) {
                let key = Self.segmentKey(segment)
                // Only set if no user annotation exists
                if segmentAnnotations[key] == nil || segmentAnnotations[key]?.hasPrefix(clusterPrefix) == true {
                    segmentAnnotations[key] = "\(clusterPrefix)\(bestCluster)"
                }
            }
        }
    }
}
