import AVFoundation
import Combine
import Foundation
import AHCClustering

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

    /// Latest live dendrogram model built from SpeakerProfile clustering
    @Published private(set) var dendrogramModel: AHCDendrogramModel = .empty
    
    /// All recorded audio samples for playback (16kHz mono)
    private(set) var recordedAudio: [Float] = []
    
    // MARK: - Private Properties
    
    private var diarizer: SortformerDiarizer?
    private var audioEngine: AVAudioEngine?
    private var processingTask: Task<Void, Never>?
    private var audioPlayer: AVAudioPlayer?
    private var audioConverter: AudioConverter
    private let dendrogramClusteringConfig = ClusteringConfig(
        linkagePolicy: dendrogramLinkagePolicy,
        minSeparation: 0.3,
        clusteringThreshold: 0.3,
        maxEmbeddings: 20,
        minEmbeddings: 10,
        maxRepresentatives: 1
    )
    private var speakerProfile: SpeakerProfile
    private var streamedFinalizedEmbeddingSegmentCount = 0
    // Keep replay traffic bounded so replay/incremental catch-up cannot explode memory.
    private let dendrogramMaxFinalizedReplaySegments = 3000
    private let dendrogramMaxTentativeReplaySegments = 1500
    private let wardLogFloor: Float = 1e-12
    
    private let sampleRate: Double = 16000.0
    
    // Audio buffer for accumulating samples between processing
    private var audioBuffer: [Float] = []
    private let audioBufferLock = NSLock()
    
    // MARK: - Initialization
    
    init() {
        self.audioConverter = AudioConverter()
        self.speakerProfile = SpeakerProfile(config: dendrogramClusteringConfig)
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
            fileProcessingProgress = 0.02
            
            // Allow UI to update
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            
            let audioFile = try AVAudioFile(forReading: url)
            let format = audioFile.processingFormat
            let frameCount = AVAudioFrameCount(audioFile.length)
            
            guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
                statusMessage = "Failed to create audio buffer"
                return
            }
            
            fileProcessingProgress = 0.04
            statusMessage = "Reading audio data..."
            
            // Allow UI to update
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            
            try audioFile.read(into: buffer)
            
            guard let channelData = buffer.floatChannelData else {
                statusMessage = "Failed to read audio data"
                return
            }
            
            // Convert to mono
            fileProcessingProgress = 0.06
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
            fileProcessingProgress = 0.08
            statusMessage = "Resampling audio..."
            
            // Allow UI to update
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            
            let resampledSamples: [Float]
            if Swift.abs(format.sampleRate - sampleRate) > 1.0 {
                resampledSamples = try audioConverter.resample(samples, from: format.sampleRate)
            } else {
                resampledSamples = samples
            }
            
            // Store for playback
            recordedAudio = resampledSamples
            fileProcessingProgress = 0.1
            statusMessage = "Processing \(url.lastPathComponent)..."
            
            // Allow UI to update before heavy processing
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            
            // Process complete audio
            timeline = try await diarizer.processComplete(resampledSamples) { processed, total in
                self.fileProcessingProgress = 0.1 + Double(processed) / Double(total) * 0.89
                print("Processed \(processed) out of \(total)")
                try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            }
            fileProcessingProgress = 0.99
            statusMessage = "Loading results..."
            
            // Allow UI to update
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
            
//            try? timeline?.finalize()  // Finalize all tentative predictions and segments
            spkcachePreds = diarizer.state.spkcachePreds  // Update speaker cache display
            fifoPreds = diarizer.state.fifoPreds  // Update FIFO queue display
            updateTrigger += 1

            updateDendrogram()
            
            fileProcessingProgress = 1.0
            
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
        if Swift.abs(inputSampleRate - sampleRate) > 1.0 {
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
        speakerProfile = SpeakerProfile(config: dendrogramClusteringConfig)
        streamedFinalizedEmbeddingSegmentCount = 0
        dendrogramModel = .empty
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

        let pendingFinalizedCount = tl.embeddingSegments.count - streamedFinalizedEmbeddingSegmentCount
        if pendingFinalizedCount > dendrogramMaxFinalizedReplaySegments {
            print("[Dendrogram] Backlog too large (\(pendingFinalizedCount)); replaying capped window")
            _ = replayFullDendrogramState(from: tl)
        }

        let newFinalized = Array(tl.embeddingSegments.dropFirst(streamedFinalizedEmbeddingSegmentCount))
        let tentativeWindow = Array(tl.tentativeEmbeddingSegments.suffix(dendrogramMaxTentativeReplaySegments))
        speakerProfile.stream(newFinalized: newFinalized, newTentative: tentativeWindow)
        streamedFinalizedEmbeddingSegmentCount = tl.embeddingSegments.count

        let speakerIndexBySegmentId = makeSpeakerIndexBySegmentId(from: tl)
        var nextModel = makeDendrogramModel(
            from: speakerProfile,
            speakerIndexBySegmentId: speakerIndexBySegmentId
        )
        if nextModel.nodes.contains(where: { !$0.mergeDistance.isFinite }) {
            print("[Dendrogram] Invalid merge distance detected; replaying full clustering state")
            if replayFullDendrogramState(from: tl) {
                nextModel = makeDendrogramModel(
                    from: speakerProfile,
                    speakerIndexBySegmentId: speakerIndexBySegmentId
                )
            }
        }
        dendrogramModel = nextModel
    }

    @discardableResult
    private func replayFullDendrogramState(from timeline: SortformerTimeline) -> Bool {
        speakerProfile = SpeakerProfile(config: dendrogramClusteringConfig)
        let finalizedWindow = Array(timeline.embeddingSegments.suffix(dendrogramMaxFinalizedReplaySegments))
        let tentativeWindow = Array(timeline.tentativeEmbeddingSegments.suffix(dendrogramMaxTentativeReplaySegments))
        speakerProfile.stream(newFinalized: finalizedWindow, newTentative: tentativeWindow)
        print("[Dendrogram] Full replay succeeded (\(timeline.embeddingSegments.count) finalized segments)")
        streamedFinalizedEmbeddingSegmentCount = timeline.embeddingSegments.count
        return true
    }

    private func makeSpeakerIndexBySegmentId(from timeline: SortformerTimeline) -> [UUID: Int] {
        var lookup: [UUID: Int] = [:]
        lookup.reserveCapacity(timeline.embeddingSegments.count + timeline.tentativeEmbeddingSegments.count)

        for segment in timeline.embeddingSegments {
            lookup[segment.id] = segment.speakerIndex
        }
        for segment in timeline.tentativeEmbeddingSegments {
            lookup[segment.id] = segment.speakerIndex
        }

        return lookup
    }

    private func makeDendrogramModel(
        from profile: SpeakerProfile,
        speakerIndexBySegmentId: [UUID: Int]
    ) -> AHCDendrogramModel {
        let matrix = profile.matrix
        let dendrogram = profile.dendrogram
        let rootIndex = Int(dendrogram.rootId())
        let nodeCount = Int(dendrogram.nodeCount())
        let activeLeafCount = max(0, Int(matrix.embeddingCount()))

        guard rootIndex >= 0, rootIndex < nodeCount else {
            return AHCDendrogramModel(rootIndex: -1, activeLeafCount: activeLeafCount, nodes: [])
        }

        let nodeIds = reachableNodeIds(in: dendrogram, rootIndex: rootIndex, nodeCount: nodeCount)
        guard !nodeIds.isEmpty else {
            return AHCDendrogramModel(rootIndex: -1, activeLeafCount: activeLeafCount, nodes: [])
        }

        let linkagePolicy = profile.config.linkagePolicy
        let useDynamicWardThreshold = linkagePolicy == .wardLinkage
        var hasLinkageCutoff = false
        var linkageCutoff: Float = 0

        if useDynamicWardThreshold {
            var internalMergeDistances: [Float] = []
            internalMergeDistances.reserveCapacity(nodeIds.count)

            for nodeId in nodeIds {
                let node = dendrogram.node(nodeId)
                let leftChild = Int(node.leftChild)
                let rightChild = Int(node.rightChild)
                let isInternalNode = leftChild >= 0 && rightChild >= 0 &&
                    leftChild < nodeCount && rightChild < nodeCount
                guard isInternalNode else { continue }
                internalMergeDistances.append(
                    transformedLinkageDistance(node.mergeDistance, linkagePolicy: linkagePolicy)
                )
            }

            if !internalMergeDistances.isEmpty {
                let sum = internalMergeDistances.reduce(0, +)
                let mean = sum / Float(internalMergeDistances.count)
                let variance = internalMergeDistances.reduce(0) { partial, distance in
                    let delta = distance - mean
                    return partial + delta * delta
                } / Float(internalMergeDistances.count)
                let stddev = sqrtf(max(variance, 0))
                linkageCutoff = mean + 2 * stddev
                hasLinkageCutoff = linkageCutoff.isFinite
            }
        } else {
            linkageCutoff = profile.config.mergeThreshold
            hasLinkageCutoff = linkageCutoff.isFinite
        }

        var nodes: [AHCDendrogramNodeModel] = []
        nodes.reserveCapacity(nodeIds.count)
        let matrixSize = Int(matrix.size())

        for nodeId in nodeIds {
            let node = dendrogram.node(nodeId)
            let matrixIndex = Int(node.matrixIndex)
            let leftChild = Int(node.leftChild)
            let rightChild = Int(node.rightChild)
            let isLeaf = leftChild < 0 || rightChild < 0
            var speakerIndex = -1

            if isLeaf && matrixIndex >= 0 && matrixIndex < matrixSize {
                let embedding = matrix.embedding(matrixIndex)
                if embedding.hasVector() {
                    speakerIndex = speakerIndexBySegmentId[uuid(from: embedding.id())] ?? -1
                }
            }

            let mergeDistance = transformedLinkageDistance(node.mergeDistance, linkagePolicy: linkagePolicy)
            let isInternalNode = leftChild >= 0 && rightChild >= 0 &&
                leftChild < nodeCount && rightChild < nodeCount
            let exceedsLinkageThreshold = isInternalNode && hasLinkageCutoff &&
                mergeDistance.isFinite && mergeDistance > linkageCutoff

            nodes.append(
                AHCDendrogramNodeModel(
                    id: nodeId,
                    matrixIndex: matrixIndex,
                    leftChild: leftChild,
                    rightChild: rightChild,
                    speakerIndex: speakerIndex,
                    count: Int(node.count),
                    weight: node.weight,
                    mergeDistance: mergeDistance,
                    mustLink: false,
                    exceedsLinkageThreshold: exceedsLinkageThreshold
                )
            )
        }

        return AHCDendrogramModel(
            rootIndex: rootIndex,
            activeLeafCount: activeLeafCount,
            nodes: nodes
        )
    }

    private func reachableNodeIds(in dendrogram: Dendrogram, rootIndex: Int, nodeCount: Int) -> [Int] {
        guard rootIndex >= 0, rootIndex < nodeCount else {
            return []
        }

        var visited: Set<Int> = []
        visited.reserveCapacity(nodeCount)
        var stack: [Int] = [rootIndex]
        stack.reserveCapacity(nodeCount)

        while let nodeId = stack.popLast() {
            guard nodeId >= 0, nodeId < nodeCount, visited.insert(nodeId).inserted else {
                continue
            }

            let node = dendrogram.node(nodeId)
            let leftChild = Int(node.leftChild)
            let rightChild = Int(node.rightChild)

            if leftChild >= 0 {
                stack.append(leftChild)
            }
            if rightChild >= 0 {
                stack.append(rightChild)
            }
        }

        return visited.sorted()
    }

    private func transformedLinkageDistance(_ mergeDistance: Float, linkagePolicy: LinkagePolicyType) -> Float {
        guard mergeDistance.isFinite else {
            return 0
        }

        if linkagePolicy == .wardLinkage {
            let flooredDistance = max(mergeDistance, wardLogFloor)
            let loggedDistance = logf(flooredDistance)
            return loggedDistance.isFinite ? loggedDistance : 0
        }

        return mergeDistance < 0 ? 0 : mergeDistance
    }

    private func uuid(from wrapper: UUIDWrapper) -> UUID {
        var uuidTuple: uuid_t = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        var words = (wrapper.data.0, wrapper.data.1)
        withUnsafeMutableBytes(of: &uuidTuple) { destination in
            withUnsafeBytes(of: &words) { source in
                destination.copyBytes(from: source)
            }
        }
        return UUID(uuid: uuidTuple)
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
    
    // MARK: - Speaker Purge

    /// Remove a speaker from the streaming state and refresh the UI.
    ///
    /// This scrubs the speaker's activity from the FIFO and speaker cache so
    /// future model passes no longer see that speaker.
    func purgeSpeaker(at speakerIndex: Int) {
        guard let diarizer = diarizer else { return }

        diarizer.removeSpeaker(at: speakerIndex)

        // Refresh cached predictions for UI
        spkcachePreds = diarizer.state.spkcachePreds
        fifoPreds = diarizer.state.fifoPreds
        updateTrigger += 1

        statusMessage = "Purged speaker \(speakerIndex) from streaming state"
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
