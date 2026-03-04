import AVFoundation
import Combine
import Foundation
import simd

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

    /// Distance from each timeline segment centroid to its nearest live cluster centroid.
    @Published private(set) var segmentCentroidDistances: [UInt64: Float] = [:]

    /// Distance from each embedding segment centroid to its nearest live cluster centroid.
    @Published private(set) var embeddingSegmentCentroidDistances: [UUID: Float] = [:]
    
    /// 3D PCA visualization built directly from timeline embeddings/clusters.
    @Published private(set) var kmeansPCAPlotModel: KMeansPCAPlotModel = .empty
    
    /// All recorded audio samples for playback (16kHz mono)
    private(set) var recordedAudio: [Float] = []
    
    // MARK: - Private Properties
    
    private var diarizer: SortformerDiarizer?
    private var audioEngine: AVAudioEngine?
    private var processingTask: Task<Void, Never>?
    private var audioPlayer: AVAudioPlayer?
    private var audioConverter: AudioConverter
    // Keep live updates responsive while recording.
    private let streamingPCARefreshInterval: TimeInterval = 0.2
    private let streamingPCAPointLimit = 900
    private let finalizedPCAPointLimit = 2600
    private var lastStreamingPCARefreshTime: TimeInterval = 0
    private let pcaBuildQueue = DispatchQueue(label: "SortformerTest.PCA.Build", qos: .utility)
    private var pendingPCARawPoints: [PCAPointRaw]?
    private var isPCABuildInFlight = false
    private var pcaBuildGeneration: UInt64 = 0

    private struct DendrogramWorkingCluster {
        let nodeId: Int
        let vector: [Float]
        let weight: Float
        let leafCount: Int
        let speakerHistogram: [Int: Int]
    }

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
        resetClusterState()
        
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

        updateClusterVisualization(forcePlotRefresh: true)
        
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

        resetClusterState()
        
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

            updateClusterVisualization()
            
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
            let newDiarizer = SortformerDiarizer(
                config: globalConfig,
                postProcessingConfig: globalTimelineConfig,
                embeddingConfig: globalEmbeddingConfig
            )
            let models = try await SortformerModels.loadFromHuggingFace(config: globalConfig)
            newDiarizer.initialize(models: models)
            
            // Initialize embedding manager
            let newEmbeddingManager = EmbeddingManager(
                frameDurationSeconds: globalConfig.frameDurationSeconds
            )
            try newEmbeddingManager.initialize()
            
            self.diarizer = newDiarizer
            self.timeline = newDiarizer.timeline
            resetClusterState()
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
            let samples = audioBufferLock.withLock {
                let result = audioBuffer
                audioBuffer = []
                return result
            }
            
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
            updateClusterVisualization()
            
            // Don't update graph during streaming - it causes lag and isn't visible anyway
            // Graph will be updated when clustering is triggered
        } catch {
            statusMessage = "Processing error: \(error.localizedDescription)"
            print("Diarizer processing error: \(error)")
        }
    }

    private func resetClusterState() {
        cancelPendingPCAWork()
        lastStreamingPCARefreshTime = 0
        segmentCentroidDistances = [:]
        embeddingSegmentCentroidDistances = [:]
        kmeansPCAPlotModel = .empty
    }

    private func updateClusterVisualization(forcePlotRefresh: Bool = false) {
        guard let tl = timeline else {
            cancelPendingPCAWork()
            segmentCentroidDistances = [:]
            embeddingSegmentCentroidDistances = [:]
            kmeansPCAPlotModel = .empty
            return
        }

        let distanceMaps = makeDistanceMaps(from: tl)
        segmentCentroidDistances = distanceMaps.bySegmentID
        embeddingSegmentCentroidDistances = distanceMaps.byEmbeddingSegmentID
        if shouldRefreshPCAPlot(force: forcePlotRefresh) {
            schedulePCAPlotRefresh(from: tl)
        }
    }

    private func shouldRefreshPCAPlot(force: Bool) -> Bool {
        let now = Date.timeIntervalSinceReferenceDate
        if force || !isRecording {
            lastStreamingPCARefreshTime = now
            return true
        }
        if now - lastStreamingPCARefreshTime >= streamingPCARefreshInterval {
            lastStreamingPCARefreshTime = now
            return true
        }
        return false
    }


    private func makeDistanceMaps(
        from timeline: SortformerTimeline
    ) -> (bySegmentID: [UInt64: Float], byEmbeddingSegmentID: [UUID: Float]) {
        let activeProfilesBySlot = timeline.activeSpeakers
        let sortedActive = activeProfilesBySlot.keys.sorted()

        func profileCentroids(_ profile: SpeakerProfile) -> [SpeakerClusterCentroid] {
            if !profile.tentativeClusters.isEmpty {
                return profile.tentativeClusters
            }
            return profile.finalizedClusters
        }

        var clustersBySlot: [Int: [[Float]]] = [:]
        var allClusterVectors: [[Float]] = []
        for slot in sortedActive {
            guard let profile = activeProfilesBySlot[slot] else { continue }
            let vectors = profileCentroids(profile).compactMap { centroid -> [Float]? in
                let vector = normalizedVector(Array(centroid.buffer))
                return vector.isEmpty ? nil : vector
            }
            if !vectors.isEmpty {
                clustersBySlot[slot] = vectors
                allClusterVectors.append(contentsOf: vectors)
            }
        }

        // Fallback: if no active centroids are available, use inactive profiles.
        if allClusterVectors.isEmpty {
            for profile in timeline.inactiveSpeakers {
                let vectors = profileCentroids(profile).compactMap { centroid -> [Float]? in
                    let vector = normalizedVector(Array(centroid.buffer))
                    return vector.isEmpty ? nil : vector
                }
                allClusterVectors.append(contentsOf: vectors)
            }
        }

        guard !allClusterVectors.isEmpty else {
            return ([:], [:])
        }

        var bySegmentID: [UInt64: Float] = [:]
        var byEmbeddingSegmentID: [UUID: Float] = [:]
        let embeddingSegments = timeline.embeddingSegments + timeline.tentativeEmbeddingSegments
        byEmbeddingSegmentID.reserveCapacity(embeddingSegments.count)
        
        var timelineSegmentsBySlot: [Int: [SpeakerSegment]] = [:]
        for (slot, segments) in timeline.finalizedSegments.enumerated() where !segments.isEmpty {
            timelineSegmentsBySlot[slot, default: []].append(contentsOf: segments)
        }
        for (slot, segments) in timeline.tentativeSegments.enumerated() where !segments.isEmpty {
            timelineSegmentsBySlot[slot, default: []].append(contentsOf: segments)
        }

        for embeddingSegment in embeddingSegments {
            guard let segmentVector = centroidVector(for: embeddingSegment) else {
                continue
            }

            let candidateClusterVectors = clustersBySlot[embeddingSegment.speakerId] ?? allClusterVectors
            var bestDistance = Float.greatestFiniteMagnitude
            for clusterVector in candidateClusterVectors {
                bestDistance = min(bestDistance, cosineDistance(segmentVector, clusterVector))
            }
            guard bestDistance.isFinite else {
                continue
            }

            byEmbeddingSegmentID[embeddingSegment.id] = bestDistance
            
            guard let candidates = timelineSegmentsBySlot[embeddingSegment.speakerId] else {
                continue
            }
            for segment in candidates
            where segment.endFrame > embeddingSegment.startFrame
                && segment.startFrame < embeddingSegment.endFrame {
                if let current = bySegmentID[segment.id] {
                    bySegmentID[segment.id] = min(current, bestDistance)
                } else {
                    bySegmentID[segment.id] = bestDistance
                }
            }
        }

        return (bySegmentID, byEmbeddingSegmentID)
    }

    private func centroidVector(for embeddingSegment: EmbeddingSegment) -> [Float]? {
        if let centroid = embeddingSegment.centroid {
            let vector = normalizedVector(Array(centroid.buffer))
            return vector.isEmpty ? nil : vector
        }

        let embeddings = embeddingSegment.embeddings
        guard let first = embeddings.first else {
            return nil
        }

        let dimension = first.count
        guard dimension > 0 else { return nil }

        var accumulator = Array(repeating: Float.zero, count: dimension)
        var totalWeight: Float = 0

        for embedding in embeddings {
            guard embedding.count == dimension else { continue }
            let weight = Float(max(embedding.length, 1))
            totalWeight += weight
            let buffer = embedding.bufferView
            for dim in 0..<dimension {
                accumulator[dim] += buffer[dim] * weight
            }
        }

        guard totalWeight > 0 else {
            return nil
        }

        for dim in 0..<dimension {
            accumulator[dim] /= totalWeight
        }

        let normalized = normalizedVector(accumulator)
        return normalized.isEmpty ? nil : normalized
    }

    private func dominantSpeakerSlot(from histogram: [Int: Int]) -> Int {
        guard !histogram.isEmpty else { return -1 }
        return histogram.max { lhs, rhs in
            if lhs.value == rhs.value {
                return lhs.key > rhs.key
            }
            return lhs.value < rhs.value
        }?.key ?? -1
    }

    private func mergedSpeakerHistogram(_ lhs: [Int: Int], _ rhs: [Int: Int]) -> [Int: Int] {
        var merged = lhs
        for (speaker, count) in rhs {
            merged[speaker, default: 0] += count
        }
        return merged
    }

    private func mergedClusterVector(left: DendrogramWorkingCluster, right: DendrogramWorkingCluster) -> [Float] {
        let dimension = max(left.vector.count, right.vector.count)
        guard dimension > 0 else { return [] }

        let leftWeight = max(left.weight, 1e-6)
        let rightWeight = max(right.weight, 1e-6)
        let totalWeight = leftWeight + rightWeight
        var merged = Array(repeating: Float.zero, count: dimension)

        for dim in 0..<dimension {
            let leftValue = dim < left.vector.count ? left.vector[dim] : 0
            let rightValue = dim < right.vector.count ? right.vector[dim] : 0
            merged[dim] = (leftValue * leftWeight + rightValue * rightWeight) / totalWeight
        }
        return normalizedVector(merged)
    }

    private func cosineDistance(_ lhs: [Float], _ rhs: [Float]) -> Float {
        let dimension = min(lhs.count, rhs.count)
        guard dimension > 0 else {
            return 1
        }

        var dot: Float = 0
        var lhsNormSq: Float = 0
        var rhsNormSq: Float = 0

        for dim in 0..<dimension {
            dot += lhs[dim] * rhs[dim]
            lhsNormSq += lhs[dim] * lhs[dim]
            rhsNormSq += rhs[dim] * rhs[dim]
        }

        let norm = sqrt(max(lhsNormSq, 1e-12) * max(rhsNormSq, 1e-12))
        guard norm > 0 else { return 1 }
        return max(0, 1 - dot / norm)
    }

    private func normalizedVector(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return [] }

        var sumSquares: Float = 0
        for value in vector {
            sumSquares += value * value
        }
        let norm = sqrt(max(sumSquares, 1e-12))
        guard norm > 0 else { return vector }
        return vector.map { $0 / norm }
    }

    private func orderedNodeIDs(_ lhs: Int, _ rhs: Int) -> (Int, Int) {
        lhs <= rhs ? (lhs, rhs) : (rhs, lhs)
    }
    
    private struct PCAProjection {
        let mean: [Float]
        let components: [[Float]]
    }
    
    private struct PCAPointRaw {
        let id: UUID
        let vector: [Float]
        let speakerID: Int
        let slot: Int
        let clusterID: Int
        let isInactive: Bool
    }
    
    private struct PCAClusterSeed {
        let id: UUID
        let clusterID: Int
        let speakerID: Int
        let slot: Int
        let centerVector: [Float]
        let isInactive: Bool
    }
    
    private enum PCAPlotDetail {
        case streaming
        case full
    }
    
    private func makePCARawPoints(from timeline: SortformerTimeline, detail: PCAPlotDetail) -> [PCAPointRaw] {
        let embeddingSegments = timeline.embeddingSegments + timeline.tentativeEmbeddingSegments
        let clusterSeeds = makeClusterSeeds(from: timeline)
        guard !embeddingSegments.isEmpty || !clusterSeeds.isEmpty else { return [] }
        
        let activeSpeakerIDBySlot = Dictionary(
            uniqueKeysWithValues: timeline.activeSpeakers.map { ($0.key, $0.value.speakerId) }
        )
        let activeSpeakerIDs = Set(activeSpeakerIDBySlot.values)
        let seedsBySpeakerID = Dictionary(grouping: clusterSeeds, by: \.speakerID)
        
        var rawPoints: [PCAPointRaw] = []
        if detail == .streaming {
            rawPoints.reserveCapacity(embeddingSegments.count + clusterSeeds.count)
        } else {
            rawPoints.reserveCapacity(embeddingSegments.reduce(0) { $0 + max(1, $1.embeddings.count) } + clusterSeeds.count)
        }
        var speakersWithLivePoints: Set<Int> = []
        
        for segment in embeddingSegments {
            let slot = segment.speakerId
            let speakerID = activeSpeakerIDBySlot[slot] ?? slot
            let clusterCandidates = seedsBySpeakerID[speakerID] ?? []
            let isInactive = !activeSpeakerIDs.contains(speakerID)
            speakersWithLivePoints.insert(speakerID)
            
            if detail == .streaming {
                guard let vector = centroidVector(for: segment) else {
                    continue
                }
                let clusterID = nearestClusterID(for: vector, clusters: clusterCandidates)
                rawPoints.append(
                    PCAPointRaw(
                        id: segment.id,
                        vector: vector,
                        speakerID: speakerID,
                        slot: slot,
                        clusterID: clusterID,
                        isInactive: isInactive
                    )
                )
                continue
            }
            
            let segmentVector = centroidVector(for: segment)
            let clusterID = segmentVector.map { nearestClusterID(for: $0, clusters: clusterCandidates) } ?? -1
            
            if segment.embeddings.isEmpty {
                guard let segmentVector else { continue }
                rawPoints.append(
                    PCAPointRaw(
                        id: segment.id,
                        vector: segmentVector,
                        speakerID: speakerID,
                        slot: slot,
                        clusterID: clusterID,
                        isInactive: isInactive
                    )
                )
                continue
            }
            
            for embedding in segment.embeddings {
                rawPoints.append(
                    PCAPointRaw(
                        id: embedding.id,
                        vector: Array(embedding.bufferView),
                        speakerID: speakerID,
                        slot: slot,
                        clusterID: clusterID,
                        isInactive: isInactive
                    )
                )
            }
        }
        
        for seed in clusterSeeds where seed.isInactive || !speakersWithLivePoints.contains(seed.speakerID) {
            rawPoints.append(
                PCAPointRaw(
                    id: seed.id,
                    vector: seed.centerVector,
                    speakerID: seed.speakerID,
                    slot: seed.slot,
                    clusterID: seed.clusterID,
                    isInactive: seed.isInactive
                )
            )
        }
        
        return rawPoints
    }

    nonisolated private static func makeKMeansPCAPlotModel(from rawPoints: [PCAPointRaw]) -> KMeansPCAPlotModel {
        guard !rawPoints.isEmpty else {
            return .empty
        }
        
        let projection = computePCAProjection(vectors: rawPoints.map(\.vector), components: 3)
        
        var points = rawPoints.map { point in
            KMeansPCAPlotPoint(
                id: point.id,
                position: project(point.vector, with: projection),
                speakerID: point.speakerID,
                slot: point.slot,
                clusterID: point.clusterID,
                isInactive: point.isInactive
            )
        }
        
        normalizeScale(points: &points)
        return KMeansPCAPlotModel(points: points)
    }

    private func schedulePCAPlotRefresh(from timeline: SortformerTimeline) {
        let detail: PCAPlotDetail = isRecording ? .streaming : .full
        let pointLimit = isRecording ? streamingPCAPointLimit : finalizedPCAPointLimit
        let rawPoints = Self.downsampleRawPoints(
            makePCARawPoints(from: timeline, detail: detail),
            limit: pointLimit
        )
        guard !rawPoints.isEmpty else {
            cancelPendingPCAWork()
            kmeansPCAPlotModel = .empty
            return
        }
        
        pendingPCARawPoints = rawPoints
        startNextPCABuildIfNeeded()
    }
    
    private func cancelPendingPCAWork() {
        pendingPCARawPoints = nil
        pcaBuildGeneration &+= 1
    }
    
    private func startNextPCABuildIfNeeded() {
        guard !isPCABuildInFlight,
              let rawPoints = pendingPCARawPoints else {
            return
        }
        
        pendingPCARawPoints = nil
        isPCABuildInFlight = true
        let cancellationToken = pcaBuildGeneration
        
        pcaBuildQueue.async { [rawPoints] in
            let plotModel = Self.makeKMeansPCAPlotModel(from: rawPoints)
            Task { @MainActor [weak self] in
                guard let self else { return }
                if cancellationToken == self.pcaBuildGeneration {
                    self.kmeansPCAPlotModel = plotModel
                }
                self.isPCABuildInFlight = false
                self.startNextPCABuildIfNeeded()
            }
        }
    }
    
    private func makeClusterSeeds(from timeline: SortformerTimeline) -> [PCAClusterSeed] {
        var results: [PCAClusterSeed] = []
        results.reserveCapacity(64)
        var nextClusterID = 0
        
        let orderedActive = timeline.activeSpeakers.sorted { $0.key < $1.key }
        for (slot, profile) in orderedActive {
            let clusters = profile.finalizedClusters + profile.tentativeClusters
            for cluster in clusters {
                let vector = normalizedVector(Array(cluster.buffer))
                guard !vector.isEmpty else { continue }
                
                results.append(
                    PCAClusterSeed(
                        id: cluster.id,
                        clusterID: nextClusterID,
                        speakerID: profile.speakerId,
                        slot: slot,
                        centerVector: vector,
                        isInactive: false
                    )
                )
                nextClusterID += 1
            }
        }
        
        for profile in timeline.inactiveSpeakers {
            let clusters = profile.finalizedClusters + profile.tentativeClusters
            for cluster in clusters {
                let vector = normalizedVector(Array(cluster.buffer))
                guard !vector.isEmpty else { continue }
                
                results.append(
                    PCAClusterSeed(
                        id: cluster.id,
                        clusterID: nextClusterID,
                        speakerID: profile.speakerId,
                        slot: -1,
                        centerVector: vector,
                        isInactive: true
                    )
                )
                nextClusterID += 1
            }
        }
        
        return results
    }
    
    private func nearestClusterID(for vector: [Float], clusters: [PCAClusterSeed]) -> Int {
        guard !clusters.isEmpty else { return -1 }
        var bestDistance = Float.greatestFiniteMagnitude
        var bestClusterID = -1
        
        for cluster in clusters {
            let distance = cosineDistance(vector, cluster.centerVector)
            if distance < bestDistance {
                bestDistance = distance
                bestClusterID = cluster.clusterID
            }
        }
        
        return bestClusterID
    }
    
    nonisolated private static func downsampleRawPoints(_ points: [PCAPointRaw], limit: Int) -> [PCAPointRaw] {
        guard limit > 0, points.count > limit else { return points }
        
        let inactivePoints = points.filter(\.isInactive)
        if inactivePoints.count >= limit {
            return Array(inactivePoints.prefix(limit))
        }
        
        let activePoints = points.filter { !$0.isInactive }
        guard !activePoints.isEmpty else {
            return Array(inactivePoints.prefix(limit))
        }
        
        let activeBudget = max(0, limit - inactivePoints.count)
        guard activeBudget > 0 else {
            return Array(inactivePoints.prefix(limit))
        }
        
        let sortedActive = activePoints.sorted { lhs, rhs in
            if lhs.speakerID != rhs.speakerID {
                return lhs.speakerID < rhs.speakerID
            }
            if lhs.clusterID != rhs.clusterID {
                return lhs.clusterID < rhs.clusterID
            }
            return lhs.id.uuidString < rhs.id.uuidString
        }
        
        var sampled = inactivePoints
        sampled.reserveCapacity(limit)
        let stride = Double(sortedActive.count) / Double(activeBudget)
        var cursor = 0.0
        
        for _ in 0..<activeBudget {
            sampled.append(sortedActive[min(Int(cursor), sortedActive.count - 1)])
            cursor += stride
        }
        
        return sampled
    }
    
    nonisolated private static func computePCAProjection(vectors: [[Float]], components k: Int) -> PCAProjection {
        guard let first = vectors.first, !first.isEmpty else {
            return PCAProjection(mean: [], components: [])
        }
        
        let dimension = first.count
        let count = vectors.count
        var mean = Array(repeating: Float.zero, count: dimension)
        
        for vector in vectors {
            guard vector.count == dimension else { continue }
            for i in 0..<dimension {
                mean[i] += vector[i]
            }
        }
        
        let countFloat = Float(max(count, 1))
        for i in 0..<dimension {
            mean[i] /= countFloat
        }
        
        var covariance = Array(repeating: Float.zero, count: dimension * dimension)
        if count > 1 {
            for vector in vectors {
                guard vector.count == dimension else { continue }
                var centered = Array(repeating: Float.zero, count: dimension)
                for i in 0..<dimension {
                    centered[i] = vector[i] - mean[i]
                }
                
                for row in 0..<dimension {
                    let rowValue = centered[row]
                    for col in row..<dimension {
                        covariance[row * dimension + col] += rowValue * centered[col]
                    }
                }
            }
            
            let normalizer = Float(count - 1)
            for row in 0..<dimension {
                for col in row..<dimension {
                    let value = covariance[row * dimension + col] / normalizer
                    covariance[row * dimension + col] = value
                    covariance[col * dimension + row] = value
                }
            }
        }
        
        var principalComponents: [[Float]] = []
        principalComponents.reserveCapacity(k)
        
        for componentIndex in 0..<k {
            var vector = Array(repeating: Float.zero, count: dimension)
            for i in 0..<dimension {
                vector[i] = Float(((i + 3) * (componentIndex + 5)) % 17 + 1)
            }
            vector = normalizedVector(vector)
            
            for _ in 0..<48 {
                var next = matrixVectorMultiply(matrix: covariance, vector: vector, dimension: dimension)
                
                for previous in principalComponents where previous.count == dimension {
                    let projection = dot(next, previous)
                    for i in 0..<dimension {
                        next[i] -= projection * previous[i]
                    }
                }
                
                let normalized = normalizedVector(next)
                if normalized.allSatisfy({ abs($0) < 1e-6 }) {
                    break
                }
                vector = normalized
            }
            
            principalComponents.append(vector)
        }
        
        while principalComponents.count < k {
            var axis = Array(repeating: Float.zero, count: dimension)
            axis[min(principalComponents.count, dimension - 1)] = 1
            principalComponents.append(axis)
        }
        
        return PCAProjection(mean: mean, components: principalComponents)
    }
    
    nonisolated private static func matrixVectorMultiply(matrix: [Float], vector: [Float], dimension: Int) -> [Float] {
        guard vector.count == dimension, matrix.count == dimension * dimension else {
            return Array(repeating: Float.zero, count: dimension)
        }
        
        var result = Array(repeating: Float.zero, count: dimension)
        for row in 0..<dimension {
            var value: Float = 0
            for col in 0..<dimension {
                value += matrix[row * dimension + col] * vector[col]
            }
            result[row] = value
        }
        return result
    }
    
    nonisolated private static func dot(_ lhs: [Float], _ rhs: [Float]) -> Float {
        let count = min(lhs.count, rhs.count)
        guard count > 0 else { return 0 }
        var result: Float = 0
        for i in 0..<count {
            result += lhs[i] * rhs[i]
        }
        return result
    }
    
    nonisolated private static func project(_ vector: [Float], with projection: PCAProjection) -> SIMD3<Float> {
        guard !projection.components.isEmpty, !projection.mean.isEmpty else {
            return SIMD3<Float>(0, 0, 0)
        }
        
        let dimension = min(vector.count, projection.mean.count)
        guard dimension > 0 else {
            return SIMD3<Float>(0, 0, 0)
        }
        
        var centered = Array(repeating: Float.zero, count: dimension)
        for i in 0..<dimension {
            centered[i] = vector[i] - projection.mean[i]
        }
        
        let x = projection.components.indices.contains(0) ? dot(centered, projection.components[0]) : 0
        let y = projection.components.indices.contains(1) ? dot(centered, projection.components[1]) : 0
        let z = projection.components.indices.contains(2) ? dot(centered, projection.components[2]) : 0
        return SIMD3<Float>(x, y, z)
    }
    
    nonisolated private static func normalizeScale(points: inout [KMeansPCAPlotPoint]) {
        var maxAbs: Float = 0
        
        for point in points {
            maxAbs = max(maxAbs, abs(point.position.x))
            maxAbs = max(maxAbs, abs(point.position.y))
            maxAbs = max(maxAbs, abs(point.position.z))
        }

        let scale = max(maxAbs * 1.12, 1e-6)
        
        points = points.map { point in
            KMeansPCAPlotPoint(
                id: point.id,
                position: point.position / scale,
                speakerID: point.speakerID,
                slot: point.slot,
                clusterID: point.clusterID,
                isInactive: point.isInactive
            )
        }
    }

    nonisolated private static func normalizedVector(_ vector: [Float]) -> [Float] {
        guard !vector.isEmpty else { return [] }
        
        var sumSquares: Float = 0
        for value in vector {
            sumSquares += value * value
        }
        let norm = sqrt(max(sumSquares, 1e-12))
        guard norm > 0 else { return vector }
        return vector.map { $0 / norm }
    }
    
    /// Print all segments in CSV format: speaker_id,start_time,end_time
    private func printSegments() {
        guard let tl = timeline else { return }
        
        print("\n--- Diarization Segments ---")
        print("speaker_id,start_time,end_time")
        
        // Collect all segments and sort by start time
        let allSegments = tl.finalizedSegments.flatMap { $0 }.sorted { $0.startTime < $1.startTime }
        
        for segment in allSegments {
            let startStr = String(format: "%.2f", segment.startTime)
            let endStr = String(format: "%.2f", segment.endTime)
            print("\(segment.slot),\(startStr),\(endStr)")
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
    static func segmentKey(_ segment: SpeakerSegment) -> String {
        return "\(segment.id)"
    }
    
    /// Set annotation for a segment
    func setAnnotation(for segment: SpeakerSegment, label: String) {
        let key = Self.segmentKey(segment)
        if label.isEmpty {
            segmentAnnotations.removeValue(forKey: key)
        } else {
            segmentAnnotations[key] = label
        }
    }
    
    /// Get annotation for a segment (nil if not annotated)
    func getAnnotation(for segment: SpeakerSegment) -> String? {
        let key = Self.segmentKey(segment)
        return segmentAnnotations[key]
    }
    
    /// Export segments to a text file
    func exportSegments(to url: URL) {
        guard let tl = timeline else { return }
        
        var output = "speaker_id,start_time,end_time\n"
        
        // Collect all segments and sort by start time
        let allSegments = tl.finalizedSegments.flatMap { $0 }.sorted { $0.startTime < $1.startTime }
        
        for segment in allSegments {
            // Use annotation if available, otherwise use original speaker index
            let label = getAnnotation(for: segment) ?? String(segment.slot)
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
