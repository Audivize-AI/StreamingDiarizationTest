//
//  EmbeddingManager.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/15/26.
//

import Foundation
import Accelerate

/// Manages speaker embedding extraction asynchronously
/// Handles segment merging, pruning, and embedding lifecycle
public class EmbeddingManager {
    
    // MARK: - Configuration
    
    /// Embedding extraction configuration
    public let config: EmbeddingConfig
    
    /// TitaNet configuration
    public let titanetConfig: TitaNetConfig
    
    // MARK: - State
    
    /// The embedding extractor
    private var extractor: TitaNetEmbeddingExtractor?
    
    /// Audio buffer for extraction (circular buffer of audio samples)
    private var audioBuffer: [Float] = []
    
    /// Frame offset of the audio buffer start
    private var audioBufferStartFrame: Int = 0
    
    /// Maximum audio buffer size in samples
    private let maxAudioBufferSamples: Int
    
    /// Sample rate for audio
    private let sampleRate: Float = 16_000
    
    /// Samples per frame
    private let samplesPerFrame: Int
    
    /// Queue for background extraction
    private let extractionQueue = DispatchQueue(label: "com.sortformer.embedding", qos: .userInitiated)
    
    /// Lock for thread-safe access
    private let lock = NSLock()
    
    /// Currently processing requests by key (speaker_start_end) to avoid duplicates
    private var processingRequests: Set<String> = []
    
    /// Orphaned embeddings that couldn't be assigned to a segment
    /// Will try to assign on subsequent processRequests calls
    private var orphanedEmbeddings: [(embedding: [Float], speakerIndex: Int, startFrame: Int, endFrame: Int, age: Int)] = []
    
    /// Maximum age before orphans are discarded
    private let maxOrphanAge: Int = 10
    
    /// Logger
    private static let logger = AppLogger(category: "EmbeddingManager")
    
    // MARK: - Initialization
    
    public init(
        config: EmbeddingConfig = EmbeddingConfig(),
        titanetConfig: TitaNetConfig = TitaNetConfig(),
        frameDurationSeconds: Float = 0.08  // Match Sortformer's frame duration
    ) {
        self.config = config
        self.titanetConfig = titanetConfig
        self.samplesPerFrame = Int(frameDurationSeconds * sampleRate)
        
        // Buffer enough audio for the max embedding plus some margin
        self.maxAudioBufferSamples = titanetConfig.maxInputSamples * 4
        
        // Initialize extractor lazily on first use
    }
    
    /// Initialize the extractor (call before first use)
    public func initialize() throws {
        lock.lock()
        defer { lock.unlock() }
        
        if extractor == nil {
            extractor = try TitaNetEmbeddingExtractor(config: titanetConfig)
        }
    }
    
    // MARK: - Audio Buffer Management
    
    /// Append audio samples to the buffer
    public func appendAudio(_ samples: [Float]) {
        lock.lock()
        defer { lock.unlock() }
        
        audioBuffer.append(contentsOf: samples)
        
        // Trim buffer if too large
        if audioBuffer.count > maxAudioBufferSamples {
            let samplesToRemove = audioBuffer.count - maxAudioBufferSamples
            let framesToRemove = samplesToRemove / samplesPerFrame
            let actualSamplesToRemove = framesToRemove * samplesPerFrame
            
            audioBuffer.removeFirst(actualSamplesToRemove)
            audioBufferStartFrame += framesToRemove
        }
    }
    
    /// Get audio slice for a frame range
    private func getAudioSlice(startFrame: Int, endFrame: Int) -> [Float]? {
        let startSample = (startFrame - audioBufferStartFrame) * samplesPerFrame
        let endSample = (endFrame - audioBufferStartFrame) * samplesPerFrame
        
        guard startSample >= 0, endSample <= audioBuffer.count else {
            return nil
        }
        
        return Array(audioBuffer[startSample..<endSample])
    }
    
    /// Reset the audio buffer
    public func resetAudioBuffer() {
        lock.lock()
        defer { lock.unlock() }
        
        audioBuffer.removeAll(keepingCapacity: true)
        audioBufferStartFrame = 0
    }
    
    /// Get the frame range currently available in the audio buffer
    public var availableAudioFrameRange: Range<Int> {
        lock.lock()
        defer { lock.unlock() }
        
        let endFrame = audioBufferStartFrame + (audioBuffer.count / samplesPerFrame)
        return audioBufferStartFrame..<endFrame
    }
    
    // MARK: - Embedding Extraction
    
    /// Process pending embedding requests from a timeline asynchronously
    public func processRequests(
        from timeline: SortformerTimeline,
        completion: ((Int) -> Void)? = nil
    ) {
        let requests = timeline.consumePendingEmbeddingRequests()
        
        guard !requests.isEmpty else {
            completion?(0)
            return
        }
        
        // Get available audio range and filter requests
        let availableRange = availableAudioFrameRange
        let validRequests = requests.filter { req in
            req.startFrame >= availableRange.lowerBound && req.endFrame <= availableRange.upperBound
        }
        
        // Log filtered requests for debugging
        let filteredCount = requests.count - validRequests.count
        if filteredCount > 0 {
            Self.logger.debug("Filtered \(filteredCount)/\(requests.count) requests. Available range: [\(availableRange.lowerBound)-\(availableRange.upperBound)]")
            for req in requests where !(req.startFrame >= availableRange.lowerBound && req.endFrame <= availableRange.upperBound) {
                Self.logger.debug("  Filtered: [\(req.startFrame)-\(req.endFrame)] (outside buffer)")
            }
        }
        
        guard !validRequests.isEmpty else {
            completion?(0)
            return
        }
        
        // Capture timeline reference - we'll access current segments at extraction time
        // to ensure we add embeddings to the current segments, not stale ones
        
        extractionQueue.async { [weak self, weak timeline] in
            guard let self = self, let timeline = timeline else { return }
            
            var successCount = 0
            
            // First, try to assign orphaned embeddings to current segments
            self.lock.lock()
            var remainingOrphans: [(embedding: [Float], speakerIndex: Int, startFrame: Int, endFrame: Int, age: Int)] = []
            let currentOrphans = self.orphanedEmbeddings
            self.orphanedEmbeddings = []
            self.lock.unlock()
            
            // Search both finalized and tentative segments for matches
            let currentSegments = timeline.embeddingSegments
            let tentativeSegments = timeline.tentativeEmbeddingSegments
            let allSegments = currentSegments + tentativeSegments
            
            for orphan in currentOrphans {
                // First, try to find a segment with matching speaker
                var matchedSegment: EmbeddingSegment? = allSegments.first(where: { seg in
                    seg.speakerIndex == orphan.speakerIndex &&
                    seg.startFrame <= orphan.startFrame &&
                    seg.endFrame >= orphan.endFrame
                })
                
                // Fallback: if no speaker match, try frame range only (speaker assignments can change in tentative)
                if matchedSegment == nil {
                    matchedSegment = allSegments.first(where: { seg in
                        seg.startFrame <= orphan.startFrame &&
                        seg.endFrame >= orphan.endFrame
                    })
                    if matchedSegment != nil {
                        Self.logger.debug("Speaker-agnostic match for orphan [\(orphan.startFrame)-\(orphan.endFrame)] (was speaker \(orphan.speakerIndex), now \(matchedSegment!.speakerIndex))")
                    }
                }
                
                if let segment = matchedSegment {
                    // Found a matching segment!
                    let titanetEmb = TitaNetEmbedding(
                        embedding: orphan.embedding,
                        startFrame: orphan.startFrame,
                        endFrame: orphan.endFrame
                    )
                    
                    // Only add if not already present
                    if !segment.embeddings.contains(where: { $0.startFrame == orphan.startFrame && $0.endFrame == orphan.endFrame }) {
                        segment.addEmbeddings([titanetEmb])
                        Self.logger.debug("Assigned orphaned embedding [\(orphan.startFrame)-\(orphan.endFrame)] to segment")
                    }
                } else {
                    // Still no match - keep as orphan if not too old
                    let newAge = orphan.age + 1
                    if newAge <= self.maxOrphanAge {
                        remainingOrphans.append((orphan.embedding, orphan.speakerIndex, orphan.startFrame, orphan.endFrame, newAge))
                    }
                }
            }
            
            // Process new requests
            for request in validRequests {
                // Get CURRENT segments from timeline at extraction time
                // ALWAYS check both finalized and tentative
                let currentSegments = timeline.embeddingSegments
                let tentativeSegments = timeline.tentativeEmbeddingSegments
                let allSegments = currentSegments + tentativeSegments
                
                // First try speaker-specific match
                var targetSegment = allSegments.first { seg in
                    seg.speakerIndex == request.speakerIndex &&
                    seg.startFrame <= request.startFrame &&
                    seg.endFrame >= request.endFrame
                }
                
                // Fallback: speaker-agnostic match (speaker IDs can change in tentative predictions)
                if targetSegment == nil {
                    targetSegment = allSegments.first { seg in
                        seg.startFrame <= request.startFrame &&
                        seg.endFrame >= request.endFrame
                    }
                }
                
                // Skip if segment already has an embedding covering this range (strict overlap)
                if let segment = targetSegment {
                    let alreadyHasEmbedding = segment.embeddings.contains { emb in
                        // Check if existing embedding covers 90% of the requested range
                        // Or if the requested range is fully contained in an existing embedding
                        let intersectionStart = max(emb.startFrame, request.startFrame)
                        let intersectionEnd = min(emb.endFrame, request.endFrame)
                        let overlap = max(0, intersectionEnd - intersectionStart)
                        
                        let requestLen = request.endFrame - request.startFrame
                        let coverage = Float(overlap) / Float(requestLen)
                        
                        return coverage > 0.9
                    }
                    if alreadyHasEmbedding {
                        continue
                    }
                }
                
                // Deduplicate by request frame range (not segment object which changes each update)
                let requestKey = "\(request.speakerIndex)_\(request.startFrame)_\(request.endFrame)"
                
                self.lock.lock()
                let isProcessing = self.processingRequests.contains(requestKey)
                if !isProcessing {
                    self.processingRequests.insert(requestKey)
                }
                self.lock.unlock()
                
                if isProcessing {
                    continue
                }
                
                defer {
                    self.lock.lock()
                    self.processingRequests.remove(requestKey)
                    self.lock.unlock()
                }
                
                // Get audio for this request
                self.lock.lock()
                let audioSlice = self.getAudioSlice(startFrame: request.startFrame, endFrame: request.endFrame)
                let extractorCopy = self.extractor
                self.lock.unlock()
                
                guard let audio = audioSlice, let extractor = extractorCopy else {
                    Self.logger.warning("Failed to get audio for embedding request (\(request.startFrame)-\(request.endFrame))")
                    continue
                }
                
                do {
                    let embedding = try extractor.extractEmbedding(from: audio)
                    
                    // Try to add to segment
                    if let segment = targetSegment {
                        if request.addEmbedding(embedding, fallbackSegment: segment) {
                            successCount += 1
                        }
                    } else {
                        // No segment found - cache as orphan
                        Self.logger.debug("Caching orphan embedding [\(request.startFrame)-\(request.endFrame)] speaker \(request.speakerIndex)")
                        remainingOrphans.append((embedding, request.speakerIndex, request.startFrame, request.endFrame, 0))
                    }
                } catch {
                    Self.logger.error("Embedding extraction failed for [\(request.startFrame)-\(request.endFrame)] (audio: \(audio.count) samples): \(error)")
                }
            }
            
            // Store remaining orphans
            self.lock.lock()
            self.orphanedEmbeddings.append(contentsOf: remainingOrphans)
            self.lock.unlock()
            
            DispatchQueue.main.async {
                completion?(successCount)
            }
        }
    }
    
    /// Process a single request synchronously (for testing)
    public func processRequestSync(_ request: EmbeddingExtractionRequest) throws {
        lock.lock()
        let audioSlice = getAudioSlice(startFrame: request.startFrame, endFrame: request.endFrame)
        let extractorCopy = extractor
        lock.unlock()
        
        guard let audio = audioSlice else {
            throw EmbeddingManagerError.audioNotAvailable
        }
        
        guard let extractor = extractorCopy else {
            throw EmbeddingManagerError.extractorNotInitialized
        }
        
        let embedding = try extractor.extractEmbedding(from: audio)
        request.addEmbedding(embedding)
    }
    
    // MARK: - Segment Management
    
    /// Merge adjacent embedding segments of the same speaker
    /// Returns the merged segment if a merge occurred, nil otherwise
    public func mergeAdjacentSegments(
        _ segments: inout [EmbeddingSegment],
        maxGapFrames: Int
    ) {
        guard segments.count > 1 else { return }
        
        // Sort by start frame first
        segments.sort { $0.startFrame < $1.startFrame }
        
        var i = 0
        while i < segments.count - 1 {
            let current = segments[i]
            let next = segments[i + 1]
            
            // Check if same speaker and close enough
            if current.speakerIndex == next.speakerIndex &&
               next.startFrame - current.endFrame <= maxGapFrames {
                
                // Merge: extend current to include next
                current.endFrame = next.endFrame
                
                // Transfer embeddings from next to current
                current.addEmbeddings(next.embeddings)
                
                // Remove the merged segment
                segments.remove(at: i + 1)
                // Don't increment i - check if we can merge more
            } else {
                i += 1
            }
        }
    }
    
    /// Prune segments that are too short or have insufficient embeddings
    public func pruneSegments(
        _ segments: inout [EmbeddingSegment],
        minFrames: Int,
        minEmbeddings: Int = 0
    ) {
        segments.removeAll { segment in
            segment.length < minFrames ||
            (minEmbeddings > 0 && segment.embeddings.count < minEmbeddings)
        }
    }
    
    /// Update embedding segments from timeline changes
    /// Handles merging, splitting, and pruning
    public func updateSegments(
        in timeline: SortformerTimeline,
        mergeGapFrames: Int = 5,
        minSegmentFrames: Int = 10
    ) {
        // Merge adjacent segments
        mergeAdjacentSegments(&timeline.embeddingSegments, maxGapFrames: mergeGapFrames)
        
        // Prune short segments
        pruneSegments(&timeline.embeddingSegments, minFrames: minSegmentFrames)
        
        // Discard bad embeddings and request new ones
        for segment in timeline.embeddingSegments {
            _ = segment.discardBadEmbeddings(maxOutsideFrames: config.maxOutsideFrames)
            
            let requests = segment.getEmbeddingRequests(
                maxGapSize: config.maxEmbeddingGap,
                maxEmbeddingLength: config.maxEmbeddingFrames
            )
            timeline.pendingEmbeddingRequests.append(contentsOf: requests)
        }
    }
    
    // MARK: - Speaker Clustering
    
    /// Compute average embedding for a segment
    public func averageEmbedding(for segment: EmbeddingSegment) -> [Float]? {
        let embeddings = segment.embeddings
        guard !embeddings.isEmpty else { return nil }
        
        let embeddingDim = embeddings[0].embedding.count
        var result = [Float](repeating: 0, count: embeddingDim)
        
        for emb in embeddings {
            vDSP_vadd(result, 1, emb.embedding, 1, &result, 1, vDSP_Length(embeddingDim))
        }
        
        var scale = 1.0 / Float(embeddings.count)
        vDSP_vsmul(result, 1, &scale, &result, 1, vDSP_Length(embeddingDim))
        
        // L2 normalize
        var norm: Float = 0
        vDSP_svesq(result, 1, &norm, vDSP_Length(embeddingDim))
        norm = sqrt(norm)
        
        if norm > 0 {
            vDSP_vsdiv(result, 1, &norm, &result, 1, vDSP_Length(embeddingDim))
        }
        
        return result
    }
    
    /// Compute cosine similarity between two embeddings
    public func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 0 }
        
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        
        var normA: Float = 0
        var normB: Float = 0
        vDSP_svesq(a, 1, &normA, vDSP_Length(a.count))
        vDSP_svesq(b, 1, &normB, vDSP_Length(b.count))
        
        let denom = sqrt(normA * normB)
        return denom > 0 ? dot / denom : 0
    }
}

// MARK: - Errors

public enum EmbeddingManagerError: Error, LocalizedError {
    case audioNotAvailable
    case extractorNotInitialized
    
    public var errorDescription: String? {
        switch self {
        case .audioNotAvailable:
            return "Audio not available in buffer for requested frame range"
        case .extractorNotInitialized:
            return "Embedding extractor not initialized. Call initialize() first."
        }
    }
}
