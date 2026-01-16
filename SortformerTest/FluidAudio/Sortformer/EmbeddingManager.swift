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
    
    /// Currently processing requests (to avoid duplicates)
    private var processingRequests: Set<ObjectIdentifier> = []
    
    /// Logger
    private static let logger = AppLogger(category: "EmbeddingManager")
    
    // MARK: - Initialization
    
    public init(
        config: EmbeddingConfig = EmbeddingConfig(),
        titanetConfig: TitaNetConfig = TitaNetConfig(),
        frameDurationSeconds: Float = 0.04
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
        
        extractionQueue.async { [weak self] in
            guard let self = self else { return }
            
            var successCount = 0
            
            for request in requests {
                // Check if already processing
                let requestId = ObjectIdentifier(request.segment)
                
                self.lock.lock()
                let isProcessing = self.processingRequests.contains(requestId)
                if !isProcessing {
                    self.processingRequests.insert(requestId)
                }
                self.lock.unlock()
                
                if isProcessing {
                    continue
                }
                
                defer {
                    self.lock.lock()
                    self.processingRequests.remove(requestId)
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
                    request.addEmbedding(embedding)
                    successCount += 1
                } catch {
                    Self.logger.error("Embedding extraction failed: \(error.localizedDescription)")
                }
            }
            
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
