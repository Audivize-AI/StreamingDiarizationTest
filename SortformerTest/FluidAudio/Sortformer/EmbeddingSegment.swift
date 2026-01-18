//
//  SortformerSegmentEmbeddings.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/15/26.
//

import Foundation
import Accelerate
import CoreML




/// Represents a region where an embedding was extracted
public struct TitaNetEmbedding: Identifiable, SortformerFrameRange {
    /// Embedding ID
    public let id: UUID
    
    /// Start frame of the embedding region
    public var startFrame: Int { frames.lowerBound }
    
    /// End frame of the embedding region (non-inclusive)
    public var endFrame: Int { frames.upperBound }
    
    /// Frame range
    public let frames: Range<Int>
    
    /// The actual embedding vector
    public let embedding: [Float]
    
    /// Length in frames
    public var length: Int { frames.count }
    
    public init(embedding: [Float], startFrame: Int, endFrame: Int) {
        self.id = UUID()
        self.embedding = embedding
        self.frames = startFrame..<endFrame
    }
    
    /// Get the cosine distance to another embedding
    /// - Warning: Does not check if any of the vectors have a magnitude of 0.
    public func cosineDistance(to other: TitaNetEmbedding) -> Float {
        let a = self.embedding
        let b = other.embedding
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        let denom = sqrt(vDSP.sumOfSquares(a) * vDSP.sumOfSquares(b))
        return 1.0 - dot / denom
    }
    
    /// Check if this region contains a frame
    public func contains(_ frame: Int) -> Bool {
        return frames.contains(frame)
    }
    
    /// Check if this region is contiguous with another one
    public func isContiguous<T>(with other: T) -> Bool
    where T : SortformerFrameRange {
        return startFrame <= other.endFrame && endFrame >= other.startFrame
    }
    
    /// Check if this region overlaps another one
    public func overlaps<T>(with other: T) -> Bool
    where T : SortformerFrameRange {
        return frames.overlaps(other.frames)
    }
    
    /// Calculate the overlap length with a segment
    public func overlapLength<T>(with segment: T) -> Int
    where T: SortformerFrameRange {
        let overlapStart = max(startFrame, segment.startFrame)
        let overlapEnd = min(endFrame, segment.endFrame)
        return max(0, overlapEnd - overlapStart)
    }
    
    /// Calculate the proportion of this embedding that is outside the segment
    public func outsideProportion<T>(for segment: T) -> Float
    where T: SortformerFrameRange {
        let overlap = overlapLength(with: segment)
        return Float(length - overlap) / Float(length)
    }
    
    /// Calculate the coverage this embedding provides for a segment
    public func coverageRatio<T>(for segment: T) -> Float
    where T: SortformerFrameRange {
        let overlap = overlapLength(with: segment)
        return Float(overlap) / Float(segment.length)
    }
    
    /// Calculate the IoU with a given segment
    public func iou<T>(with segment: T) -> Float
    where T: SortformerFrameRange {
        let intersection = overlapLength(with: segment)
        let union = length + segment.length - intersection
        return Float(intersection) / Float(union)
    }
}


/// Tracks embeddings for a disjoint segment
public class EmbeddingSegment: SpeakerFrameRange, Identifiable {
    /// Segment ID
    public let id: UUID

    /// Speaker index in Sortformer output
    public var speakerIndex: Int

    /// Index of segment start frame
    public var startFrame: Int

    /// Index of segment end frame
    public var endFrame: Int

    /// Range of frames that this segment covers
    public var frames: Range<Int> { startFrame..<endFrame }

    /// Length of the segment in frames
    public var length: Int { endFrame - startFrame }
    
    /// Whether this segment is subject to updates
    public var isFinalized: Bool
    
    /// Lock for thread-safe access to mutable state
    private let lock = NSLock()
    
    /// Extracted embedding regions for this segment
    private var _embeddings: [TitaNetEmbedding]
    
    /// Thread-safe access to embeddings
    public var embeddings: [TitaNetEmbedding] {
        lock.lock()
        defer { lock.unlock() }
        return _embeddings
    }
    
    /// Pending embedding requests for this segment
    public private(set) var embeddingRequests: [EmbeddingExtractionRequest] = []
    
    /// Intra-cluster distances
    public private(set) var distances: [Float] = []
    
    public init(
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        finalized: Bool = true,
        embeddings: [TitaNetEmbedding] = []
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.isFinalized = finalized
        self._embeddings = embeddings
        self.sortEmbeddingsUnsafe()
    }
    
    public func contains(_ frame: Int) -> Bool {
        frames.contains(frame)
    }
    
    public func isContiguous<T>(with other: T) -> Bool where T : SortformerFrameRange {
        startFrame <= other.endFrame && endFrame >= other.startFrame
    }
    
    public func overlaps<T>(with other: T) -> Bool where T : SortformerFrameRange {
        frames.overlaps(other.frames)
    }
    
    /// Pop all embeddings that extend too far outside this segment (thread-safe)
    public func discardBadEmbeddings(maxOutsideFrames: Int) -> [TitaNetEmbedding] {
        lock.lock()
        defer { lock.unlock() }
        
        var discarded: [TitaNetEmbedding] = []
        let outer = startFrame-maxOutsideFrames..<endFrame+maxOutsideFrames
        for i in (0..<_embeddings.count).reversed() {
            if outer.contains(_embeddings[i].frames),
               frames.overlaps(_embeddings[i].frames) {
                continue
            }
            discarded.append(_embeddings.remove(at: i))
        }
        return discarded
    }
    
    /// Determine the required embeddings needed to close all gaps
    /// Uses a greedy interval covering algorithm to minimize the number of requests
    public func getEmbeddingRequests(maxGapSize: Int, maxEmbeddingLength: Int, minEmbeddingLength: Int = 12) -> [EmbeddingExtractionRequest] {
        lock.lock()
        let currentEmbeddings = _embeddings
        lock.unlock()
        
        // Skip if segment is too short for valid embeddings
        guard length >= minEmbeddingLength else { return [] }
        
        // If no embeddings exist, create initial coverage
        if currentEmbeddings.isEmpty {
            return generateInitialRequests(maxEmbeddingLength: maxEmbeddingLength, minEmbeddingLength: minEmbeddingLength)
        }
        
        // Find all uncovered regions (gaps)
        var gaps: [(start: Int, end: Int)] = []
        var currentPos = startFrame
        
        for embedding in currentEmbeddings {
            if embedding.startFrame > currentPos {
                // There's a gap before this embedding
                gaps.append((currentPos, embedding.startFrame))
            }
            currentPos = max(currentPos, embedding.endFrame)
        }
        
        // Check for gap at the end
        if currentPos < endFrame {
            gaps.append((currentPos, endFrame))
        }
        
        // Filter to only large gaps that need coverage
        let largeGaps = gaps.filter { $0.end - $0.start > maxGapSize }
        
        guard !largeGaps.isEmpty else {
            return []
        }
        
        // Greedy algorithm: cover gaps with minimum number of requests
        var requests: [EmbeddingExtractionRequest] = []
        var coveredUntil = startFrame
        var gapIndex = 0
        
        while gapIndex < largeGaps.count {
            let gap = largeGaps[gapIndex]
            
            // Skip gaps we've already covered
            if gap.end <= coveredUntil {
                gapIndex += 1
                continue
            }
            
            // Position the request to maximize coverage
            // Start from the gap start (or coveredUntil if we partially covered it)
            var requestStart = max(gap.start, coveredUntil)
            var requestEnd = min(requestStart + maxEmbeddingLength, endFrame)
            
            // Extend backwards if possible to cover more of the segment
            // But don't go before the segment start
            let backwardsSlack = min(requestStart - startFrame, maxEmbeddingLength - (requestEnd - requestStart))
            if backwardsSlack > 0 {
                requestStart -= backwardsSlack
            }
            
            // Try to extend forward to cover more gaps
            while gapIndex + 1 < largeGaps.count {
                let nextGap = largeGaps[gapIndex + 1]
                if nextGap.start < requestEnd {
                    // We can partially cover the next gap with this request
                    gapIndex += 1
                } else {
                    break
                }
            }
            
            // Clamp to segment bounds
            requestStart = max(requestStart, startFrame)
            requestEnd = min(requestEnd, endFrame)
            
            // Only add if request has meaningful length
            if requestEnd - requestStart > 0 {
                requests.append(EmbeddingExtractionRequest(
                    segment: self,
                    startFrame: requestStart,
                    endFrame: requestEnd
                ))
                coveredUntil = requestEnd
            }
            
            gapIndex += 1
        }
        
        return requests
    }
    
    /// Generate initial embedding requests when no embeddings exist
    /// Uses non-overlapping windows to minimize redundancy
    private func generateInitialRequests(maxEmbeddingLength: Int, minEmbeddingLength: Int) -> [EmbeddingExtractionRequest] {
        var requests: [EmbeddingExtractionRequest] = []
        
        // Don't generate requests for segments that are too short
        guard length >= minEmbeddingLength else { return [] }
        
        // If segment fits in one embedding
        if length <= maxEmbeddingLength {
            requests.append(EmbeddingExtractionRequest(
                segment: self,
                startFrame: startFrame,
                endFrame: endFrame
            ))
            return requests
        }
        
        // Use non-overlapping windows (stride = maxEmbeddingLength)
        // This provides 1x coverage for most, and ensures no request exceeds maxEmbeddingLength
        var currentStart = startFrame
        
        while currentStart < endFrame {
            // Define the natural end of this window
            let requestedEnd = currentStart + maxEmbeddingLength
            
            // If this window would go past the end, clamp it
            let currentEnd = min(requestedEnd, endFrame)
            
            // Only add if the chunk is at least minEmbeddingLength
            if currentEnd - currentStart >= minEmbeddingLength {
                requests.append(EmbeddingExtractionRequest(
                    segment: self,
                    startFrame: currentStart,
                    endFrame: currentEnd
                ))
            }
            
            if currentEnd >= endFrame {
                break
            }
            
            // Move to next window
            currentStart += maxEmbeddingLength
            
            // Check for potential tiny leftover at the end
            // If the remaining frames are fewer than minEmbeddingLength,
            // we create one final request that is anchored to the endFrame
            // to ensure the tail gets a good quality embedding (by overlapping previous)
            if currentStart < endFrame && (endFrame - currentStart) < minEmbeddingLength {
                let backfilledStart = max(startFrame, endFrame - maxEmbeddingLength)
                // Only add if backfilled chunk is valid
                if endFrame - backfilledStart >= minEmbeddingLength {
                    requests.append(EmbeddingExtractionRequest(
                        segment: self,
                        startFrame: backfilledStart,
                        endFrame: endFrame
                    ))
                }
                break
            }
        }
        
        return requests
    }
    
    /// Add embeddings to this segment (thread-safe)
    /// Automatically deduplicates by embedding ID
    public func addEmbeddings(_ embeddings: [TitaNetEmbedding]) {
        lock.lock()
        defer { lock.unlock() }
        
        // Deduplicate: only add embeddings we don't already have
        let existingIds = Set(_embeddings.map { $0.id })
        let newEmbeddings = embeddings.filter { !existingIds.contains($0.id) }
        
        guard !newEmbeddings.isEmpty else { return }
        
        _embeddings.append(contentsOf: newEmbeddings)
        sortEmbeddingsUnsafe()
    }
    
    /// Sort embeddings - call only when lock is already held
    @inline(__always)
    private func sortEmbeddingsUnsafe() {
        _embeddings.sort {
            $0.startFrame < $1.startFrame ||
            ($0.startFrame == $1.startFrame && $0.endFrame < $1.endFrame)
        }
    }
}

/// Request for a new embedding to be extracted
public struct EmbeddingExtractionRequest {
    /// The segment this embedding is for (weak reference since segment may be replaced during updates)
    public weak var segment: EmbeddingSegment?
    
    /// Speaker index for this request (used to find new segment if original was deallocated)
    public let speakerIndex: Int
    
    /// Frame range for the embedding
    public let startFrame: Int
    public let endFrame: Int
    
    public var length: Int { endFrame - startFrame }
    
    /// Whether this request is still valid (segment hasn't been deallocated)
    public var isValid: Bool { segment != nil }
    
    public init(segment: EmbeddingSegment, startFrame: Int, endFrame: Int) {
        self.segment = segment
        self.speakerIndex = segment.speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
    }
    
    /// Add the extracted embedding to the segment
    /// - Parameters:
    ///   - embedding: The extracted embedding vector
    ///   - fallbackSegment: Optional segment to use if original segment was deallocated
    /// - Returns: true if successful, false if no valid segment found
    @discardableResult
    public func addEmbedding(_ embedding: [Float], fallbackSegment: EmbeddingSegment? = nil) -> Bool {
        // Try original segment first, then fallback
        guard let targetSegment = segment ?? fallbackSegment else {
            return false
        }
        
        let titanetEmbedding = TitaNetEmbedding(
            embedding: embedding,
            startFrame: startFrame,
            endFrame: endFrame
        )
        targetSegment.addEmbeddings([titanetEmbedding])
        return true
    }
}


public extension SortformerTimeline {
    
    /// Process embedding extraction results from a batch extraction
    /// Uses direct segment references from requests - no lookup needed
    /// - Parameters:
    ///   - requests: The original extraction requests
    ///   - embeddings: The extracted embedding vectors (parallel to requests)
    func processExtractionResults(
        requests: [EmbeddingExtractionRequest],
        embeddings: [[Float]]
    ) {
        guard requests.count == embeddings.count else {
            SortformerTimeline.logger.warning("Request count (\(requests.count)) doesn't match embedding count (\(embeddings.count))")
            return
        }
        
        // Add embeddings directly to segments via the request's reference
        for (request, embedding) in zip(requests, embeddings) {
            request.addEmbedding(embedding)
        }
        
        // Remove completed requests
        let completedStarts = Set(requests.map { $0.startFrame })
        pendingEmbeddingRequests.removeAll { completedStarts.contains($0.startFrame) }
    }
    
    /// Update embedding segments when segment boundaries change
    /// This handles segment merging, splitting, and boundary adjustments
    func updateEmbeddingSegments(from oldSegments: [EmbeddingSegment]) {
        for segment in embeddingSegments {
            // Find matching old segment(s) by overlap
            let matchingOld = oldSegments.filter { old in
                segment.frames.overlaps(old.frames)
            }
            
            // Transfer embeddings from matching old segments
            for old in matchingOld {
                let validEmbeddings = old.embeddings.filter { embedding in
                    // Keep embeddings that don't extend too far outside the new segment
                    let outsideFrames = embedding.length - embedding.overlapLength(with: segment)
                    return outsideFrames <= embeddingConfig.maxOutsideFrames
                }
                segment.addEmbeddings(validEmbeddings)
            }
            
            // Discard embeddings that extend too far outside
            _ = segment.discardBadEmbeddings(maxOutsideFrames: embeddingConfig.maxOutsideFrames)
            
            // Generate new embedding requests if needed
            let requests = segment.getEmbeddingRequests(
                maxGapSize: embeddingConfig.maxEmbeddingGap,
                maxEmbeddingLength: embeddingConfig.maxEmbeddingFrames
            )
            pendingEmbeddingRequests.append(contentsOf: requests)
        }
    }
    
    /// Get all embeddings for a specific speaker
    func embeddings(forSpeaker speakerIndex: Int) -> [TitaNetEmbedding] {
        return embeddingSegments
            .filter { $0.speakerIndex == speakerIndex }
            .flatMap { $0.embeddings }
    }
    
    /// Get the embedding segment containing a specific frame, if any
    func embeddingSegment(containingFrame frame: Int) -> EmbeddingSegment? {
        return embeddingSegments.first { $0.contains(frame) } ??
               tentativeEmbeddingSegments.first { $0.contains(frame) }
    }
    
    /// Check if there are pending embedding extraction requests
    var hasPendingEmbeddingRequests: Bool {
        !pendingEmbeddingRequests.isEmpty
    }
    
    /// Get and clear all pending embedding requests
    func consumePendingEmbeddingRequests() -> [EmbeddingExtractionRequest] {
        let requests = pendingEmbeddingRequests
        pendingEmbeddingRequests.removeAll(keepingCapacity: true)
        return requests
    }
    
    /// Get embedding coverage statistics
    var embeddingCoverageStats: (totalFrames: Int, coveredFrames: Int, coverageRatio: Float) {
        var totalFrames = 0
        var coveredFrames = 0
        
        for segment in embeddingSegments {
            totalFrames += segment.length
            for embedding in segment.embeddings {
                coveredFrames += embedding.overlapLength(with: segment)
            }
        }
        
        let ratio: Float = totalFrames > 0 ? Float(coveredFrames) / Float(totalFrames) : 0
        return (totalFrames, min(coveredFrames, totalFrames), min(ratio, 1.0))
    }
}
