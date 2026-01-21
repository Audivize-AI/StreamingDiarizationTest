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
public struct TitaNetEmbedding: Identifiable, Hashable, SortformerFrameRange {
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
        return SortformerFrameRangeHelpers.isContiguous(self, other)
    }
    
    /// Check if this region overlaps another one
    public func overlaps<T>(with other: T) -> Bool
    where T : SortformerFrameRange {
        return frames.overlaps(other.frames)
    }
    
    /// Calculate the overlap length with a segment
    public func overlapLength<T>(with segment: T) -> Int
    where T: SortformerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, segment)
    }
    
    /// Calculate the overlap length with a segment
    public func framesOutside<T>(of segment: T) -> Int
    where T: SortformerFrameRange {
        return length - SortformerFrameRangeHelpers.overlapLength(self, segment)
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(startFrame)
        hasher.combine(endFrame)
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
        _embeddings.sort {
            SortformerFrameRangeHelpers.checkLessThan($0, $1)
        }
    }
    
    public func contains(_ frame: Int) -> Bool {
        frames.contains(frame)
    }
    
    public func isContiguous<T>(with other: T) -> Bool where T : SortformerFrameRange {
        SortformerFrameRangeHelpers.isContiguous(self, other)
    }
    
    public func isContiguous<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool where T : SpeakerFrameRange {
        SortformerFrameRangeHelpers.isContiguous(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    public func overlaps<T>(with other: T) -> Bool where T : SortformerFrameRange {
        frames.overlaps(other.frames)
    }
    
    public func overlaps<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool where T : SpeakerFrameRange {
        SortformerFrameRangeHelpers.overlaps(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    public func overlapLength<T>(with other: T) -> Int where T : SortformerFrameRange {
        SortformerFrameRangeHelpers.overlapLength(self, other)
    }
    
    public func overlapLength<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Int where T : SpeakerFrameRange {
        SortformerFrameRangeHelpers.overlapLength(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    /// Determine the required embeddings needed to close all gaps
    /// Uses a greedy interval covering algorithm to minimize the number of requests
    /// - Parameters:
    ///   - config: Embedding configuration
    ///   - availableRange: Optional range of frames available in the mel buffer. Requests are clipped to this range.
    public func getEmbeddingRequests(config: EmbeddingConfig, availableRange: Range<Int>? = nil) -> [EmbeddingExtractionRequest] {
        lock.lock()
        let currentEmbeddings = _embeddings
        lock.unlock()
        
        let minEmbeddingLength = config.minEmbeddingFrames
        let maxEmbeddingLength = config.maxEmbeddingFrames
        let maxGapSize = config.maxEmbeddingGap
        
        // Clip segment bounds to available range if provided
        let effectiveStart: Int
        let effectiveEnd: Int
        if let availableRange = availableRange {
            effectiveStart = max(startFrame, availableRange.lowerBound)
            effectiveEnd = min(endFrame, availableRange.upperBound)
        } else {
            effectiveStart = startFrame
            effectiveEnd = endFrame
        }
        
        let effectiveLength = effectiveEnd - effectiveStart
        
        // Skip if effective segment is too short for valid embeddings
        guard effectiveLength >= minEmbeddingLength else { return [] }
        
        // If no embeddings exist, create initial coverage for the effective range
        if currentEmbeddings.isEmpty {
            return generateInitialRequests(
                effectiveStart: effectiveStart,
                effectiveEnd: effectiveEnd,
                maxEmbeddingLength: maxEmbeddingLength,
                minEmbeddingLength: minEmbeddingLength
            )
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
                requests.append(EmbeddingExtractionRequest(startFrame: requestStart, endFrame: requestEnd))
                coveredUntil = requestEnd
            }
            
            gapIndex += 1
        }
        
        return requests
    }
    
    /// Generate initial embedding requests when no embeddings exist
    /// Uses non-overlapping windows to minimize redundancy
    private func generateInitialRequests(
        effectiveStart: Int,
        effectiveEnd: Int,
        maxEmbeddingLength: Int,
        minEmbeddingLength: Int
    ) -> [EmbeddingExtractionRequest] {
        var requests: [EmbeddingExtractionRequest] = []
        
        let effectiveLength = effectiveEnd - effectiveStart
        
        // Don't generate requests for segments that are too short
        guard effectiveLength >= minEmbeddingLength else { return [] }
        
        // If segment fits in one embedding
        if effectiveLength <= maxEmbeddingLength {
            requests.append(EmbeddingExtractionRequest(startFrame: effectiveStart, endFrame: effectiveEnd))
            return requests
        }
        
        // Use non-overlapping windows (stride = maxEmbeddingLength)
        // This provides 1x coverage for most, and ensures no request exceeds maxEmbeddingLength
        var currentStart = effectiveStart
        
        while currentStart < effectiveEnd {
            // Define the natural end of this window
            let requestedEnd = currentStart + maxEmbeddingLength
            
            // If this window would go past the end, clamp it
            let currentEnd = min(requestedEnd, effectiveEnd)
            
            // Only add if the chunk is at least minEmbeddingLength
            if currentEnd - currentStart >= minEmbeddingLength {
                requests.append(EmbeddingExtractionRequest(startFrame: currentStart, endFrame: currentEnd))
            }
            
            if currentEnd >= effectiveEnd {
                break
            }
            
            // Move to next window
            currentStart += maxEmbeddingLength
            
            // Check for potential tiny leftover at the end
            // If the remaining frames are fewer than minEmbeddingLength,
            // we create one final request that is anchored to the effectiveEnd
            // to ensure the tail gets a good quality embedding (by overlapping previous)
            if currentStart < effectiveEnd && (effectiveEnd - currentStart) < minEmbeddingLength {
                let backfilledStart = max(effectiveStart, effectiveEnd - maxEmbeddingLength)
                // Only add if backfilled chunk is valid
                if effectiveEnd - backfilledStart >= minEmbeddingLength {
                    requests.append(EmbeddingExtractionRequest(startFrame: backfilledStart, endFrame: effectiveEnd))
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
        
        guard !embeddings.isEmpty else { return }
        
        _embeddings.append(contentsOf: embeddings)
        _embeddings.sort {
            SortformerFrameRangeHelpers.checkLessThan($0, $1)
        }
    }
}

public struct EmbeddingExtractionRequest: SortformerFrameRange, Hashable {
    public let startFrame: Int
    public let endFrame: Int
    public var frames: Range<Int> { startFrame..<endFrame }
    public var length: Int { endFrame - startFrame }
    
    /// Create a request with explicit frame range
    public init(startFrame: Int, endFrame: Int) {
        self.startFrame = startFrame
        self.endFrame = endFrame
    }
    
    /// Create a request covering an entire segment's range
    public init<T>(for segment: T) where T: SortformerFrameRange {
        self.startFrame = segment.startFrame
        self.endFrame = segment.endFrame
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(startFrame)
        hasher.combine(endFrame)
    }
    
    public func contains(_ frame: Int) -> Bool {
        return frames.contains(frame)
    }
    
    public func isContiguous<T>(with other: T) -> Bool where T : SortformerFrameRange {
        return SortformerFrameRangeHelpers.isContiguous(self, other)
    }
    
    public func overlaps<T>(with other: T) -> Bool where T : SortformerFrameRange {
        return frames.overlaps(other.frames)
    }
    
    public func overlapLength<T>(with other: T) -> Int where T : SortformerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, other)
    }
}
