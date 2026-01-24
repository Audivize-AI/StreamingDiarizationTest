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
public class SpeakerEmbedding: Identifiable, Hashable, Comparable, SortformerFrameRange {
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
    public func cosineDistance(to other: SpeakerEmbedding) -> Float {
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
    
    public static func < (lhs: SpeakerEmbedding, rhs: SpeakerEmbedding) -> Bool {
        (lhs.startFrame, lhs.endFrame) < (rhs.startFrame, rhs.endFrame)
    }
    
    public static func == (lhs: SpeakerEmbedding, rhs: SpeakerEmbedding) -> Bool {
        lhs.id == rhs.id
    }
}


/// Tracks embeddings for a disjoint segment
public struct EmbeddingSegment: SpeakerFrameRange, Identifiable {
    /// Segment ID
    public let id: UUID

    /// Speaker index in Sortformer output
    public let speakerIndex: Int

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
    
    /// Extracted embedding regions for this segment
    public private(set) var embeddings: [SpeakerEmbedding]
    
    /// Intra-cluster distances
    public private(set) var distances: [Float] = []
    
    public var isNone: Bool { speakerIndex < 0 }
    public var isValid: Bool { speakerIndex >= 0 }
    
    public static let none: EmbeddingSegment = .init(speakerIndex: -1, startFrame: 0, endFrame: 0)
    
    // MARK: - Init
    public init(
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        finalized: Bool = true
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.isFinalized = finalized
        self.embeddings = []
    }
    
    // MARK: - Speaker Frame Range
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
    
    // MARK: - Embedding extraction
    
    /// Determine the required embeddings needed to close all gaps
    /// Uses a greedy interval covering algorithm to minimize the number of requests
    /// - Parameters:
    ///   - extractor: Embedding Manager
    ///   - streamingHorizonFrame: If the segment ends before this frame, then its end frame is probably finalized.
    public mutating func initializeEmbeddings(
        with extractor: EmbeddingManager,
        streamingHorizonFrame: Int
    ) throws {
        embeddings = extractor.takeMatches(for: self)
        
        let isComplete = endFrame < streamingHorizonFrame
        let requests: [EmbeddingRequest]
        let minEmbeddingLength = extractor.config.minEmbeddingFrames
        let maxEmbeddingLength = extractor.config.maxEmbeddingFrames
        let maxGapLength = extractor.config.maxEmbeddingGap
        
        if embeddings.isEmpty {
            requests = generateInitialRequests(
                minEmbeddingLength: minEmbeddingLength,
                maxEmbeddingLength: maxEmbeddingLength,
                maxGapLength: maxGapLength,
                isComplete: isComplete
            )
        } else {
            let embeddingFrames = embeddings.map(\.frames).sorted {
                ($0.lowerBound, $0.upperBound) < ($1.lowerBound, $1.upperBound)
            }
            var cutoff = endFrame
            if !isComplete {
                cutoff -= maxEmbeddingLength
            }
            
            let gaps = getGaps(
                in: embeddingFrames,
                before: cutoff,
                ifAnyExceed: extractor.config.maxEmbeddingGap
            )
            
            guard let lastEmbeddingEnd = embeddingFrames.last?.upperBound else {
                preconditionFailure("Last embedding has no upper bound")
            }
            
            let requestStarts = generateCovering(
                for: gaps,
                biggerThan: extractor.config.maxEmbeddingGap,
                withCoverLength: extractor.config.maxEmbeddingFrames,
                lastEmbeddingEnd: lastEmbeddingEnd,
            )
            
            requests = requestStarts.map { start in
                EmbeddingRequest(
                    startFrame: max(startFrame, start),
                    endFrame: start + extractor.config.maxEmbeddingFrames
                )
            }
        }
        
        let newEmbeddings = try extractor.processRequests(requests)
        embeddings.append(contentsOf: newEmbeddings)
    }
    
    /// - Parameters:
    ///   - gaps: Disjoint gap intervals sorted by start frame in decending order
    ///   - maxGapSize: Maximum gap between embeddings
    ///   - coverLength: Number of frames that one embedding can cover
    ///   - lastEmbeddingEnd: The end frame of the last embedding
    /// - Returns: List of cover start indices
    /// - Precondition: `gaps` is disjoint and sorted in decending order by start/end frame
    /// - Precondition: The union of all intervals in `gaps` is a subset of `self.startFrame ..< self.endFrame`
    /// - Precondition: `lastEmbeddingEnd <= self.endFrame`
    /// - Precondition: If `lastEmbeddingEnd < self.endFrame`, then `gaps[0] = lastEmbeddingEnd ..< self.endFrame`
    /// - Precondition: `coverLength <= self.length`
    private func generateCovering(
        for gaps: [(start: Int, end: Int)],
        biggerThan maxGapSize: Int,
        withCoverLength coverLength: Int,
        lastEmbeddingEnd: Int,
    ) -> [Int] {
        guard gaps.count > 0 else { return [] }
        
        var covers: [(start: Int, gapStart: Int)] = []
        
        // It's covered starting from this frame
        var coveredFrom = gaps[0].end
        
        // Backward pass to generate coverings
        for (start, end) in gaps {
            // Check if the gap overlaps with the last cover
            if let previousCover = covers.last, previousCover.start < end {
                covers[covers.count-1].gapStart = start
            }
            
            // There is an embedding that starts at this gap's end, so extend the range of covered frames
            coveredFrom = min(coveredFrom, end)
            
            // Tile the entire gap
            while coveredFrom - maxGapSize > start {
                coveredFrom = coveredFrom - maxGapSize - coverLength
                covers.append((start: coveredFrom, gapStart: start))
            }
        }
        
        guard covers.count > 0 else {
            return []
        }
        
        // Forward pass to optimize coverage
        
        var previousEnd: Int = .min
        var nextUncovered: Int = 0
        
        for i in (0..<covers.count).reversed() {
            // Shift the segment as far forward as possible so it's not touching something in front
            nextUncovered = max(previousEnd, covers[i].gapStart)
            
            if nextUncovered > covers[i].start {
                covers[i].start = nextUncovered
            }
            previousEnd = covers[i].start + coverLength
        }
        
        // Partial backwards pass to make the last cover align with the end frame if possible.
        // Coverings are in reverse order, so the last cover is at index 0
        var lastCover: (start: Int, gapStart: Int) {
            get { covers[0] }
            set { covers[0] = newValue }
        }
        
        let lastCoverEnd = lastCover.start + coverLength
        if lastCoverEnd > lastEmbeddingEnd {
            var shift = endFrame - lastCoverEnd
            if shift > 0 {
                let currentGap = lastCover.start - nextUncovered
                let maxShift = maxGapSize - currentGap
                shift = min(shift, maxShift)
            }
            lastCover.start += shift
            
            // Ensure that the previous segments don't pass the end frame
            if shift < 0 && covers.count > 1 {
                var nextStart = covers[0].start
                
                for i in 1..<covers.count {
                    let overlap = covers[i].start + coverLength - nextStart
                    guard overlap > 0 else { break }
                    nextStart = covers[i].start - overlap
                    covers[i].start = max(startFrame, nextStart)
                    guard nextStart > startFrame else { break }
                }
            }
        }
        
        return covers.reversed().map(\.start)
    }
    
    /// Get embedding gaps in reversed order
    /// - Parameters:
    ///   - embeddingRanges: Covered embedding ranges sorted in ascending order
    ///   - cutoffFrame: Gaps will be ignored begining at this frame
    ///   - maxGapSize: Maximum allowed gap size. If no gaps exceed this threshold, then nothing is returned.
    /// - Returns: Frame intervals `[start, end)` that not covered by the embeddings
    private func getGaps(
        in embeddingRanges: [Range<Int>],
        before cutoffFrame: Int,
        ifAnyExceed maxGapSize: Int
    ) -> [(start: Int, end: Int)] {
        var gaps: [(start: Int, end: Int)] = []
        var end = cutoffFrame
        var hasLargeGap = false
        
        for range in embeddingRanges.reversed() {
            if range.upperBound < end {
                // There's a gap after this embedding
                gaps.append((range.upperBound, end))
                
                if end - range.upperBound > maxGapSize {
                    hasLargeGap = true
                }
            }
            // Next gap may end at the start of this range
            end = min(end, range.lowerBound)
        }
        
        // Check for gap at the start
        if (hasLargeGap && end > startFrame) || (end - startFrame > maxGapSize) {
            gaps.append((startFrame, end))
            hasLargeGap = true
        }
        
        guard hasLargeGap else { return [] }
        return gaps
    }
    
    /// Generate initial embedding requests when no embeddings exist
    /// Uses non-overlapping windows to minimize redundancy
    private func generateInitialRequests(
        minEmbeddingLength: Int,
        maxEmbeddingLength: Int,
        maxGapLength: Int,
        isComplete: Bool
    ) -> [EmbeddingRequest] {
        // Don't generate requests for segments that are too short
        guard length >= minEmbeddingLength else { return [] }
        
        // If segment fits in one embedding
        if length <= maxEmbeddingLength {
            return [EmbeddingRequest(for: self)]
        }
        
        var requests: [EmbeddingRequest] = []
        var currentStart = startFrame

        // Since start = endFrame - 1 -> gap = 0, gap = endFrame - start - 1
        // So, endFrame - start - 1 ≤ maxGap, or start < endFrame - maxGap
        let maxEndGap = isComplete ? maxGapLength : maxEmbeddingLength - 1
        let firstOptionalFrame = endFrame - maxEndGap
        
        while currentStart < firstOptionalFrame {
            var currentEnd = currentStart + maxEmbeddingLength
            let overflow = currentEnd - endFrame
            
            if overflow > 0 {
                currentStart = max(startFrame, currentStart - overflow)
                currentEnd = endFrame
            }
            
            let currentLength = currentEnd - currentStart
            
            // Only add if the chunk is at least minEmbeddingLength
            // This guard should always pass
            guard currentLength >= minEmbeddingLength else {
                print("WARNING: Found a segment with an embedding length too short (\(currentLength) < \(minEmbeddingLength)). Terminating.")
                return requests
            }
            
            requests.append(EmbeddingRequest(
                startFrame: currentStart,
                endFrame: currentEnd
            ))
            
            currentStart = currentEnd + maxGapLength
        }
        
        return requests
    }
}

public struct EmbeddingRequest: SortformerFrameRange, Hashable {
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
