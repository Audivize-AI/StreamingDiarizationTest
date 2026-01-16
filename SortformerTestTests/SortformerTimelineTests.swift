//
//  SortformerTimelineTests.swift
//  SortformerTestTests
//
//  Created on 1/16/26.
//

import XCTest
@testable import SortformerTest

final class SortformerTimelineTests: XCTestCase {
    
    // MARK: - Test Helpers
    
    /// Create a simple chunk result for testing
    private func makeChunk(
        predictions: [[Float]],  // [numFrames][numSpeakers]
        tentative: [[Float]] = [],
        startFrame: Int = 0,
        fifoLength: Int = 0
    ) -> SortformerChunkResult {
        let numSpeakers = predictions.first?.count ?? 4
        let flattened = predictions.flatMap { $0 }
        let tentativeFlat = tentative.flatMap { $0 }
        
        // Create empty FIFO predictions if not specified
        let fifoPreds = fifoLength > 0 ? [Float](repeating: 0.5, count: fifoLength * numSpeakers) : []
        
        return SortformerChunkResult(
            startFrame: startFrame,
            speakerPredictions: flattened,
            frameCount: predictions.count,
            tentativePredictions: tentativeFlat,
            tentativeFrameCount: tentative.count,
            fifoPredictions: fifoPreds,
            fifoFrameCount: fifoLength
        )
    }
    
    /// Create a config with simple thresholds
    private func makeConfig() -> SortformerPostProcessingConfig {
        return SortformerPostProcessingConfig(
            onsetThreshold: 0.5,
            offsetThreshold: 0.5,
            onsetPadFrames: 0,
            offsetPadFrames: 0,
            minFramesOn: 1,
            minFramesOff: 0,
            numFilteredFrames: 0
        )
    }
    
    // MARK: - Basic Tests
    
    func testEmptyTimeline() {
        let config = makeConfig()
        let timeline = SortformerTimeline(config: config)
        
        XCTAssertEqual(timeline.nextFrame, 0)
        XCTAssertTrue(timeline.segments.flatMap { $0 }.isEmpty)
        XCTAssertTrue(timeline.embeddingSegments.isEmpty)
    }
    
    func testSingleSpeakerSegment() {
        let config = makeConfig()
        let timeline = SortformerTimeline(config: config)
        
        // Speaker 0 speaking for 10 frames (4 speakers total as default)
        var predictions: [[Float]] = []
        for _ in 0..<10 {
            predictions.append([0.8, 0.2, 0.1, 0.1])  // Speaker 0 on
        }
        
        let chunk = makeChunk(predictions: predictions)
        _ = timeline.addChunk(chunk)
        
        XCTAssertEqual(timeline.nextFrame, 10)
        
        // Should have one segment for speaker 0
        let speaker0Segs = timeline.segments[0]
        XCTAssertEqual(speaker0Segs.count, 1)
        XCTAssertEqual(speaker0Segs[0].startFrame, 0)
        XCTAssertEqual(speaker0Segs[0].endFrame, 10)
    }
    
    // MARK: - Disjoint Segment Tests
    
    func testDisjointSegmentsNoOverlap() {
        let config = makeConfig()
        let timeline = SortformerTimeline(config: config)
        
        // Speaker 0: frames 0-5, Speaker 1: frames 5-10 (no overlap)
        var predictions: [[Float]] = []
        for i in 0..<10 {
            if i < 5 {
                predictions.append([0.8, 0.2, 0.1, 0.1])  // Speaker 0
            } else {
                predictions.append([0.2, 0.8, 0.1, 0.1])  // Speaker 1
            }
        }
        
        let chunk = makeChunk(predictions: predictions)
        _ = timeline.addChunk(chunk)
        
        // Should have embedding segments for both single-speaker regions
        XCTAssertGreaterThanOrEqual(timeline.embeddingSegments.count, 2)
        
        // Check no duplicates (segments should not overlap for same speaker)
        for i in 0..<timeline.embeddingSegments.count {
            for j in (i+1)..<timeline.embeddingSegments.count {
                let seg1 = timeline.embeddingSegments[i]
                let seg2 = timeline.embeddingSegments[j]
                if seg1.speakerIndex == seg2.speakerIndex {
                    XCTAssertFalse(
                        seg1.frames.overlaps(seg2.frames),
                        "Overlapping embedding segments: \(seg1.startFrame)-\(seg1.endFrame) and \(seg2.startFrame)-\(seg2.endFrame)"
                    )
                }
            }
        }
    }
    
    func testAdjacentSegmentsSameSpeaker() {
        let config = makeConfig()
        let timeline = SortformerTimeline(config: config)
        
        // Speaker 0 speaking continuously
        var predictions: [[Float]] = []
        for _ in 0..<10 {
            predictions.append([0.8, 0.2, 0.1, 0.1])  // Speaker 0 continuous
        }
        
        let chunk = makeChunk(predictions: predictions)
        _ = timeline.addChunk(chunk)
        
        // Should have one continuous segment
        let speaker0Segs = timeline.segments[0]
        XCTAssertEqual(speaker0Segs.count, 1)
    }
    
    // MARK: - Streaming Update Tests
    
    func testStreamingChunksPreserveOrder() {
        let config = makeConfig()
        let timeline = SortformerTimeline(config: config)
        
        // Add first chunk - speaker 0
        let chunk1 = makeChunk(predictions: Array(repeating: [Float(0.8), 0.2, 0.1, 0.1], count: 10))
        _ = timeline.addChunk(chunk1)
        
        // Add second chunk - speaker 1
        let chunk2 = makeChunk(
            predictions: Array(repeating: [Float(0.2), 0.8, 0.1, 0.1], count: 10),
            startFrame: 10
        )
        _ = timeline.addChunk(chunk2)
        
        XCTAssertEqual(timeline.nextFrame, 20)
        
        // Verify segments are in chronological order
        for speakerSegs in timeline.segments {
            for i in 1..<speakerSegs.count {
                XCTAssertLessThanOrEqual(speakerSegs[i-1].endFrame, speakerSegs[i].startFrame,
                    "Segments not in chronological order")
            }
        }
        
        // Verify embedding segments are in order
        for i in 1..<timeline.embeddingSegments.count {
            XCTAssertLessThanOrEqual(timeline.embeddingSegments[i-1].startFrame, 
                                      timeline.embeddingSegments[i].startFrame,
                "Embedding segments not in chronological order")
        }
    }
    
    func testEmbeddingSegmentsNoDuplicates() {
        let config = makeConfig()
        let timeline = SortformerTimeline(config: config)
        
        // Add multiple chunks
        for i in 0..<5 {
            let chunk = makeChunk(
                predictions: Array(repeating: [Float(0.8), 0.2, 0.1, 0.1], count: 10),
                startFrame: i * 10
            )
            _ = timeline.addChunk(chunk)
        }
        
        // Check for duplicates
        let segmentIDs = timeline.embeddingSegments.map { $0.id }
        let uniqueIDs = Set(segmentIDs)
        XCTAssertEqual(segmentIDs.count, uniqueIDs.count, "Duplicate embedding segments detected")
        
        // Check for overlapping segments with same speaker
        for i in 0..<timeline.embeddingSegments.count {
            for j in (i+1)..<timeline.embeddingSegments.count {
                let seg1 = timeline.embeddingSegments[i]
                let seg2 = timeline.embeddingSegments[j]
                if seg1.speakerIndex == seg2.speakerIndex {
                    XCTAssertFalse(seg1.frames.overlaps(seg2.frames),
                        "Overlapping embedding segments for same speaker: [\(seg1.startFrame)-\(seg1.endFrame)] and [\(seg2.startFrame)-\(seg2.endFrame)]")
                }
            }
        }
    }
    
    // MARK: - FIFO Filter Update Tests
    
    func testFifoFilterDoesNotDeleteSegments() {
        let config = SortformerPostProcessingConfig(
            onsetThreshold: 0.5,
            offsetThreshold: 0.5,
            onsetPadFrames: 0,
            offsetPadFrames: 0,
            minFramesOn: 1,
            minFramesOff: 0,
            numFilteredFrames: 10  // Enable filtering
        )
        let timeline = SortformerTimeline(config: config)
        
        // Add initial chunk
        let chunk1 = makeChunk(
            predictions: Array(repeating: [Float(0.8), 0.2, 0.1, 0.1], count: 20)
        )
        _ = timeline.addChunk(chunk1)
        
        let segmentCountBefore = timeline.embeddingSegments.count
        
        // Add second chunk with FIFO that slightly changes predictions
        let chunk2 = makeChunk(
            predictions: Array(repeating: [Float(0.8), 0.2, 0.1, 0.1], count: 10),
            startFrame: 20,
            fifoLength: 10
        )
        _ = timeline.addChunk(chunk2)
        
        // Segment count should not decrease (existing segments preserved)
        XCTAssertGreaterThanOrEqual(timeline.embeddingSegments.count, segmentCountBefore,
            "Embedding segments were deleted during FIFO update")
    }
    
    // MARK: - Boundary Condition Tests
    
    func testExactBoundaryMatch() {
        // Test that segments with exact same boundaries are correctly matched
        let config = makeConfig()
        let timeline = SortformerTimeline(config: config)
        
        // Create a segment with speaker 0
        let predictions: [[Float]] = Array(repeating: [0.8, 0.2, 0.1, 0.1], count: 30)
        let chunk = makeChunk(predictions: predictions)
        _ = timeline.addChunk(chunk)
        
        let initialCount = timeline.embeddingSegments.count
        
        // Add more with same pattern - existing segments should be matched
        let chunk2 = makeChunk(
            predictions: Array(repeating: [Float(0.8), 0.2, 0.1, 0.1], count: 10),
            startFrame: 30
        )
        _ = timeline.addChunk(chunk2)
        
        // Should not have excessive growth
        XCTAssertLessThan(timeline.embeddingSegments.count, initialCount * 2,
            "Too many embedding segments - likely duplicates")
    }
    
    // MARK: - Off-by-one Tests
    
    func testSegmentBoundariesAreExact() {
        let config = makeConfig()
        let timeline = SortformerTimeline(config: config)
        
        // Speaker 0 on for frames 0-4, off for 5-9
        var predictions: [[Float]] = []
        for i in 0..<10 {
            if i < 5 {
                predictions.append([0.8, 0.2, 0.1, 0.1])
            } else {
                predictions.append([0.2, 0.2, 0.1, 0.1])  // All off
            }
        }
        
        let chunk = makeChunk(predictions: predictions)
        _ = timeline.addChunk(chunk)
        
        // Speaker 0 should have segment ending at frame 5 (exclusive)
        if let seg = timeline.segments[0].first {
            XCTAssertEqual(seg.startFrame, 0, "Segment start frame incorrect")
            XCTAssertEqual(seg.endFrame, 5, "Segment end frame incorrect")
        } else {
            XCTFail("Expected segment for speaker 0")
        }
    }
}
