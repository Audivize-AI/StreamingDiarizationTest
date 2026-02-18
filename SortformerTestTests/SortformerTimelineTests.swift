//
//  SortformerTimelineTests.swift
//  SortformerTestTests
//
//  Created on 1/16/26.
//

import XCTest
@testable import SortformerTest

// TODO: These tests need to be updated after the major refactor
// The old SortformerChunkResult was renamed to SortformerStateUpdateResult
// and the initialization API has changed significantly.

final class SortformerTimelineTests: XCTestCase {
    
    func testPlaceholder() {
        // Placeholder test - needs major refactor
        XCTAssertTrue(true)
    }
    
    
    // Scenario: There is a finalized segment for speaker 0 from 0..<T-1 and a tentative segment for speaker 0 from T..<(T+12)
    // where T is the first tentative frame for Sortformer segments.
    func testEmbeddingSegmentsWhenSpeakerSegmentFinalized() {
        
        
    }
}
