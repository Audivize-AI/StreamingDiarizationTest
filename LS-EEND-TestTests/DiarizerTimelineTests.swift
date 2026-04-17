//
//  DiarizerTimelineTests.swift
//  LS-EEND-TestTests
//
//  Regression tests for DiarizerTimeline segment/speaker detection logic.
//  Specifically guards against the "return-inside-for-loop" bug that aborted
//  per-slot scratch updates after slot 0.
//

import XCTest
@testable import LS_EEND_Test

final class DiarizerTimelineTests: XCTestCase {

    /// Drive a synthetic 4-speaker activation pattern through addPredictions
    /// and assert that segments are created for all 4 slots — not just slot 0.
    ///
    /// Before the fix in DiarizerTimeline.updateSegments, the `return` inside
    /// `for speakerIndex in 0..<speakerCapacity` exited the entire function
    /// after slot 0, so slots 1-3 never committed any segments.
    func testSegmentsCommitAcrossAllSlots() throws {
        let numSpeakers = 4
        let config = DiarizerTimelineConfig.default(
            numSpeakers: numSpeakers,
            frameDurationSeconds: 0.1
        )
        let timeline = DiarizerTimeline(config: config)

        // 40 frames total. Each speaker active for 10 frames sequentially.
        //   Frames  0- 9: slot 0 at 0.9, others at 0.1
        //   Frames 10-19: slot 1 at 0.9, others at 0.1
        //   Frames 20-29: slot 2 at 0.9, others at 0.1
        //   Frames 30-39: slot 3 at 0.9, others at 0.1
        let frames = 40
        var preds = [Float](repeating: 0.1, count: frames * numSpeakers)
        for f in 0..<frames {
            let activeSlot = f / 10
            preds[f * numSpeakers + activeSlot] = 0.9
        }

        try timeline.addPredictions(
            finalizedPredictions: preds,
            tentativePredictions: []
        )

        // All 4 slots should have created speaker entries + at least 1 segment.
        for slot in 0..<numSpeakers {
            guard let speaker = timeline.speakers[slot] else {
                XCTFail("slot \(slot) missing from timeline.speakers — "
                      + "updateSegments aborted early?")
                continue
            }
            XCTAssertFalse(
                speaker.finalizedSegments.isEmpty
                    && speaker.tentativeSegments.isEmpty,
                "slot \(slot) has no segments"
            )
        }
    }

    /// Guard the inverse: single-speaker audio must still work and must not
    /// leak activity into other slots.
    func testSingleSpeakerDoesNotActivateOthers() throws {
        let numSpeakers = 4
        let timeline = DiarizerTimeline(
            config: .default(numSpeakers: numSpeakers, frameDurationSeconds: 0.1)
        )
        let frames = 20
        var preds = [Float](repeating: 0.1, count: frames * numSpeakers)
        for f in 0..<frames {
            preds[f * numSpeakers + 0] = 0.9  // only slot 0
        }

        try timeline.addPredictions(
            finalizedPredictions: preds,
            tentativePredictions: []
        )

        XCTAssertNotNil(timeline.speakers[0])
        for slot in 1..<numSpeakers {
            if let sp = timeline.speakers[slot] {
                XCTAssertTrue(
                    sp.finalizedSegments.isEmpty && sp.tentativeSegments.isEmpty,
                    "slot \(slot) committed a segment from below-threshold activity"
                )
            }
        }
    }
}
