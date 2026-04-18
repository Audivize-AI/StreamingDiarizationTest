//
//  DiarizerViewModelTests.swift
//  LS-EEND-TestTests
//
//  State-transition tests for DiarizerViewModel that don't require loading
//  a real CoreML model. Constructed with `skipAutoLoad: true` so init()
//  doesn't fire an HF download or .mlpackage compile.
//

import XCTest
@testable import LS_EEND_Test

@MainActor
final class DiarizerViewModelTests: XCTestCase {

    func testInitialLoadStateIsIdle() {
        let vm = DiarizerViewModel(skipAutoLoad: true)
        XCTAssertEqual(vm.loadState, .idle)
        XCTAssertFalse(vm.isProcessing)
        XCTAssertEqual(vm.progress, 0)
        XCTAssertTrue(vm.statusMessage.isEmpty)
        XCTAssertTrue(vm.segments.isEmpty)
        XCTAssertTrue(vm.finalizedPredictions.isEmpty)
    }

    func testResetClearsPopulatedSnapshotEvenWithoutModel() {
        let vm = DiarizerViewModel(skipAutoLoad: true)
        vm.finalizedPredictions = [0.1, 0.2, 0.3]
        vm.tentativePredictions = [0.9]
        vm.finalizedFrameCount = 3
        vm.tentativeFrameCount = 1
        vm.segments = []

        vm.reset()

        XCTAssertTrue(vm.finalizedPredictions.isEmpty)
        XCTAssertTrue(vm.tentativePredictions.isEmpty)
        XCTAssertEqual(vm.finalizedFrameCount, 0)
        XCTAssertEqual(vm.tentativeFrameCount, 0)
    }

    func testRenameSpeakerWithoutModelIsNoOp() {
        let vm = DiarizerViewModel(skipAutoLoad: true)
        // Must not crash when diarizer is nil.
        vm.renameSpeaker(slot: 0, to: "Alice")
        XCTAssertEqual(vm.segments.count, 0)
    }

    func testProcessFileSetsStatusWhenModelNotReady() async {
        let vm = DiarizerViewModel(skipAutoLoad: true)
        let bogusURL = URL(fileURLWithPath: "/nonexistent.wav")
        await vm.processFile(bogusURL)
        XCTAssertEqual(vm.statusMessage, "Model not ready yet.")
        XCTAssertFalse(vm.isProcessing)
    }

    func testStartMicrophoneThrowsWhenModelNotReady() async {
        let vm = DiarizerViewModel(skipAutoLoad: true)
        do {
            try await vm.startMicrophone()
            XCTFail("expected startMicrophone to throw when diarizer is nil")
        } catch {
            // Expected — surface the message for diagnostic value.
            let nsErr = error as NSError
            XCTAssertEqual(nsErr.domain, "LSEEND.Mic")
            XCTAssertEqual(nsErr.code, 1)
        }
    }

    /// Exercises the AsyncStream → MainActor apply path. Both event kinds
    /// drive through the same consumer Task; polling lets the consumer
    /// drain without assuming a fixed number of yields.
    func testProgressEventIsAppliedOnMainActor() async {
        let vm = DiarizerViewModel(skipAutoLoad: true)
        vm._testInjectProgress(0.42)
        await waitUntil(timeout: 1.0) { vm.progress == 0.42 }
        XCTAssertEqual(vm.progress, 0.42, accuracy: 1e-9)
    }

    func testStatusEventIsAppliedOnMainActor() async {
        let vm = DiarizerViewModel(skipAutoLoad: true)
        vm._testInjectStatus("hello")
        await waitUntil(timeout: 1.0) { vm.statusMessage == "hello" }
        XCTAssertEqual(vm.statusMessage, "hello")
    }

    /// Poll `predicate` on the main actor, yielding between checks, until
    /// it's true or `timeout` (seconds) elapses.
    private func waitUntil(timeout: Double, _ predicate: () -> Bool) async {
        let deadline = Date().addingTimeInterval(timeout)
        while !predicate() && Date() < deadline {
            try? await Task.sleep(nanoseconds: 10_000_000) // 10ms
        }
    }
}
