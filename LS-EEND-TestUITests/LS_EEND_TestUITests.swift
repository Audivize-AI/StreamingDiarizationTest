//
//  LS_EEND_TestUITests.swift
//  LS-EEND-TestUITests
//

import XCTest

/// Minimal UI-test skeleton. Full accessibility-driven tests for the
/// segmented Mode picker, microphone button, and model picker are blocked
/// on the macOS XCTest runner returning "Application has not loaded
/// accessibility" for this app. Once the runner attaches reliably, these
/// should grow back to exercise the real UI flows — file picker, mic
/// start/stop, segment list rename — ideally keyed off explicit
/// `accessibilityIdentifier` values added to ContentView.
final class LS_EEND_TestUITests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    @MainActor
    func testAppLaunchesWithoutCrashing() throws {
        let app = XCUIApplication()
        app.launch()
        // Passing condition: launch returned without throwing.
        XCTAssertTrue(app.exists)
    }

    @MainActor
    func testLaunchPerformance() throws {
        measure(metrics: [XCTApplicationLaunchMetric()]) {
            XCUIApplication().launch()
        }
    }
}
