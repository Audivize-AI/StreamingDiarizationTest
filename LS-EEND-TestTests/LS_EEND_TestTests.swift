//
//  LS_EEND_TestTests.swift
//  LS-EEND-TestTests
//
//  Created by Benjamin Lee on 4/16/26.
//

import XCTest
@testable import LS_EEND_Test
import CoreML

final class LS_EEND_TestTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testExample() async throws {
        do {
            let model = try await LSEENDModel.loadFromHuggingFace(variant: .dihard3, stepSize: .step300ms)
            print(model.metadata)
        } catch {
            XCTFail("Failed to load model from hf: \(error)")
        }
    }

    func testPerformanceExample() throws {
        // This is an example of a performance test case.
        self.measure {
            // Put the code you want to measure the time of here.
        }
    }

}
