//
//  LS_EENDTestApp.swift
//  LS-EENDTest
//
//  Created by Benjamin Lee on 3/13/26.
//

import SwiftUI

private extension ProcessInfo {
    var isRunningXCTest: Bool {
        environment["XCTestConfigurationFilePath"] != nil
    }
}

@main
struct LS_EENDTestApp: App {
    init() {
        LSEENDRuntimeProbeSupport.runIfRequested()
    }

    var body: some Scene {
        WindowGroup {
            if ProcessInfo.processInfo.isRunningXCTest {
                Text("LS-EENDTest XCTest Host")
                    .frame(minWidth: 320, minHeight: 200)
            } else {
                ContentView()
            }
        }
    }
}
