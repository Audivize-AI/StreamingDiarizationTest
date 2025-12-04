//
//  Config.swift
//  DiarizationTest
//
//  Created by Benjamin Lee on 12/4/25.
//

import Foundation


public struct Config {
    public static let chunkSkip: TimeInterval = 1.0 / 3.0
    public static let chunkDuration: TimeInterval = 10.0
    public static let boundaryTolerance: TimeInterval = chunkSkip / 2.0
}
