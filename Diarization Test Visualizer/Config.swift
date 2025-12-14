//
//  Config.swift
//  DiarizationTest
//
//  Created by Benjamin Lee on 12/4/25.
//

import Foundation


public struct Config {
    public static let chunkSkip: TimeInterval = 0.333
    public static let chunkDuration: TimeInterval = 10.0
    public static let boundaryTolerance: TimeInterval = 1.0 / 6.0
    public static let iouMatchingThreshold: Double = 0.80
    public static let maximumThreshold: Double = 0.80
    public static let embeddingThreshold: Float = 0.70
    public static let embeddingDistanceAlpha: Float = 0.25
    
}
