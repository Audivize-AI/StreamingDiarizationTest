//
//  TitaNetConfig.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/13/26.
//

import Foundation

public struct EmbeddingConfig {
    /// Minimum number of frames in a segment to receive an embedding
    public let minEmbeddingFrames: Int
    
    /// Minimum number of frames one embedding may belong to
    public let maxEmbeddingFrames: Int
    
    /// Maximum number of frames between embeddings within the same segment
    public let maxEmbeddingGap: Int

    /// Minimum number of frames between same-speaker segments to avoid merging
    public let minSegmentGap: Int

    /// Maximum number of frames outside a segment an embedding may cover
    public let maxOutsideFrames: Int
    
    // Non-configurable properties
    public let frameDuration: Float = 0.08
    public let subsamplingFactor: Int = 8
    public let melFeatures: Int = 80
    public let melStride: Int = 160
    public let melWindow: Int = 400
    public let melPadTo: Int = 16
    public static let embeddingFeatures: Int = 192

    // Computed properties
    var minMelLength: Int { minEmbeddingFrames * subsamplingFactor }
    var maxMelLength: Int { IndexUtils.nextMultiple(of: melPadTo, for: maxEmbeddingFrames * subsamplingFactor) }
    var melFeaturesPerFrame: Int { subsamplingFactor * melFeatures }
    var minInputSamples: Int { minMelLength * melStride }
    var maxInputSamples: Int { maxMelLength * melStride }
    var minInputDuration: Float { Float(minEmbeddingFrames) * frameDuration }
    var maxInputDuration: Float { Float(maxEmbeddingFrames) * frameDuration }
    
    public static let `default` = EmbeddingConfig()
    
    public init(
        minEmbeddingFrames: Int = 12,
        maxEmbeddingFrames: Int = 31,
        maxEmbeddingGap: Int = 6,
        minSegmentGap: Int = 4,
        maxOutsideFrames: Int = 2,
    ) {
        self.minEmbeddingFrames = minEmbeddingFrames
        self.maxEmbeddingFrames = maxEmbeddingFrames
        self.minSegmentGap = minSegmentGap
        self.maxOutsideFrames = maxOutsideFrames
        self.maxEmbeddingGap = maxEmbeddingGap
    }
}
