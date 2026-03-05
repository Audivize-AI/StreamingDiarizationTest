//
//  ClusteringConfig.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/18/26.
//

import Foundation

public struct ClusteringConfig {    
    /// Intra-cluster separation threshold
    var clusteringThreshold: Float
    
    /// Maximum cosine distance to update a cluster
    var updateThreshold: Float
    
    /// Chamfer distance threshold to match with another speaker profile
    var matchThreshold: Float
    
    /// Maximum number of speakers supported by the EEND model
    let numSlots: Int
    
    init(
        clusteringThreshold: Float = 0.25,
        updateThreshold: Float = 0.2,
        matchThreshold: Float = 0.2,
        numSlots: Int = 4,
    ) {
        self.clusteringThreshold = clusteringThreshold
        self.updateThreshold = updateThreshold
        self.matchThreshold = matchThreshold
        self.numSlots = numSlots
    }
    
    init(from config: SortformerTimelineConfig) {
        self.clusteringThreshold = config.clusteringThreshold
        self.updateThreshold = config.updateThreshold
        self.matchThreshold = config.matchThreshold
        self.numSlots = config.numSpeakers
    }
}
