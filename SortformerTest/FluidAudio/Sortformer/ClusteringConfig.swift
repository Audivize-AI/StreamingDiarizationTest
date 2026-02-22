//
//  ClusteringConfig.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/18/26.
//

import Foundation

struct ClusteringConfig {    
    /// Intra-cluster separation threshold
    var clusteringThreshold: Float
    
    /// Maximum number of speakers supported by the EEND model
    let eendSpeakerCapacity: Int
    
    init(
        clusteringThreshold: Float = 0.25,
        eendSpeakerCapacity: Int = 4,
    ) {
        self.clusteringThreshold = clusteringThreshold
        self.eendSpeakerCapacity = eendSpeakerCapacity
    }
}
