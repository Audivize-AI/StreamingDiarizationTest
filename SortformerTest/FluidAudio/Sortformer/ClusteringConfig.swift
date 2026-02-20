//
//  ClusteringConfig.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/18/26.
//

import Foundation
import AHCClustering

struct ClusteringConfig {
    /// Linkage Policy (`wardLinkage`, `upgma`, `singleLinkage`, or `completeLinkage`)
    var linkagePolicy: LinkagePolicyType = .completeLinkage
    
    /// Minimum separation distance to avoid merging clusters upon compression
    var minSeparation: Float
    
    /// Intra-cluster separation threshold
    var clusteringThreshold: Float

    /// Alias used by dendrogram visualization to decide whether a merge exceeds linkage.
    var mergeThreshold: Float {
        get { clusteringThreshold }
        set { clusteringThreshold = newValue }
    }
    
    /// Maximum number of finalized embeddings
    var maxEmbeddings: Int
    
    /// Number of finalized embeddings to keep when compressed
    var minEmbeddings: Int
    
    /// Maximum number of representatives per speaker
    var maxRepresentatives: Int
    
    /// Maximum number of speakers supported by the EEND model
    var eendSpeakerCapacity: Int
    
    init(
        linkagePolicy: LinkagePolicyType = .upgma,
        minSeparation: Float = 0.1,
        clusteringThreshold: Float = 0.3,
        maxEmbeddings: Int = 20,
        minEmbeddings: Int = 8,
        maxRepresentatives: Int = 4,
        eendSpeakerCapacity: Int = 4,
    ) {
        self.linkagePolicy = linkagePolicy
        self.clusteringThreshold = clusteringThreshold
        self.minSeparation = minSeparation
        self.minEmbeddings = minEmbeddings
        self.maxEmbeddings = maxEmbeddings
        self.maxRepresentatives = maxRepresentatives
        self.eendSpeakerCapacity = eendSpeakerCapacity
    }
}
