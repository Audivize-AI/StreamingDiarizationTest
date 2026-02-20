//
//  SpeakerDatabase.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/19/26.
//

import Foundation

class SpeakerDatabase {
    let config: ClusteringConfig
    var slots: [Int] = []
    var speakers: [SpeakerProfile] = []
    var slotsWithOutliers: [Int] = []
    
    init(config: ClusteringConfig) {
        self.config = config
    }
    
    public func stream(newFinalized: [EmbeddingSegment], newTentative: [EmbeddingSegment]) {
        var sortedFinalized: [[EmbeddingSegment]] = []
        var sortedTentative: [[EmbeddingSegment]] = []
    }
}
