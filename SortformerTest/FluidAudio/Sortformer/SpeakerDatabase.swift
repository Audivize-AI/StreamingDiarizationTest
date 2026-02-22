//
//  SpeakerDatabase.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/19/26.
//

import Foundation

class SpeakerDatabase {
    public let config: ClusteringConfig
    public private(set)var speakers: [SpeakerProfile] = []
    public private(set) var slotToSpeaker: [Int] = []
    public private(set) var slotsWithOutliers: [Int] = []
    
    public init(config: ClusteringConfig) {
        self.config = config
    }
    
    public func stream(newFinalized: [EmbeddingSegment], newTentative: [EmbeddingSegment]) {
        var sortedFinalized: [[EmbeddingSegment]] = Array(repeating: [], count: config.eendSpeakerCapacity)
        var sortedTentative: [[EmbeddingSegment]] = Array(repeating: [], count: config.eendSpeakerCapacity)
        
        for segment in newFinalized {
            sortedFinalized[segment.slot].append(segment)
        }
        
        for segment in newTentative {
            sortedTentative[segment.slot].append(segment)
        }
        
        slotsWithOutliers.removeAll(keepingCapacity: true)
        
        let checkOutliers = slotToSpeaker.count >= config.eendSpeakerCapacity
        
        for slot in 0..<config.eendSpeakerCapacity {
            // Initialize the speaker if they don't exist
            if slotToSpeaker.count <= slot {
                slotToSpeaker.append(speakers.count)
                speakers.append(
                    SpeakerProfile(config: config, speakerIndex: speakers.count, slot: slot))
            }
            
            // Update speaker embeddings
            let speaker = speakers[slotToSpeaker[slot]]
            
            speaker.stream(
                newFinalized: sortedFinalized[slot],
                newTentative: sortedTentative[slot],
                updateOutliers: checkOutliers
            )
            
            if speaker.hasOutliers {
                slotsWithOutliers.append(slot)
            }
        }
    }
    
    public func selectSlotToDrop() -> Int? {
        // Free a slot if outliers are found
        guard !slotsWithOutliers.isEmpty else {
            return nil
        }
        
        // Pick the a slot without outliers with the most embeddings
        let droppableSlots = (0..<config.eendSpeakerCapacity).filter { !slotsWithOutliers.contains($0)
        }
        
        guard !droppableSlots.isEmpty else {
            print("All slots have outliers. Can't drop a slot.")
            return nil
        }
        
        return droppableSlots.max {
            speakers[$0].weight > speakers[$1].weight
        }
    }
    
    public func drop(slot: Int) {
        let splitSlot = slotsWithOutliers.removeFirst()
//        slotToSpeaker.remove(at: slot) = -1
    }
}
