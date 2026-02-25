//
//  SpeakerDatabase.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/19/26.
//

import Foundation

public class SpeakerDatabase {
    public let config: ClusteringConfig
    public private(set) var inactiveSpeakers: [Int : SpeakerProfile] = [:]
    public private(set) var activeSpeakers: [Int : SpeakerProfile] = [:]
    public private(set) var isOverflowing: Bool
    
    private var nextSpeakerId: Int = 0
    
    public var droppableSpeakerIndices: [Int] {
        activeSpeakers.filter{ !$1.hasOutliers }.map(\.key)
    }
    
    public var numSlots: Int {
        config.numSlots
    }
    
    public var hasVacantSlots: Bool {
        activeSpeakers[numSlots - 1] == nil
    }
    
    public init(config: SortformerTimelineConfig) {
        self.config = .init(from: config)
        self.activeSpeakers.reserveCapacity(numSlots)
    }
    
    public func stream(
        newFinalized: [EmbeddingSegment],
        newTentative: [EmbeddingSegment]
    ) {
        var binnedSegments: [Int : (finalized: [EmbeddingSegment], tentative: [EmbeddingSegment])] = [:]
        binnedSegments.reserveCapacity(numSlots)
        
        for segment in newFinalized {
            binnedSegments[segment.speakerId, default: ([], [])].finalized
                .append(segment)
        }
        
        for segment in newTentative {
            binnedSegments[segment.speakerId, default: ([], [])].tentative
                .append(segment)
        }
        
        isOverflowing = false
        
        let checkOutliers = !self.hasVacantSlots
        
        for (speakerIndex, (finalized, tentative)) in binnedSegments {
            // Initialize the speaker if needed
            guard let speaker = activeSpeakers[speakerIndex] else {
                debugPrint("WARNING: Found a new speaker in the stream, but it was not active.")
                continue
            }
            
            speaker.stream(
                newFinalized: finalized,
                newTentative: tentative,
                updateOutliers: checkOutliers
            )
            
            if speaker.hasOutliers {
                isOverflowing = true
            }
        }
    }
    
    public func freeSlot(_ slot: Int) {
        // Purge the old slot
        if let speaker = activeSpeakers[slot] {
            if let matchIndex = checkInactiveMatches(for: speaker, threshold: config.matchThreshold) {
                inactiveSpeakers[matchIndex]?.absorb(speaker)
            }
        }
        
        guard let sourceIndex = slotToSpeaker.values.first(where: { speakers[$0].hasOutliers }) else {
            
            slotToSpeaker[slot] = nil
            return
        }
        
        let newSpeaker = speakers[sourceIndex].fromOutliers(slot: slot, speakerIndex: speakers.count)
        if let newIndex = checkInactiveMatches(for: newSpeaker, threshold: config.matchThreshold) {
            newSpeaker.speakerIndex = newIndex
        }
        slotToSpeaker[slot] = newSpeaker.speakerIndex
        speakers.append(newSpeaker)
    }
    
    public func nextSpeakerIndex() -> Int {
        
    }
    
    private func checkInactiveMatches(for speaker: SpeakerProfile, threshold: Float) -> Int? {
        var bestDistance: Float = threshold.nextUp
        var bestMatch: Int? = nil
        
        for candidate in speakers where candidate.slot == nil {
            let distance = speaker.distance(to: candidate)
            if distance <= bestDistance {
                bestMatch = candidate.speakerIndex
                bestDistance = distance
            }
        }
        
        return bestMatch
    }
}
