//
//  SpeakerDatabase.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/19/26.
//

import Foundation

public class SpeakerDatabase {
    public let config: ClusteringConfig
    public private(set) var inactiveSpeakers: [SpeakerProfile] = []
    public private(set) var activeSpeakers: [Int : SpeakerProfile] = [:]
    public private(set) var isOverflowing: Bool = false
    
    private var nextSpeakerId: Int = 0
    
    public var droppableSlots: [Int] {
        activeSpeakers.filter{ !$1.hasOutliers }.map(\.key)
    }
    
    public var numSlots: Int {
        config.numSlots
    }
    
    public var hasVacantSlots: Bool {
        activeSpeakers[numSlots - 1] == nil
    }
    
    // MARK: - Init
    public init(config: SortformerTimelineConfig) {
        self.config = .init(from: config)
        self.activeSpeakers.reserveCapacity(config.numSpeakers)
    }
    
    public func stream<CFinalized, CTentative>(
        newFinalized: CFinalized,
        newTentative: CTentative,
        onSlotFreed: (@escaping (Int) -> Void)?
    ) where CFinalized: Sequence, CFinalized.Element == EmbeddingSegment,
            CTentative: Sequence, CTentative.Element == EmbeddingSegment
    {
        var binnedSegments: [Int : (finalized: [EmbeddingSegment], tentative: [EmbeddingSegment])] = [:]
        binnedSegments.reserveCapacity(activeSpeakers.count)
        
        for (slot, _) in activeSpeakers {
            binnedSegments[slot] = ([], [])
        }
        
        for segment in newFinalized {
            binnedSegments[segment.speakerId]?.finalized.append(segment)
        }
        
        for segment in newTentative {
            binnedSegments[segment.speakerId]?.tentative.append(segment)
        }
        
        isOverflowing = false
        
        let checkOutliers = !self.hasVacantSlots
        
        for (slot, (finalized, tentative)) in binnedSegments {
            // Initialize the speaker if needed
            guard let speaker = activeSpeakers[slot] else {
                debugPrint("WARNING: Found a new speaker in the stream, but it was not active.")
                continue
            }
            
            speaker.stream(
                newFinalized: finalized,
                newTentative: tentative,
                updateOutliers: checkOutliers
            )
            
            // Return speaker to their parent if their clusters were moved to another speaker
            guard speaker.hasClusters || (speaker.parent == nil && speaker.hasSegments) else {
                speaker.returnToParent()
                activeSpeakers[slot] = nil
                continue
            }
            
            if speaker.hasOutliers {
                isOverflowing = true
            }
        }
        
        while let droppedSlot = pickSlotToDrop() {
            freeSlot(droppedSlot)
            onSlotFreed?(droppedSlot)
        }
    }
    
    public func finalizeAll() {
        for (_, speaker) in activeSpeakers {
            speaker.finalize()
        }
        for speaker in inactiveSpeakers {
            speaker.finalize()
        }
    }
    
    public func clear() {
        activeSpeakers.removeAll()
        inactiveSpeakers.removeAll()
        nextSpeakerId = 0
    }
    
    public func getSpeaker(atSlot slot: Int) -> SpeakerProfile {
        if let existing = activeSpeakers[slot] {
            return existing
        }
        
        let newSpeaker = SpeakerProfile(config: config, speakerId: nextSpeakerId)
        activeSpeakers[slot] = newSpeaker
        nextSpeakerId += 1
        return newSpeaker
    }
    
    public func freeSlot(_ slot: Int) {
        // Purge the old speaker
        if let speaker = activeSpeakers[slot] {
            if let match = checkInactiveMatches(for: speaker, threshold: config.matchThreshold) {
                match.absorbAndFinalize(speaker)
            } else {
                inactiveSpeakers.append(speaker)
            }
        }
        
        // Take outliers from the speaker with the most outliers
        guard let splitSpeaker = activeSpeakers.values.max(by: {
            $0.outlierWeight > $1.outlierWeight
        }), splitSpeaker.hasOutliers
        else {
            activeSpeakers[slot] = nil
            return
        }
        
        let cannotLink = Set(activeSpeakers.values.map(\.speakerId))
        let newSpeaker = splitSpeaker.takeOutliers(speakerId: nextSpeakerId,
                                                   cannotLink: cannotLink)
        if let match = checkInactiveMatches(for: newSpeaker, threshold: config.matchThreshold) {
            newSpeaker.speakerId = match.speakerId
        } else {
            nextSpeakerId += 1
        }
        
        // move slots over
        for oldSlot in slot..<config.numSlots {
            activeSpeakers[oldSlot-1] = activeSpeakers[slot]
        }
        
        activeSpeakers[config.numSlots-1] = newSpeaker
    }
    
    private func checkInactiveMatches(for speaker: SpeakerProfile, threshold: Float) -> SpeakerProfile? {
        var bestDistance: Float = threshold.nextUp
        var bestMatch: SpeakerProfile? = nil
        
        for candidate in inactiveSpeakers {
            let distance = speaker.distance(to: candidate)
            if distance <= bestDistance {
                bestMatch = candidate
                bestDistance = distance
            }
        }
        
        return bestMatch
    }
    
    private func pickSlotToDrop() -> Int? {
        guard isOverflowing else { return nil }
        
        guard let droppedSlot = droppableSlots.min(by: {
            guard let end0 = activeSpeakers[$0]?.lastActiveFrame,
                  let end1 = activeSpeakers[$1]?.lastActiveFrame else {
                return false
            }
            return end0 < end1
        }) else {
            return nil
        }
        
        return droppedSlot
    }
}
