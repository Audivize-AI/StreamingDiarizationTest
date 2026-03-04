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
    
    private var nextSpeakerId: Int = 0
    
    public var droppableSlots: [Int] {
        activeSpeakers
            .filter { $1.isDroppable }
            .map(\.key)
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
    
    public func stream<F, T>(
        newFinalized: F,
        newTentative: T,
        onSlotFreed: ((Int) -> Void)?
    ) where F: Sequence, F.Element == EmbeddingSegment,
            T: Sequence, T.Element == EmbeddingSegment
    {
        var binnedSegments: [Int : (finalized: [EmbeddingSegment], tentative: [EmbeddingSegment])] = [:]
        var segmentCounts: [Int : Int] = [:]
        binnedSegments.reserveCapacity(activeSpeakers.count)
        segmentCounts.reserveCapacity(activeSpeakers.count)
        
        for (slot, _) in activeSpeakers {
            binnedSegments[slot] = ([], [])
            segmentCounts[slot] = 0
        }
        
        for segment in newFinalized {
            binnedSegments[segment.speakerId]?.finalized.append(segment)
            segmentCounts[segment.speakerId]? += 1
        }
        
        for segment in newTentative {
            binnedSegments[segment.speakerId]?.tentative.append(segment)
            segmentCounts[segment.speakerId]? += 1
        }
        
        var numSlotsToDrop = 0
        
        let orderedSegments = segmentCounts
            .sorted { $0.value < $1.value }
            .map { ($0.key, binnedSegments[$0.key]!) }
        
        for (slot, (finalized, tentative)) in orderedSegments {
            // Initialize the speaker if needed
            guard let speaker = activeSpeakers[slot] else {
                debugPrint("WARNING: Found a new speaker in the stream, but it was not active.")
                continue
            }
            
            speaker.stream(
                newFinalized: finalized,
                newTentative: tentative,
                checkOutliers: activeSpeakers.count >= numSlots
            )
            
            // Get rid empty speakers
            guard speaker.hasSegments else {
                activeSpeakers[slot] = nil
                numSlotsToDrop -= 1
                continue
            }
            
            if speaker.hasOutliers {
                numSlotsToDrop += 1
            }
        }
        
        updateSpeakerIds()
        
        guard numSlotsToDrop > 0 else { return }
        var droppedSlots: [Int] = []
        droppedSlots.reserveCapacity(numSlotsToDrop)
        
        for _ in 0..<numSlotsToDrop {
            guard let droppedSlot = pickSlotToDrop(otherThan: droppedSlots) else {
                break
            }
            droppedSlots.append(droppedSlot)
            freeSlot(droppedSlot, refindMatch: false)
            onSlotFreed?(droppedSlot)
            print("Dropping slot: \(droppedSlot)")
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
        nextSpeakerId += 1
        activeSpeakers[slot] = newSpeaker

        return newSpeaker
    }
    
    public func freeSlot(_ slot: Int, refindMatch: Bool = true) {
        // Remove the old speaker
        if let speaker = activeSpeakers.removeValue(forKey: slot) {
            let removedId = speaker.speakerId
            let match = refindMatch
                ? findMatchingSpeaker(for: speaker)
                : inactiveSpeakers.first { $0.speakerId == removedId }
            if let match {
                match.absorbAndFinalize(speaker)
            } else {
                inactiveSpeakers.append(speaker)
            }
            
            // Only update cannot-link constraints from speakers as they are deactivated to avoid stale IDs
            for other in activeSpeakers.values {
                other.updateCannotLink(with: removedId)
            }
        }
        
        shiftSlotsLeft(startingAt: slot)
    }
    
    private func findMatchingSpeaker(for speaker: SpeakerProfile) -> SpeakerProfile? {
        var bestDistance: Float = config.matchThreshold.nextUp
        var bestMatch: SpeakerProfile? = nil
        
        let activeIds = activeSpeakers.values.map(\.speakerId)
        
        for candidate in inactiveSpeakers {
            // Skip if the speaker is assigned to someone already
            guard !activeIds.contains(candidate.speakerId) else {
                continue
            }
            
            // Determine how close it is
            let distance = speaker.distance(to: candidate)
            if distance <= bestDistance {
                bestMatch = candidate
                bestDistance = distance
            }
        }
        
        return bestMatch
    }
    
    private func pickSlotToDrop(otherThan droppedSlots: [Int] = []) -> Int? {
        guard activeSpeakers.values.contains(where: \.hasOutliers) else {
            return nil
        }
        
        guard activeSpeakers.count + droppedSlots.count == numSlots else {
            // Return the first unused slot
            var isUnused = Array(repeating: true, count: numSlots)
            for slot in activeSpeakers.keys {
                isUnused[slot] = false
            }
            for slot in droppedSlots {
                isUnused[slot] = false
            }
            return isUnused.firstIndex(of: true)
        }
        
        return droppableSlots.min {
            guard let end0 = activeSpeakers[$0]?.lastActiveFrame,
                  let end1 = activeSpeakers[$1]?.lastActiveFrame else {
                return false
            }
            return end0 < end1
        }
    }
    
    private func shiftSlotsLeft(startingAt slot: Int) {
        guard slot < numSlots else { return }
        for i in slot..<numSlots - 1 {
            activeSpeakers[i] = activeSpeakers[i + 1]
        }
        activeSpeakers[numSlots - 1] = nil
    }
    
    private func updateSpeakerIds() {
        guard !inactiveSpeakers.isEmpty else {
            // No inactive speakers to match against — revert to birth IDs
            for (_, speaker) in activeSpeakers {
                speaker.speakerId = speaker.defaultSpeakerId
            }
            return
        }
        
        let threshold = config.matchThreshold

        var numRows = activeSpeakers.count
        var numCols = inactiveSpeakers.count
        
        var costMatrix: [Float] = []
        var rowToSlot: [Int] = []
        var columnToIndex: [Int] = Array(0..<numCols)
        costMatrix.reserveCapacity(numRows * numCols)
        rowToSlot.reserveCapacity(numRows)
        
        var isColumnMatched = Array(repeating: false, count: numCols)
        
        // Build cost matrix
        for (slot, speaker) in activeSpeakers {
            var foundMatch = false
            
            for (i, candidate) in inactiveSpeakers.enumerated() {
                let distance = speaker.distance(to: candidate)
                
                guard distance <= threshold else {
                    costMatrix.append(.infinity)
                    continue
                }
                
                costMatrix.append(distance)
                
                foundMatch = true
                isColumnMatched[i] = true
            }
            
            guard foundMatch else {
                // Undo the row
                costMatrix.removeLast(numCols)
                continue
            }
            
            rowToSlot.append(slot)
        }
        
        numRows = rowToSlot.count
        
        // Note: if numRows > 0, then numCols > 0
        guard numRows > 0 else {
            // No matches found — revert to birth IDs
            for (_, speaker) in activeSpeakers {
                speaker.speakerId = speaker.defaultSpeakerId
            }
            return
        }
        
        // Remove columns with no matches
        while let removedCol = isColumnMatched.lastIndex(of: false) {
            for row in (0..<numRows).reversed() {
                costMatrix.remove(at: row * numCols + removedCol)
            }
            isColumnMatched.remove(at: removedCol)
            columnToIndex.remove(at: removedCol)
            numCols -= 1
        }
        
        // Solve the assignment
        guard let assignments = solveRectangularLinearAssignment(
            numRows: numRows, numCols: numCols, costMatrix: costMatrix) else {
            fatalError("Failed to solve ID assignments")
        }
        
        // Apply matched IDs and collect claimed inactive IDs
        var matchedSlots = Set<Int>()
        var claimedIds = Set<Int>()
        
        for (row, col) in zip(assignments.rows, assignments.cols) {
            let slot = rowToSlot[row]
            let inactiveIndex = columnToIndex[col]
            let matchedId = inactiveSpeakers[inactiveIndex].speakerId
            activeSpeakers[slot]?.speakerId = matchedId
            matchedSlots.insert(slot)
            claimedIds.insert(matchedId)
        }
        
        // Unmatched speakers revert to their birth ID.
        // If a birth ID was claimed by a matched speaker (stolen match),
        // assign a new unique ID to avoid collisions.
        for (slot, speaker) in activeSpeakers where !matchedSlots.contains(slot) {
            if claimedIds.contains(speaker.defaultSpeakerId) {
                speaker.speakerId = nextSpeakerId
                nextSpeakerId += 1
            } else {
                speaker.speakerId = speaker.defaultSpeakerId
            }
        }
    }
}
