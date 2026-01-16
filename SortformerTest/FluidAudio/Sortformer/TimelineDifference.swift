//
//  SortformerTimelineDifference.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/15/26.
//

import Foundation
import Accelerate


/// Represents an operation to modify the timeline
public protocol SortformerTimelineDelta: SpeakerFrameRange {
    var operation: SortformerTimelineDeltaType { get }
}

public enum SortformerTimelineDeltaType: Equatable {
    case insertion
    case deletion
}

public struct SortformerTimelineDifference {
    public var insertions: [SortformerTimelineInsertion]
    public var deletions: [SortformerTimelineDeletion]
    public var updates: [SortformerTimelineUpdate]
    
    /// Compute the difference between old and new segment arrays
    /// Assumes the segments for each speaker are disjoint and sorted from oldest to newest
    /// Two segments are considered "matched" if they overlap (same segment with possibly updated boundaries)
    public init(
        oldSegments: [[SortformerSegment]],
        newSegments: [[SortformerSegment]]
    ) {
        var insertions: [SortformerTimelineInsertion] = []
        var deletions: [SortformerTimelineDeletion] = []
        var updates: [SortformerTimelineUpdate] = []
        
        let numSpeakers = min(oldSegments.count, newSegments.count)
        
        for speakerIndex in 0..<numSpeakers {
            let oldSpeaker = oldSegments[speakerIndex]
            let newSpeaker = newSegments[speakerIndex]
            
            var oldIdx = 0
            var newIdx = 0
            
            // Two-pointer approach: walk through both sorted lists
            while oldIdx < oldSpeaker.count && newIdx < newSpeaker.count {
                let oldSeg = oldSpeaker[oldIdx]
                let newSeg = newSpeaker[newIdx]
                
                // Check if segments overlap: oldStart < newEnd && newStart < oldEnd
                let overlaps = oldSeg.startFrame < newSeg.endFrame && newSeg.startFrame < oldSeg.endFrame
                
                if overlaps {
                    // Segments match - check if boundaries changed
                    if oldSeg.startFrame != newSeg.startFrame || oldSeg.endFrame != newSeg.endFrame {
                        updates.append(SortformerTimelineUpdate(
                            segmentID: oldSeg.id,
                            speakerIndex: speakerIndex,
                            oldStartFrame: oldSeg.startFrame,
                            oldEndFrame: oldSeg.endFrame,
                            newStartFrame: newSeg.startFrame,
                            newEndFrame: newSeg.endFrame
                        ))
                    }
                    // Move both pointers forward
                    oldIdx += 1
                    newIdx += 1
                } else if oldSeg.endFrame <= newSeg.startFrame {
                    // Old segment ends before new segment starts - old is deleted
                    deletions.append(SortformerTimelineDeletion(from: oldSeg))
                    oldIdx += 1
                } else {
                    // New segment ends before old segment starts - new is inserted
                    insertions.append(SortformerTimelineInsertion(
                        speakerIndex: speakerIndex,
                        startFrame: newSeg.startFrame,
                        endFrame: newSeg.endFrame
                    ))
                    newIdx += 1
                }
            }
            
            // Any remaining old segments are deletions
            while oldIdx < oldSpeaker.count {
                deletions.append(SortformerTimelineDeletion(from: oldSpeaker[oldIdx]))
                oldIdx += 1
            }
            
            // Any remaining new segments are insertions
            while newIdx < newSpeaker.count {
                let newSeg = newSpeaker[newIdx]
                insertions.append(SortformerTimelineInsertion(
                    speakerIndex: speakerIndex,
                    startFrame: newSeg.startFrame,
                    endFrame: newSeg.endFrame
                ))
                newIdx += 1
            }
        }
        
        self.insertions = insertions
        self.deletions = deletions
        self.updates = updates
    }
    
    /// Empty difference
    public init() {
        self.insertions = []
        self.deletions = []
        self.updates = []
    }
    
    /// Check if there are any changes
    public var isEmpty: Bool {
        insertions.isEmpty && deletions.isEmpty && updates.isEmpty
    }
    
    /// Total number of changes
    public var count: Int {
        insertions.count + deletions.count + updates.count
    }
}

public struct SortformerTimelineInsertion: SortformerTimelineDelta {
    public let speakerIndex: Int
    public let startFrame: Int
    public let endFrame: Int
    public let operation: SortformerTimelineDeltaType = .insertion
    public var frames: Range<Int> { startFrame..<endFrame }
    public var length: Int { endFrame - startFrame }
    
    public init(speakerIndex: Int, startFrame: Int, endFrame: Int) {
        self.speakerIndex = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
    }
    
    public func contains(_ frame: Int) -> Bool {
        return frames.contains(frame)
    }
    
    public func isContiguous<T>(with other: T) -> Bool
    where T : SortformerFrameRange {
        return startFrame <= other.endFrame && endFrame >= other.startFrame
    }
    
    public func overlaps<T>(with other: T) -> Bool
    where T : SortformerFrameRange {
        return frames.overlaps(other.frames)
    }
}

public struct SortformerTimelineDeletion: SortformerTimelineDelta {
    public let segmentID: UUID
    public let speakerIndex: Int
    public let startFrame: Int
    public let endFrame: Int
    public var length: Int { endFrame - startFrame }
    public var frames: Range<Int> { startFrame..<endFrame }
    public let operation: SortformerTimelineDeltaType = .deletion
    
    public init(id: UUID, speakerIndex: Int, startFrame: Int, endFrame: Int) {
        self.segmentID = id
        self.speakerIndex = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
    }
    
    public init(from segment: SortformerSegment) {
        self.segmentID = segment.id
        self.speakerIndex = segment.speakerIndex
        self.startFrame = segment.startFrame
        self.endFrame = segment.endFrame
    }
    
    public func contains(_ frame: Int) -> Bool {
        return frames.contains(frame)
    }
    
    public func isContiguous<T>(with other: T) -> Bool
    where T : SortformerFrameRange {
        return startFrame <= other.endFrame && endFrame >= other.startFrame
    }
    
    public func overlaps<T>(with other: T) -> Bool
    where T : SortformerFrameRange {
        return frames.overlaps(other.frames)
    }
}

/// Represents an update to an existing segment's boundaries
public struct SortformerTimelineUpdate {
    public let segmentID: UUID
    public let speakerIndex: Int
    public let oldStartFrame: Int
    public let oldEndFrame: Int
    public let newStartFrame: Int
    public let newEndFrame: Int
    
    public init(
        segmentID: UUID,
        speakerIndex: Int,
        oldStartFrame: Int,
        oldEndFrame: Int,
        newStartFrame: Int,
        newEndFrame: Int
    ) {
        self.segmentID = segmentID
        self.speakerIndex = speakerIndex
        self.oldStartFrame = oldStartFrame
        self.oldEndFrame = oldEndFrame
        self.newStartFrame = newStartFrame
        self.newEndFrame = newEndFrame
    }
}
