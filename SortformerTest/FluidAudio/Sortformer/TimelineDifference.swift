//
//  SortformerTimelineDifference.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/15/26.
//

import Foundation
import Accelerate
import OrderedCollections


public struct SortformerTimelineDifference {
    public private(set) var insertions: Set<SortformerSegment> = []
    public private(set) var deletions: Set<SortformerSegment> = []
    
    public var isEmpty: Bool {
        return insertions.isEmpty && deletions.isEmpty
    }
    
    public var inverse: SortformerTimelineDifference {
        return SortformerTimelineDifference(
            insertions: deletions,
            deletions: insertions
        )
    }
    
    public init(
        insertions: Set<SortformerSegment>,
        deletions: Set<SortformerSegment>
    ) {
        self.insertions = insertions
        self.deletions  = deletions
    }
    
    /// Compute the difference between old and new segment arrays
    /// Assumes the segments for each speaker are disjoint and sorted from oldest to newest
    /// Two segments are considered "matched" if they are identical
    public init(
        old: [SortformerSegment],
        new: [SortformerSegment]
    ) {
        let oldSet = Set(old)
        let newSet = Set(new)
        
        insertions = newSet.subtracting(oldSet)
        deletions  = oldSet.subtracting(newSet)
    }
    
    /// Empty difference
    public init() {
        self.insertions = []
        self.deletions = []
    }
    
    public mutating func insert(_ segment: SortformerSegment) {
        // Try cancelling with a deletion
        if deletions.contains(segment) {
            deletions.remove(segment)
        } else {
            insertions.insert(segment)
        }
    }
    
    public mutating func insert(_ segments: [SortformerSegment]) {
        insertions.formUnion(Set(segments).subtracting(deletions))
    }
    
    public mutating func delete(_ segment: SortformerSegment) {
        // Try cancelling with an insertion
        if insertions.contains(segment) {
            insertions.remove(segment)
        } else {
            deletions.insert(segment)
        }
    }
    
    public mutating func delete(_ segments: [SortformerSegment]) {
        deletions.formUnion(Set(segments).subtracting(insertions))
    }
    
    public mutating func replace(_ segment: SortformerSegment, with newSegment: SortformerSegment) {
        delete(segment)
        insert(newSegment)
    }
    
    public mutating func merge(_ segments: [SortformerSegment], into newSegment: SortformerSegment) {
        delete(segments)
        insert(newSegment)
    }
    
    public mutating func split(_ segment: SortformerSegment, into newSegments: [SortformerSegment]) {
        delete(segment)
        insert(newSegments)
    }
    
    /// Merge another difference into this one
    /// Insertions and deletions from the other difference are added to this one
    /// with proper cancellation (insertion + deletion of same segment = no-op)
    public mutating func apply(_ other: SortformerTimelineDifference) {
        let ins = other.insertions.subtracting(deletions)
        let del = other.deletions.subtracting(insertions)
        insertions.subtract(other.deletions)
        deletions.subtract(other.insertions)
        insertions.formUnion(ins)
        deletions.formUnion(del)
    }
    
    public func apply(to timeline: inout OrderedSet<SortformerSegment>) -> Bool {
        guard deletions.isSubset(of: timeline) else {
            return false
        }
        
        timeline.subtract(deletions)
        for insertion in insertions {
            let index = timeline.lastIndex { $0 < insertion }.map { $0 + 1 } ?? 0
            timeline.updateOrInsert(insertion, at: index)
        }
        
        return true
    }
    
    // TODO: Optimize this
    /// - Note: Timeline *must* be sorted chronologically from oldest to newest
    public func compile<C>(for timeline: C) -> (deletions: [Int], insertions: [(index: Int, segment: SortformerSegment)])?
    where C: RandomAccessCollection, C.Element == SortformerSegment, C.Index == Int {
        var deletionIndices: [Int] = []
        deletionIndices.reserveCapacity(deletions.count)
        
        for deletion in deletions {
            guard let index = timeline.lastIndex(of: deletion) else {
                return nil
            }
            deletionIndices.append(index)
        }
        
        deletionIndices.sort(by: >)
        
        var compiledInsertions: [(index: Int, segment: SortformerSegment)] = []
        for insertion in insertions {
            var index = timeline.lastIndex { $0 < insertion }.map { $0 + 1 } ?? 0
            index -= deletions.count { $0 < insertion }
            compiledInsertions.append((index, insertion))
        }
        
        compiledInsertions.sort {
            $0.index > $1.index
        }
        
        return (deletions: deletionIndices, insertions: compiledInsertions)
    }
}

/// Represents an operation to modify the timeline

public protocol SortformerTimelineDelta: Hashable {
    associatedtype Inverse: SortformerTimelineDelta
    
    var inverse: Inverse { get }
    
    /// Apply the delta to the timeline and return true upon success
    /// - Parameter timeline: Timeline shaped as [numSegments]
    /// - Returns: `true` if successful, `false` if not
    func apply(to timeline: inout [SortformerSegment]) -> Bool
    
    /// Apply the delta to the timeline and return true upon success
    /// - Parameter timeline: Timeline shaped as [numSpeakers, numSegments]
    /// - Returns: `true` if successful, `false` if not
    func apply(to timeline: inout [[SortformerSegment]]) -> Bool
}

public struct SortformerTimelineInsertion: SortformerTimelineDelta {
    public typealias Inverse = SortformerTimelineDeletion
    
    public let segment: SortformerSegment
    
    public var inverse: Inverse {
        .init(segment)
    }
    
    public init(_ segment: SortformerSegment) {
        self.segment = segment
    }
    
    public static func == (lhs: SortformerTimelineInsertion, rhs: SortformerTimelineInsertion) -> Bool {
        return lhs.segment == rhs.segment
    }
    
    public func apply(to timeline: inout [SortformerSegment]) -> Bool {
        // Binary search for correct insertion point to maintain sorted order
        // Find first index where existing segment >= our segment
        let insertionIndex = timeline.firstIndex { $0 >= segment } ?? timeline.count
        timeline.insert(segment, at: insertionIndex)
        return true
    }
    
    public func apply(to timeline: inout [[SortformerSegment]]) -> Bool {
        let spkIdx = segment.speakerIndex
        while spkIdx >= timeline.count {
            timeline.append([])
        }
        
        // Find first index where existing segment >= our segment
        let insertionIndex = timeline[spkIdx].firstIndex { $0 >= segment } ?? timeline[spkIdx].count
        timeline[spkIdx].insert(segment, at: insertionIndex)
        return true
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(segment.key)
    }
}

public struct SortformerTimelineDeletion: SortformerTimelineDelta {
    public typealias Inverse = SortformerTimelineInsertion
    
    public let segment: SortformerSegment
    
    public var inverse: Inverse {
        .init(segment)
    }
    
    public init(_ segment: SortformerSegment) {
        self.segment = segment
    }
    
    public static func == (lhs: SortformerTimelineDeletion, rhs: SortformerTimelineDeletion) -> Bool {
        return lhs.segment == rhs.segment
    }
    
    public func apply(to timeline: inout [SortformerSegment]) -> Bool {
        guard let index = timeline.firstIndex(of: segment) else {
            return false
        }
        
        timeline.remove(at: index)
        return true
    }
    
    public func apply(to timeline: inout [[SortformerSegment]]) -> Bool {
        let spkIdx = segment.speakerIndex
        
        guard spkIdx < timeline.count, let index = timeline[spkIdx].firstIndex(of: segment)
        else {
            return false
        }
        
        timeline[spkIdx].remove(at: index)
        return true
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(segment.key)
    }
}
