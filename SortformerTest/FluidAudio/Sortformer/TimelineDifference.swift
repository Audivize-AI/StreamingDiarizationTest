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
