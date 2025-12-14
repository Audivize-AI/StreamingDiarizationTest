//
//  Segment.swift
//  DiarizationTest
//
//  Created by Benjamin Lee on 12/2/25.
//

import Foundation
import FluidAudio
import Accelerate

class DiarizedSegment: Hashable, Identifiable {
    public let id: UUID = UUID()
    public var speakerId: String
    public var string: String { "Speaker \(speakerId): \(String(format: "%.1f", start))s - \(String(format: "%.1f", end))s" }
    public private(set) var start: TimeInterval
    public private(set) var end: TimeInterval
    public private(set) var embedding: EmbeddingVector
    public var duration: TimeInterval { end - start }
    
    private var startCount: Int = 1
    private var endCount: Int = 1
    
    public init(from segment: TimedSpeakerSegment, speakerId: String) {
        self.start = TimeInterval(segment.startTimeSeconds)
        self.end = TimeInterval(segment.endTimeSeconds)
        self.speakerId = speakerId
        self.embedding = EmbeddingVector(from: segment)
    }
    
    public func isPartOf(_ other: DiarizedSegment) -> Bool {
        guard self.speakerId == other.speakerId else { return false }
        return (self.start < other.end) && (other.start < self.end)
    }
    
    public func isPartOf(_ other: TimedSpeakerSegment) -> Bool {
        return (self.start < Double(other.endTimeSeconds)) && (Double(other.startTimeSeconds) < self.end)
    }
    
    public func matchesRange(of segment: DiarizedSegment) -> Bool {
        matchesRange(startTime: segment.start, endTime: segment.end)
    }
    
    public func matchesRange(of segment: TimedSpeakerSegment) -> Bool {
        matchesRange(startTime: Double(segment.startTimeSeconds), endTime: Double(segment.endTimeSeconds))
    }
    
    public func matchesRange(startTime: TimeInterval, endTime: TimeInterval) -> Bool {
        return iou(startTime: startTime, endTime: endTime) > Config.iouMatchingThreshold
    }
    
    public func iou(startTime: TimeInterval, endTime: TimeInterval) -> Double {
        let intersection = min(end, endTime) - max(start, startTime)
        
        guard intersection > 0 else {
            return 0
        }
        
        let union = duration + (endTime - startTime) - intersection
        return intersection / union
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    public static func ==(_ a: DiarizedSegment, _ b: DiarizedSegment) -> Bool {
        return a.id == b.id
    }
    
    public func absorb(_ other: TimedSpeakerSegment) {
        let newStart = Double(other.startTimeSeconds)
        let newEnd = Double(other.endTimeSeconds)
        
        // Update start time
        if newStart < start - Config.boundaryTolerance {
            start = newStart
            startCount = 1
        } else if newStart < start + Config.boundaryTolerance {
            start = (start * Double(startCount) + newStart) / Double(startCount + 1)
            startCount += 1
        }
        
        // Update end time
        if newEnd > end + Config.boundaryTolerance {
            end = newEnd
            endCount = 1
        } else if newEnd > end - Config.boundaryTolerance {
            end = (end * Double(endCount) + newEnd) / Double(endCount + 1)
            endCount += 1
        }
        
        // Update embedding centroid
        embedding.update(with: other)
    }
    
    public func absorb(_ other: DiarizedSegment) {
        // Update start time
        if other.start < start - Config.boundaryTolerance {
            // other segment starts before this one
            start = other.start
            startCount = other.startCount
        } else if other.start < start + Config.boundaryTolerance {
            // other segment starts within tolerance
            start = (start * Double(startCount) + other.start * Double(other.startCount)) / Double(startCount + other.startCount)
            startCount += other.startCount
        }
        
        // Update end time
        if other.end > end + Config.boundaryTolerance {
            // other segment ends after this one
            end = other.end
            endCount = other.endCount
        } else if other.end > end - Config.boundaryTolerance {
            // other segment ends within tolerance
            end = (end * Double(endCount) + other.end * Double(other.endCount)) / Double(endCount + other.endCount)
            endCount += other.endCount
        }
        
        // update embedding
        embedding.update(with: other.embedding)
    }
}

extension DiarizedSegment {
    public func startsBefore(_ other: DiarizedSegment) -> Bool {
        return self.start < other.start
    }
    
    public func endsBefore(_ other: DiarizedSegment) -> Bool {
        return self.end < other.end
    }
    
    public func startsAfter(_ other: DiarizedSegment) -> Bool {
        return self.start > other.start
    }
    
    public func endsAfter(_ other: DiarizedSegment) -> Bool {
        return self.end > other.end
    }
    
    public func precedes(_ other: DiarizedSegment) -> Bool {
        return self.end < other.start
    }
    
    public func succeeds(_ other: DiarizedSegment) -> Bool {
        return self.start > other.end
    }
    
    public func startsBefore(_ other: TimedSpeakerSegment) -> Bool {
        return self.start < Double(other.startTimeSeconds)
    }
    
    public func endsBefore(_ other: TimedSpeakerSegment) -> Bool {
        return self.end < Double(other.endTimeSeconds)
    }
    
    public func startsAfter(_ other: TimedSpeakerSegment) -> Bool {
        return self.start > Double(other.startTimeSeconds)
    }
    
    public func endsAfter(_ other: TimedSpeakerSegment) -> Bool {
        return self.end > Double(other.endTimeSeconds)
    }
    
    public func precedes(_ other: TimedSpeakerSegment) -> Bool {
        return self.end < Double(other.startTimeSeconds)
    }
    
    public func succeeds(_ other: TimedSpeakerSegment) -> Bool {
        return self.start > Double(other.endTimeSeconds)
    }
}
