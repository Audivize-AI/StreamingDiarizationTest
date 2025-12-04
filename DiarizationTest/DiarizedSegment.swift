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
    public let speakerId: String
    public var string: String { "Speaker \(speakerId): \(start)s - \(end)s" }
    public private(set)var start: TimeInterval
    public private(set) var end: TimeInterval
    public private(set) var centroid: [Float]
    public private(set) var embeddingQuality: Float
    
    private var startCount: Int = 1
    private var endCount: Int = 1
    
    public init(from segment: TimedSpeakerSegment) {
        self.start = TimeInterval(segment.startTimeSeconds)
        self.end = TimeInterval(segment.endTimeSeconds)
        self.speakerId = segment.speakerId
        self.embeddingQuality = segment.qualityScore
        self.centroid = unitize(segment.embedding)
    }
    
    public func isPartOf(_ other: DiarizedSegment) -> Bool {
        guard self.speakerId == other.speakerId else { return false }
        return (self.start - Config.boundaryTolerance < other.end) &&
            (other.start - Config.boundaryTolerance < self.end)
    }
    
    public func isPartOf(_ other: TimedSpeakerSegment) -> Bool {
        guard self.speakerId == other.speakerId else { return false }
        return (self.start < Double(other.endTimeSeconds)) && (Double(other.startTimeSeconds) < self.end)
    }
    
    public func cosineDistance(to embedding: [Float]) -> Float {
        return vDSP.dot(centroid, embedding) / sqrt(vDSP.sumOfSquares(embedding))
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    public static func ==(_ a: DiarizedSegment, _ b: DiarizedSegment) -> Bool {
        return a.id == b.id
    }
    
    public func absorb(_ other: TimedSpeakerSegment) {
        self.absorb(startTime: Double(other.startTimeSeconds),
                    endTime: Double(other.endTimeSeconds),
                    embedding: other.embedding,
                    qualityScore: other.qualityScore)
    }
    
    public func absorb(_ other: DiarizedSegment) {
        self.absorb(startTime: Double(other.start),
                    endTime: Double(other.end),
                    embedding: other.centroid,
                    qualityScore: other.embeddingQuality)
    }
    
    private func absorb(startTime newStart: Double, endTime newEnd: Double, embedding: [Float], qualityScore: Float) {
        
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
        updateEmbedding(centroid: &centroid,
                        totalQuality: &embeddingQuality,
                        with: embedding,
                        quality: qualityScore)
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
