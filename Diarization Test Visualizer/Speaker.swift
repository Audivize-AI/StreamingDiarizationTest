//
//  Speaker.swift
//  DiarizationTest
//
//  Created by Benjamin Lee on 12/2/25.
//

import Foundation
import FluidAudio
import OrderedCollections
import Accelerate

class DiarizedTrack {
    public let id: String
    public var count: Int { segments.count }
    
    public private(set) var embedding: EmbeddingVector
    public private(set) var segments: OrderedSet<DiarizedSegment> = []
    
    init(from segment: DiarizedSegment) {
        self.id = segment.speakerId
        self.segments = [segment]
        self.embedding = segment.embedding
    }
    
    public func add(segment newSegment: TimedSpeakerSegment) {
        if let last = segments.last, last.precedes(newSegment) {
            print("TRACK: Appending new segment at index \(segments.endIndex)")
            segments.append(DiarizedSegment(from: newSegment, speakerId: id))
        } else {
            for (i, segment) in segments.enumerated().reversed() {
                if segment.isPartOf(newSegment) {
                    print("TRACK \(id): Absorbing new segment at index \(i)")
                    segment.absorb(newSegment)
                    
                    // Absorb neighboring segments that overlap
                    var i = i
                    while i > 0 && segments[i-1].isPartOf(segment) {
                        print("TRACK \(id): Merging into previous segment at index \(i)")
                        segment.absorb(segments.remove(at: i - 1))
                        i -= 1
                    }
                    break
                } else if segment.precedes(newSegment) {
                    print("TRACK \(id): Inserting new segment at index \(i)")
                    segments.insert(DiarizedSegment(from: newSegment, speakerId: id), at: i+1)
                    break
                }
            }
        }
        
        self.embedding.update(with: newSegment)
        
        var maxTime = -Double.infinity
        
        for segment in segments {
            if segment.start <= maxTime {
                print("ERROR: TRACK \(id) has overlapping tracks")
            }
            maxTime = max(maxTime, Double(segment.end))
        }
    }
    
    public func remove(segment: DiarizedSegment) {
        segments.remove(segment)
    }
    
    @discardableResult
    public func reassignLastSegment(ifMatching segment: TimedSpeakerSegment, to newId: String) -> DiarizedSegment? {
        var index = segments.endIndex - 1
        var match: DiarizedSegment? = nil
        while index >= 0 && !segments[index].precedes(segment) {
            if segments[index].matchesRange(of: segment) {
                match = segments[index]
                break
            }
            index -= 1
        }
        
        guard let match else {
            return nil
        }
        
        print("moving segment from \(id) to \(newId) (distance = \(cosineDistance(from: embedding.embedding, to: segment.embedding))")
        match.speakerId = newId
        return segments.remove(at: index)
    }
    
//    /// Split this track into two clusters, assigning segments to the closest one
//    /// - Parameters:
//    ///   - splitEmbedding: Centroid of the cluster to extract
//    ///   - splitId: Speaker ID of the new cluster
//    ///   - newCentroid: Cluster centroid to use for this track
//    /// - Returns: The extracted track
//    public func split(extracting splitEmbedding: [Float], toID splitId: String, keeping newCentroid: [Float]?) -> DiarizedTrack {
//        let newCentroid = newCentroid ?? self.embedding
//
//    }
}
