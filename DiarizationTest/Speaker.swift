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
    
    public private(set) var embedding: [Float]
    public private(set) var segments: OrderedSet<DiarizedSegment> = []
    private var embeddingQuality: Float
    
    init(from segment: DiarizedSegment) {
        self.id = segment.speakerId
        self.segments = [segment]
        self.embedding = segment.centroid
        self.embeddingQuality = segment.embeddingQuality
    }
    
    init(from segment: TimedSpeakerSegment) {
        self.id = segment.speakerId
        self.segments = [DiarizedSegment(from: segment)]
        self.embedding = vDSP.multiply(
            1.0 / sqrt(vDSP.sumOfSquares(segment.embedding)),
            segment.embedding
        )
        self.embeddingQuality = segment.qualityScore
    }
    
    public func add(segment newSegment: TimedSpeakerSegment) {        
        guard segments.last?.precedes(newSegment) == false else {
            print("TRACK: Appending new segment at index \(segments.endIndex)")
            segments.append(DiarizedSegment(from: newSegment))
            return
        }
        
        for (i, segment) in self.segments.enumerated().reversed() {
            if segment.isPartOf(newSegment) {
                print("TRACK: Absorbing new segment at index \(i)")
                segment.absorb(newSegment)
                
                // Absorb neighboring segments that overlap
                if i > 0 && segments[i-1].isPartOf(segment) {
                    print("TRACK: Merging into previous segment at index \(i)")
                    segment.absorb(segments.remove(at: i - 1))
                }
                return
            } else if segment.precedes(newSegment) {
                print("TRACK: Inserting new segment at index \(i)")
                segments.insert(DiarizedSegment(from: newSegment), at: i)
                return
            }
        }
        
        updateEmbedding(centroid: &embedding,
                        totalQuality: &embeddingQuality,
                        with: newSegment.embedding,
                        quality: newSegment.qualityScore)
    }
    
    public func remove(segment: DiarizedSegment) {
        segments.remove(segment)
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
