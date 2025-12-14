//
//  EmbeddingUtilities.swift
//  DiarizationTest
//
//  Created by Benjamin Lee on 12/4/25.
//

import Foundation
import Accelerate
import FluidAudio

struct EmbeddingVector {
    var embedding: [Float]
    var quality: Float
    var averageDistance: Float?
    
    init(embedding: [Float], quality: Float, cosineDistance: Float? = nil) {
        self.embedding = unitize(embedding)
        self.quality = quality
        self.averageDistance = cosineDistance
    }
    
    init(from segment: TimedSpeakerSegment) {
        self.embedding = unitize(segment.embedding)
        self.quality = segment.qualityScore
        self.averageDistance = Config.embeddingThreshold / 2
    }
    
    func cosineDistance(to embedding: [Float]) -> Float {
        return Diarization_Test_Visualizer.cosineDistance(fromUnit: self.embedding, to: embedding)
    }
    
    func cosineDistance(to other: EmbeddingVector) -> Float {
        return Diarization_Test_Visualizer.cosineDistance(fromUnit: self.embedding, toUnit: other.embedding)
    }
    
    mutating func normalize() {
        unitize(&embedding)
    }
    
    mutating func update(with newEmbedding: EmbeddingVector) {
        update(withUnitVector: newEmbedding.embedding, quality: newEmbedding.quality)
    }
    
    mutating func update(with segment: TimedSpeakerSegment) {
        update(withUnitVector: unitize(segment.embedding), quality: segment.qualityScore)
    }
    
    mutating func update(withUnitVector unitEmbedding: [Float], quality: Float, distance: Float? = nil) {
//        let cosineDistance = self.cosineDistance(to: unitEmbedding)
//        let distanceAlpha: Float
//        if let averageDistance {
//            let baselineDistance: Float
//            if let distance {
//                baselineDistance = (distance + averageDistance) / 2
//            } else {
//                baselineDistance = averageDistance
//            }
//            distanceAlpha = exp(Config.embeddingDistanceAlpha * -cosineDistance / baselineDistance)
//        } else {
//            distanceAlpha = 1
//        }
        
        let alpha = quality / (self.quality + quality)
//        let alpha = distanceAlpha * qualityAlpha
        
//        if let averageDistance {
//            self.averageDistance! += alpha * (cosineDistance - averageDistance)
//        }
        
        // Update embedding centroid with an EMA
        vDSP.add(
            self.embedding,
            vDSP.multiply(
                alpha,
                vDSP.subtract(
                    unitEmbedding,
                    self.embedding
                )
            ),
            result: &self.embedding
        )
        
        // Unitize updated embedding
        unitize(&self.embedding)
        self.quality += quality
    }
}

func unitize(_ x: inout [Float]) {
    let invNorm: Float = 1.0 / sqrt(vDSP.sumOfSquares(x))
    vDSP.multiply(invNorm, x, result: &x)
}

func unitize(_ x: [Float]) -> [Float] {
    let invNorm: Float = 1.0 / sqrt(vDSP.sumOfSquares(x))
    return vDSP.multiply(invNorm, x)
}

func cosineDistance(from a: [Float], to b: [Float]) -> Float {
    assert(a.count == b.count)
    let normSquaredA = vDSP.sumOfSquares(a)
    let normSquaredB = vDSP.sumOfSquares(b)
    let normalizer = 1 / sqrt(normSquaredA * normSquaredB)
    let dotProduct = vDSP.dot(a, b)
    return 1 - Float(dotProduct) * normalizer
}

func cosineDistance(fromUnit a: [Float], to b: [Float]) -> Float {
    assert(a.count == b.count)
    let normSquaredB = vDSP.sumOfSquares(b)
    let normalizer = 1 / sqrt(normSquaredB)
    let dotProduct = vDSP.dot(a, b)
    return 1 - Float(dotProduct) * normalizer
}

func cosineDistance(fromUnit a: [Float], toUnit b: [Float]) -> Float {
    assert(a.count == b.count)
    return 1 - vDSP.dot(a, b)
}
