//
//  EmbeddingUtilities.swift
//  DiarizationTest
//
//  Created by Benjamin Lee on 12/4/25.
//

import Foundation
import Accelerate


func updateEmbedding(centroid: inout [Float], totalQuality: inout Float, with embedding: [Float], quality: Float) {
    // Unitize incoming embedding vector
    let newUnitEmbedding = unitize(embedding)

    // Update embedding centroid with an EMA
    let alpha = quality / (totalQuality + quality)
    vDSP.add(
        centroid,
        vDSP.multiply(
            alpha,
            vDSP.subtract(
                newUnitEmbedding,
                centroid
            )
        ),
        result: &centroid
    )
    
    // Unitize updated embedding
    unitize(&centroid)
    
    totalQuality += quality
}

func unitize(_ x: inout [Float]) {
    let invNorm: Float = 1.0 / sqrt(vDSP.sumOfSquares(x))
    vDSP.multiply(invNorm, x, result: &x)
}

func unitize(_ x: [Float]) -> [Float] {
    let invNorm: Float = 1.0 / sqrt(vDSP.sumOfSquares(x))
    return vDSP.multiply(invNorm, x)
}
