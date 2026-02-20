//
//  SpeakerProfile.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/18/26.
//

import Foundation
import AHCClustering



class SpeakerProfile {
    typealias ClusterRepresentative = (centroid: SpeakerEmbeddingWrapper, cluster: Cluster)
    
    public let config: ClusteringConfig

    public private(set) var matrix: EmbeddingDistanceMatrix
    public private(set) var dendrogram: Dendrogram
    public private(set) var representatives: [ClusterRepresentative] = []
    public private(set) var slot: Int?
    public var cannotLink: Set<Int> = []

    private let linkagePolicy: UnsafePointer<LinkagePolicy>
    private var outlierIndices: [Int] = []
    
    init(config: ClusteringConfig, slot: Int? = nil) {
        self.config = config
        self.linkagePolicy = LinkagePolicy.getPolicy(config.linkagePolicy)
        self.matrix = EmbeddingDistanceMatrix(linkagePolicy)
        self.dendrogram = Dendrogram()
        self.slot = slot
    }
    
    private init(config: ClusteringConfig, slot: Int? = nil, matrix: EmbeddingDistanceMatrix, representatives: [ClusterRepresentative], cannotLink: Set<Int>) {
        self.config = config
        self.linkagePolicy = LinkagePolicy.getPolicy(config.linkagePolicy)
        self.matrix = EmbeddingDistanceMatrix(linkagePolicy)
        self.dendrogram = Dendrogram()
        self.slot = slot
        self.cannotLink = cannotLink
    }
    
    public func stream(
        newFinalized: [EmbeddingSegment],
        newTentative: [EmbeddingSegment],
        checkOutliers: Bool = false
    ) {
        @inline(__always)
        func buildWrappers(from segments: [EmbeddingSegment]) -> [EmbeddingSegmentWrapper] {
            return segments.map { segment in
                let segmentId = segment.id
                
                var wrapper = withUnsafeBytes(of: segmentId) { idBytes in
                    EmbeddingSegmentWrapper(
                        idBytes.baseAddress!,
                        segment.embeddings.count,
                        segment.segmentIds.count
                    )
                }
                
                let ids: [uuid_t] = segment.segmentIds.map { $0.uuid }
                ids.withUnsafeBufferPointer { buf in
                    guard let base = buf.baseAddress else { return }
                    for i in 0..<buf.count {
                        wrapper.addSegmentId(base.advanced(by: i)) // uuid_t*
                    }
                }
                
                for embedding in segment.embeddings {
                    wrapper.addEmbedding(Unmanaged.passUnretained(embedding).toOpaque())
                }
                
                return wrapper
            }
        }
        
        var finalizedWrappers = buildWrappers(from: newFinalized)
        var tentativeWrappers = buildWrappers(from: newTentative)
        
        finalizedWrappers.withUnsafeMutableBufferPointer { finalizedBuf in
            tentativeWrappers.withUnsafeMutableBufferPointer { tentativeBuf in
                matrix.stream(.init(finalizedBuf), .init(tentativeBuf))
            }
        }
        
        dendrogram = matrix.dendrogram(false)
        
        let oldRepresentatives = representatives
        updateRepresentatives()
        
        if checkOutliers {
            updateOutliers(oldRepresentatives: oldRepresentatives,
                           linkageThreshold: config.clusteringThreshold)
        }
        
        if matrix.finalizedCount() > config.maxEmbeddings {
            compress(maxSize: config.minEmbeddings,
                     linkageThreshold: config.minSeparation)
        }
    }
    
    public func distance(to other: SpeakerProfile) -> Float {
        var bestB: [Float] = Array(repeating: .infinity, count: other.representatives.count)
        var sumA: Float = 0
        for (embA, _) in representatives {
            var minA = Float.infinity
            for (i, (embB, _)) in other.representatives.enumerated() {
                let dist: Float = LinkagePolicy.distance(linkagePolicy, embA, embB)
                if dist < minA { minA = dist }
                if dist < bestB[i] { bestB[i] = dist }
            }
            sumA += minA
        }
        
        let sumB: Float = bestB.reduce(0, +)
        return (sumA / Float(representatives.count) + sumB / Float(other.representatives.count)) / 2
    }
    
    private func updateOutliers(oldRepresentatives: [ClusterRepresentative], linkageThreshold: Float) {
        outlierIndices = representatives.indices.filter { i in
            let embA = representatives[i].centroid
            return oldRepresentatives.allSatisfy { (embB, _) in
                LinkagePolicy.distance(linkagePolicy, embA, embB) > linkageThreshold
            }
        }
    }
    
    private func updateRepresentatives() {
        representatives = dendrogram
            .extractClusters(config.clusteringThreshold, config.maxRepresentatives)
            .map { cluster in
                (LinkagePolicy.computeCentroid(linkagePolicy, matrix, cluster), cluster)
            }
    }
    
    public func compress(maxSize: Int, linkageThreshold: Float, keepTentative: Bool = false) {
        let clusters = keepTentative
            ? matrix.dendrogram(true).extractClusters(linkageThreshold, maxSize)
            : dendrogram.extractClusters(linkageThreshold, maxSize)
        
        // Create a new matrix and reserve capacity (approximate: 2x cluster count as in original intent)
        var newMatrix = EmbeddingDistanceMatrix(linkagePolicy)
        newMatrix.reserve(config.maxEmbeddings)
        
        // Add compressed cluster centroids
        for cluster in clusters {
            newMatrix.insert(LinkagePolicy.computeCentroid(linkagePolicy, matrix, cluster))
        }
        
        if (keepTentative) {
            // Add the tentative embeddings back
            let indices = matrix.tentativeIndices()
            for i in indices {
                newMatrix.insert(matrix.embedding(i), true)
            }
        }
        
        // Replace matrix and rebuild dendrogram
        self.matrix = newMatrix
        self.dendrogram = self.matrix.dendrogram()
        self.updateRepresentatives()
    }
    
    public func isolateOutliers(slot: Int) -> SpeakerProfile {
        let outlierRepresentatives = outlierIndices.map { representatives[$0] }
        for i in outlierIndices.reversed() {
            representatives.remove(at: i)
        }
        
        let outlierMatrix = outlierRepresentatives
            .flatMap { $0.cluster.indices() }
            .withUnsafeBufferPointer { matrix.gatherAndPop(.init($0)) }
        
        dendrogram = matrix.dendrogram(false)
        
        return SpeakerProfile(
            config: config,
            slot: slot,
            matrix: outlierMatrix,
            representatives: outlierRepresentatives,
            cannotLink: cannotLink
        )
    }
    
    public func absorb(_ other: SpeakerProfile) {
        matrix.absorb(other.matrix)
        dendrogram = matrix.dendrogram(false)
        updateRepresentatives()
    }
}
