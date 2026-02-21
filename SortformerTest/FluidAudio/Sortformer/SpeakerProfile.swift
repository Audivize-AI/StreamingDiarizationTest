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

    public private(set) var matrix: AHCDistanceMatrix
    public private(set) var dendrogram: Dendrogram
    public private(set) var representatives: [ClusterRepresentative] = []
    public private(set) var slot: Int?
    public var speakerIndex: Int
    public var cannotLink: Set<Int> = []
    
    public var hasOutliers: Bool { !outlierIndices.isEmpty }
    
    public var embeddingCount: Int { matrix.embeddingCount() }
    public var weight: Float { representatives.reduce(0) { $0 + $1.cluster.weight() } }

    private let linkagePolicy: UnsafePointer<LinkagePolicy>
    private var outlierIndices: [Int] = []
    
    init(config: ClusteringConfig, speakerIndex: Int, slot: Int? = nil) {
        self.config = config
        self.linkagePolicy = LinkagePolicy.getPolicy(config.linkagePolicy)
        self.matrix = AHCDistanceMatrix(linkagePolicy: linkagePolicy)
        self.dendrogram = Dendrogram()
        self.slot = slot
        self.speakerIndex = speakerIndex
    }
    
    private init(config: ClusteringConfig, speakerIndex: Int, slot: Int? = nil, matrix: AHCDistanceMatrix, representatives: [ClusterRepresentative], cannotLink: Set<Int>) {
        self.config = config
        self.linkagePolicy = LinkagePolicy.getPolicy(config.linkagePolicy)
        self.matrix = matrix
        self.dendrogram = Dendrogram()
        self.representatives = representatives
        self.slot = slot
        self.speakerIndex = speakerIndex
        self.cannotLink = cannotLink
    }
    
    public func stream(
        newFinalized: [EmbeddingSegment],
        newTentative: [EmbeddingSegment],
        checkOutliers: Bool = false
    ) {
        matrix.stream(finalized: newFinalized, tentative: newTentative)
        
        dendrogram = matrix.dendrogram(false)
        
        let oldRepresentatives = representatives
        updateRepresentatives()
        
        if checkOutliers {
            updateOutliers(oldRepresentatives: oldRepresentatives,
                           linkageThreshold: config.clusteringThreshold)
        }
        
        if !newFinalized.isEmpty {
            compress(keepTentative: true)
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
                (matrix.computeCentroid(with: linkagePolicy, cluster: cluster), cluster)
            }
    }
    
    public func compress(keepTentative: Bool = false) {
        guard matrix.finalizedCount() > 1 ||
                (!keepTentative && matrix.embeddingCount() > 1) else {
            return
        }
        
        let linkageThreshold = config.clusteringThreshold
        
        let clusters = keepTentative
            ? matrix.dendrogram(true).extractClusters(linkageThreshold)
            : dendrogram.extractClusters(linkageThreshold)
        
        // Create a new matrix and reserve capacity (approximate: 2x cluster count as in original intent)
        let newMatrix = AHCDistanceMatrix(linkagePolicy: linkagePolicy)
        newMatrix.reserve(config.maxEmbeddings + matrix.tentativeCount())
        
        // Add compressed cluster centroids
        for cluster in clusters {
            newMatrix.insertEmbedding(
                matrix.computeCentroid(with: linkagePolicy, cluster: cluster),
                tentative: false
            )
        }
        
        if (keepTentative) {
            newMatrix.insertTentative(from: matrix)
        }
        
        // Replace matrix and rebuild dendrogram
        self.matrix = newMatrix
        self.dendrogram = matrix.dendrogram(false)
        self.updateRepresentatives()
    }
    
    public func isolateOutliers(speakerIndex: Int, slot: Int) -> SpeakerProfile {
        let outlierRepresentatives = outlierIndices.map { representatives[$0] }
        for i in outlierIndices.reversed() {
            representatives.remove(at: i)
        }
        
        var outlierIndices: [CLong] = []
        outlierIndices.reserveCapacity(
            outlierRepresentatives.reduce(0) { partialResult, representative in
                partialResult + Int(representative.cluster.count())
            }
        )
        for (_, cluster) in outlierRepresentatives {
            for index in cluster.indices() {
                outlierIndices.append(CLong(truncatingIfNeeded: index))
            }
        }
        let outlierMatrix = matrix.gatherAndPopIndices(outlierIndices)
        
        dendrogram = matrix.dendrogram(false)
        
        return SpeakerProfile(
            config: config,
            speakerIndex: speakerIndex,
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

extension AHCDistanceMatrix {
    @inline(__always)
    func gatherAndPopIndices(_ indices: [CLong]) -> AHCDistanceMatrix {
        indices.withUnsafeBufferPointer { buf in
            gatherAndPopIndices(buf.baseAddress, count: buf.count)
        }
    }

    @inline(__always)
    func stream( finalized: [EmbeddingSegment], tentative: [EmbeddingSegment]) {
        if finalized.isEmpty && tentative.isEmpty {
            stream(withFinalized: nil, finalizedCount: 0, tentative: nil, tentativeCount: 0)
            return
        }

        var finalizedWrappers = Self.buildSegmentWrappers(from: finalized)
        var tentativeWrappers = Self.buildSegmentWrappers(from: tentative)

        finalizedWrappers.withUnsafeMutableBufferPointer { finalizedBuf in
            tentativeWrappers.withUnsafeMutableBufferPointer { tentativeBuf in
                stream(
                    withFinalized: finalizedBuf.baseAddress,
                    finalizedCount: finalizedBuf.count,
                    tentative: tentativeBuf.baseAddress,
                    tentativeCount: tentativeBuf.count
                )
            }
        }
    }

    @inline(__always)
    private static func buildSegmentWrappers(from segments: [EmbeddingSegment]) -> ContiguousArray<EmbeddingSegmentWrapper> {
        var wrappers = ContiguousArray<EmbeddingSegmentWrapper>()
        wrappers.reserveCapacity(segments.count)

        for segment in segments {
            var rawSegmentId: uuid_t = segment.id.uuid
            var wrapper = withUnsafePointer(to: &rawSegmentId) { idPtr in
                EmbeddingSegmentWrapper(
                    idPtr,
                    segment.embeddings.count,
                    segment.segmentIds.count
                )
            }

            for segmentId in segment.segmentIds {
                var rawId: uuid_t = segmentId.uuid
                withUnsafePointer(to: &rawId) { idPtr in
                    wrapper.addSegmentId(idPtr)
                }
            }

            for embedding in segment.embeddings {
                wrapper.addEmbedding(Unmanaged.passUnretained(embedding).toOpaque())
            }

            wrappers.append(wrapper)
        }

        return wrappers
    }
}
