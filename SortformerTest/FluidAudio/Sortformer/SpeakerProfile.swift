//
//  SpeakerProfile.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/18/26.
//

import Foundation
import Accelerate

class SpeakerProfile {
    public let config: ClusteringConfig
    public private(set) var finalizedClusters: [SpeakerClusterCentroid] = []
    public private(set) var tentativeClusters: [SpeakerClusterCentroid] = []
    public private(set) var finalizedOutliers: [SpeakerClusterCentroid] = []
    public private(set) var tentativeOutliers: [SpeakerClusterCentroid] = []
    public private(set) var slot: Int?
    public var speakerIndex: Int
    public var cannotLink: Set<Int> = []
    
    public var hasOutliers: Bool {
        !(finalizedOutliers.isEmpty && tentativeOutliers.isEmpty)
    }
    
    public var weight: Float {
        finalizedClusters.reduce(0) { $0 + $1.weight } + tentativeClusters.reduce(0) { $0 + $1.weight }
    }
    
    init(config: ClusteringConfig, speakerIndex: Int, slot: Int? = nil) {
        self.config = config
        self.slot = slot
        self.speakerIndex = speakerIndex
    }
    
    private init(
        config: ClusteringConfig,
        speakerIndex: Int,
        slot: Int? = nil,
        finalizedClusters: [SpeakerClusterCentroid],
        tentativeClusters: [SpeakerClusterCentroid],
        cannotLink: Set<Int>
    ) {
        self.config = config
        self.finalizedClusters = finalizedClusters
        self.tentativeClusters = tentativeClusters
        self.slot = slot
        self.speakerIndex = speakerIndex
        self.cannotLink = cannotLink
    }
    
    public func stream(
        newFinalized: [EmbeddingSegment],
        newTentative: [EmbeddingSegment],
        updateOutliers: Bool = false
    ) {
        func update(clusters: inout [SpeakerClusterCentroid], outliers: inout [SpeakerClusterCentroid], from segments: [EmbeddingSegment], updateOutliers: Bool) {
            for segment in segments {
                guard let centroid = segment.centroid else {
                    continue
                }
                
                if let (cluster, _) = findCluster(for: centroid, in: clusters, maxDistance: config.clusteringThreshold) {
                    cluster.update(with: segment)
                    continue
                }
                
                if !updateOutliers {
                    clusters.append(centroid.deepCopy())
                    continue
                }
                
                if let (cluster, _) = findCluster(for: centroid, in: outliers, maxDistance: config.clusteringThreshold) {
                    cluster.update(with: segment)
                } else {
                    outliers.append(centroid.deepCopy())
                }
            }
        }
        
        // Update finalized clusters
        update(clusters: &finalizedClusters, outliers: &finalizedOutliers, from: newFinalized, updateOutliers: updateOutliers)
        
        if !updateOutliers && !finalizedOutliers.isEmpty {
            finalizedClusters.append(contentsOf: finalizedOutliers)
            finalizedOutliers.removeAll()
        }
        
        // Copy the tentative clusters
        tentativeClusters = finalizedClusters.map{
            $0.deepCopy(keepingId: true)
        }
        
        tentativeOutliers = finalizedOutliers.map{
            $0.deepCopy(keepingId: true)
        }
        
        // Update tentative clusters
        update(clusters: &tentativeClusters, outliers: &tentativeOutliers, from: newTentative, updateOutliers: updateOutliers)
    }
    
    public func findCluster(for embedding: SpeakerEmbedding, in clusters: [SpeakerClusterCentroid], maxDistance: Float = .infinity) -> (cluster: SpeakerClusterCentroid, distance: Float)? {
        if clusters.isEmpty { return nil }
        
        var bestDistance = maxDistance.nextUp
        var bestCluster: SpeakerClusterCentroid? = nil
        
        for cluster in clusters {
            let distance = cluster.cosineDistance(to: embedding)
            if distance < bestDistance {
                bestDistance = distance
                bestCluster = cluster
            }
        }
        
        guard let bestCluster else { return nil }
        return (bestCluster, bestDistance)
    }
    
    @inline(__always)
    public func findCluster(for centroid: SpeakerClusterCentroid, in clusters: [SpeakerClusterCentroid], maxDistance: Float = .infinity) -> (cluster: SpeakerClusterCentroid, distance: Float)? {
        return findCluster(for: centroid.embedding, in: clusters, maxDistance: maxDistance)
    }
    
    /// Get the chamfer distance between two SpeakerProfiles
    public func distance(to other: SpeakerProfile, useTentative: Bool = true) -> Float {
        let clustersA = useTentative ? self.tentativeClusters : self.finalizedClusters
        let clustersB = useTentative ? other.tentativeClusters : other.finalizedClusters
        
        var bestB: [Float] = Array(repeating: .infinity, count: clustersB.count)
        var sumA: Float = 0
        for embA in clustersA {
            var minA = Float.infinity
            for (i, embB) in clustersB.enumerated() {
                let dist: Float = embA.cosineDistance(to: embB)
                if dist < minA { minA = dist }
                if dist < bestB[i] { bestB[i] = dist }
            }
            sumA += minA
        }
        
        let sumB: Float = bestB.reduce(0, +)
        return (sumA / Float(clustersA.count) + sumB / Float(clustersB.count)) / 2
    }
    
    public func fromOutliers(slot: Int, speakerIndex: Int) -> SpeakerProfile {
        let result = SpeakerProfile(
            config: self.config,
            speakerIndex: speakerIndex,
            slot: slot,
            finalizedClusters: self.finalizedOutliers,
            tentativeClusters: self.tentativeOutliers,
            cannotLink: self.cannotLink
        )
        result.cannotLink.insert(self.speakerIndex)
        
        self.finalizedOutliers.removeAll()
        self.tentativeOutliers.removeAll()
        
        return result
    }
    
    public func absorb(_ other: SpeakerProfile) {
        self.finalizedClusters.append(contentsOf: other.finalizedClusters)
        self.tentativeClusters.append(contentsOf: other.tentativeClusters)
        
        let combinedFinalizedOutliers = self.finalizedOutliers + other.finalizedOutliers
        let combinedTentativeOutliers = self.finalizedOutliers + other.finalizedOutliers
        
        self.finalizedOutliers.removeAll(keepingCapacity: true)
        self.tentativeOutliers.removeAll(keepingCapacity: true)
        
        for outlier in combinedFinalizedOutliers {
            if let (cluster, _) = findCluster(for: outlier, in: finalizedClusters, maxDistance: config.clusteringThreshold) {
                cluster.update(with: outlier)
            } else {
                finalizedOutliers.append(outlier)
            }
        }
        
        for outlier in combinedTentativeOutliers {
            if let (cluster, _) = findCluster(for: outlier, in: tentativeClusters, maxDistance: config.clusteringThreshold) {
                cluster.update(with: outlier)
            } else {
                tentativeOutliers.append(outlier)
            }
        }
    }
}

public class SpeakerClusterCentroid: Identifiable {
    public var id: UUID { embedding.id }
    public let embedding: SpeakerEmbedding
    public var weight: Float
    public var segmentIds: [UUID]
    public var isFinalized: Bool
    
    public var buffer: UnsafeBufferPointer<Float> { embedding.bufferView }
    public var baseAddress: UnsafePointer<Float>? { embedding.baseAddress }
    
    public init(id: UUID = UUID(), segmentIds: [UUID] = [], weight: Float, isFinalized: Bool = false) {
        // Create a deep copy of the embedding
        self.embedding = SpeakerEmbedding(id: id, startFrame: 0, endFrame: 0)
        self.segmentIds = segmentIds
        self.isFinalized = isFinalized
        self.weight = weight
    }
    
    public init(id: UUID = UUID(), embedding: SpeakerEmbedding, segmentIds: [UUID] = [], weight: Float, isFinalized: Bool = false) {
        // Create a deep copy of the embedding
        self.embedding = SpeakerEmbedding(id: id, embedding: embedding.bufferView, startFrame: embedding.startFrame, endFrame: embedding.endFrame)
        self.segmentIds = segmentIds
        self.isFinalized = isFinalized
        self.weight = weight
    }
    
    public func update(with centroid: SpeakerClusterCentroid) {
        self.weight += centroid.weight
        var alpha = centroid.weight / self.weight
        
        self.withUnsafeMutableBufferPointer { muPtr in
            // µ_n = µ_{n-1} + w/W_n (x - µ_{n-1})
            vDSP_vintb(
                muPtr.baseAddress!, 1,
                centroid.baseAddress!, 1,
                &alpha,
                muPtr.baseAddress!, 1,
                vDSP_Length(muPtr.count)
            )
        }
        
        self.segmentIds.append(contentsOf: centroid.segmentIds)
    }
    
    /// Update the centroid in place
    @inline(__always)
    public func update(with segment: EmbeddingSegment) {
        guard let centroid = segment.centroid else {
            return
        }
        update(with: centroid)
    }
    
    /// Make an updated copy of the centroid
    public func updated(with centroid: SpeakerClusterCentroid, keepingId: Bool = false) -> SpeakerClusterCentroid {
        let result = SpeakerClusterCentroid(
            id: keepingId ? self.id : UUID(),
            embedding: self.embedding,
            segmentIds: self.segmentIds + centroid.segmentIds,
            weight: self.weight + centroid.weight,
            isFinalized: self.isFinalized && centroid.isFinalized
        )
        
        var alpha = centroid.weight / result.weight
        
        result.withUnsafeMutableBufferPointer { muPtr in
            // µ_n = µ_{n-1} + w/W_n (x - µ_{n-1})
            vDSP_vintb(
                muPtr.baseAddress!, 1,
                centroid.baseAddress!, 1,
                &alpha,
                muPtr.baseAddress!, 1,
                vDSP_Length(muPtr.count)
            )
        }
        
        return result
    }
    
    /// Make an updated copy of the centroid
    public func updated(with segment: EmbeddingSegment, keepingId: Bool = false) -> SpeakerClusterCentroid {
        guard let centroid = segment.centroid else {
            return self
        }
        return self.updated(with: centroid)
    }
    
    @inline(__always)
    public func deepCopy(keepingId: Bool = false) -> SpeakerClusterCentroid {
        SpeakerClusterCentroid(
            id: keepingId ? self.id : UUID(),
            embedding: self.embedding,
            segmentIds: self.segmentIds,
            weight: self.weight,
            isFinalized: self.isFinalized
        )
    }
    
    /// Get the cosine distance to another centroid
    /// - Parameter centroid: Another cluster centroid
    /// - Returns: The cosine distance between this centroid to the speaker embedding
    @inline(__always)
    public func cosineDistance(to centroid: SpeakerClusterCentroid) -> Float {
        return self.embedding.cosineDistance(to: centroid.embedding)
    }
    
    /// Get the cosine distance to a speaker embedding
    /// - Parameter embedding: A singleton speaker embedding
    /// - Returns: The cosine distance between this centroid to the speaker embedding
    @inline(__always)
    public func cosineDistance(to embedding: SpeakerEmbedding) -> Float {
        return self.embedding.cosineDistance(to: embedding)
    }
    
    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        return try embedding.withUnsafeBufferPointer(body)
    }
    
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        return try embedding.withUnsafeMutableBufferPointer(body)
    }
}
