//
//  SpeakerProfile.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/18/26.
//

import Foundation
import Accelerate
import OrderedCollections

public class SpeakerProfile {
    public let config: ClusteringConfig
    public private(set) var finalizedClusters: [SpeakerClusterCentroid] = []
    public private(set) var tentativeClusters: [SpeakerClusterCentroid] = []
    public private(set) var finalizedOutliers: [SpeakerClusterCentroid] = []
    public private(set) var tentativeOutliers: [SpeakerClusterCentroid] = []
    public private(set) var finalizedSegments: OrderedSet<SpeakerSegment> = []
    public private(set) var tentativeSegments: OrderedSet<SpeakerSegment> = []
    
    /// The ID of the speaker whose outliers formed this speaker profile
    public private(set) var parentSpeakerId: Int? = nil
    
    /// Speaker ID
    public var speakerId: Int
    
    /// Set of speaker IDs with which this profile cannot link
    public var cannotLink: Set<Int> = []
    
    /// Whether this speaker has any clusters
    public var hasClusters: Bool {
        !(finalizedClusters.isEmpty && tentativeClusters.isEmpty)
    }
    
    /// Whether this speaker has any segments
    public var hasSegments: Bool {
        !(finalizedSegments.isEmpty && tentativeSegments.isEmpty)
    }
    
    /// Whether this speaker has any outliers
    public var hasOutliers: Bool {
        !(finalizedOutliers.isEmpty && tentativeOutliers.isEmpty)
    }
    
    /// The combined weight of all confirmed clusters
    public var finalizedWeight: Float {
        finalizedClusters.reduce(0) { $0 + $1.weight }
    }
    
    /// The combined weight of all tentative clusters
    public var tentativeWeight: Float {
        tentativeClusters.reduce(0) { $0 + $1.weight }
    }
    
    public var isFinalized: Bool {
        tentativeClusters.isEmpty && tentativeOutliers.isEmpty && tentativeSegments.isEmpty
    }
    
    // MARK: - Init
    
    init(config: ClusteringConfig, speakerIndex: Int) {
        self.config = config
        self.speakerId = speakerIndex
    }
    
    private init(
        config: ClusteringConfig,
        speakerIndex: Int,
        finalizedSegments: OrderedSet<SpeakerSegment>,
        tentativeSegments: OrderedSet<SpeakerSegment>,
        finalizedClusters: [SpeakerClusterCentroid],
        tentativeClusters: [SpeakerClusterCentroid],
        cannotLink: Set<Int>
    ) {
        self.config = config
        self.finalizedClusters = finalizedClusters
        self.tentativeClusters = tentativeClusters
        self.finalizedSegments = finalizedSegments
        self.tentativeSegments = tentativeSegments
        self.speakerId = speakerIndex
        self.cannotLink = cannotLink
    }
    
    // MARK: - Segment Updates
    
    @inline(__always)
    public func appendFinalizedSegment(_ segment: SpeakerSegment) {
        self.finalizedSegments.updateOrAppend(segment)
    }
    
    @inline(__always)
    public func appendTentativeSegment(_ segment: SpeakerSegment) {
        self.tentativeSegments.updateOrAppend(segment)
    }
    
    @inline(__always)
    @discardableResult
    public func appendSegment(_ segment: SpeakerSegment) -> SpeakerSegment? {
        return segment.isFinalized
            ? finalizedSegments.updateOrAppend(segment)
            : tentativeSegments.updateOrAppend(segment)
    }
    
    @inline(__always)
    public func popFinalizedSegment() -> SpeakerSegment? {
        guard !finalizedSegments.isEmpty else { return nil }
        return finalizedSegments.removeLast()
    }
    
    @inline(__always)
    public func popTentativeSegment() -> SpeakerSegment? {
        guard !tentativeSegments.isEmpty else { return nil }
        return tentativeSegments.removeLast()
    }
    
    @inline(__always)
    public func popSegment(finalized: Bool) -> SpeakerSegment? {
        return finalized ? popFinalizedSegment() : popTentativeSegment()
    }
    
    @inline(__always)
    public func clearTentativeSegments() {
        return tentativeSegments.removeAll()
    }
    
    // MARK: - Embedding Updates
    
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
                
                // Find the best match
                if let (cluster, _) = Self.findCluster(for: centroid, in: clusters, maxDistance: config.clusteringThreshold) {
                    cluster.update(with: segment)
                    continue
                }
                
                // Create a new cluster if we aren't checking for outliers
                if !updateOutliers {
                    clusters.append(centroid.deepCopy())
                    continue
                }
                
                // Create an outlier cluster
                if let (cluster, _) = Self.findCluster(for: centroid, in: outliers, maxDistance: config.clusteringThreshold) {
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
    
    public static func findCluster(for embedding: SpeakerEmbedding, in clusters: [SpeakerClusterCentroid], maxDistance: Float = .infinity) -> (cluster: SpeakerClusterCentroid, distance: Float)? {
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
    public static func findCluster(for centroid: SpeakerClusterCentroid, in clusters: [SpeakerClusterCentroid], maxDistance: Float = .infinity) -> (cluster: SpeakerClusterCentroid, distance: Float)? {
        return findCluster(for: centroid.embedding, in: clusters, maxDistance: maxDistance)
    }
    
    /// Get the chamfer distance between two SpeakerProfiles
    /// - Parameters:
    ///   - other: Another speaker profile
    ///   - useTentative: Whether to use tentative clusters in the distance calculation (defaults to `true`)
    /// - Returns: The chamfer distance to the other speaker profile
    public func distance(to other: SpeakerProfile, useTentative: Bool = true) -> Float {
        guard !other.cannotLink.contains(self.speakerId),
              !self.cannotLink.contains(other.speakerId) else {
            return .infinity
        }
        
        let clustersA = useTentative ? self.tentativeClusters : self.finalizedClusters
        let clustersB = useTentative ? other.tentativeClusters : other.finalizedClusters
        
        var bestB: [(dist: Float, weight: Float)] = Array(repeating: (.infinity, 0), count: clustersB.count)
        var sumA: Float = 0
        var sumWeightsA: Float = 0
        
        for embA in clustersA {
            var minDistA = Float.infinity
            var bestWeightA: Float = 0
            for (i, embB) in clustersB.enumerated() {
                let dist: Float = embA.cosineDistance(to: embB)
                if dist < minDistA {
                    minDistA = dist
                    bestWeightA = embB.weight
                }
                if dist < bestB[i].dist {
                    bestB[i] = (dist, embB.weight * embA.weight)
                }
            }
            
            let weight = embA.weight * bestWeightA
            
            sumWeightsA += weight
            sumA += weight * minDistA
        }
        
        let (sumB, sumWeightsB) = bestB.reduce((dist: 0 as Float, weight: 0 as Float)) {
            ($0.dist + $1.dist * $1.weight, $0.weight + $1.weight)
        }
        
        return (sumA / sumWeightsA + sumB / sumWeightsB) / 2
    }
    
    public func takeOutliers(speakerId: Int, cannotLink: Set<Int> = []) -> SpeakerProfile {
        // Collect segments
        var outlierSegments = self.tentativeOutliers.flatMap(\.segments)
        let numTentativeOutliers = outlierSegments.partition(by: \.isFinalized)
        outlierSegments[0..<numTentativeOutliers].sort()
        outlierSegments[numTentativeOutliers...].sort()
        
        var outlierTentativeSegments = OrderedSet(outlierSegments.prefix(numTentativeOutliers))
        var outlierFinalizedSegments = OrderedSet(outlierSegments.suffix(from: numTentativeOutliers))
        
        self.tentativeSegments.subtract(outlierTentativeSegments)
        self.finalizedSegments.subtract(outlierFinalizedSegments)
        
        let result = SpeakerProfile(
            config: self.config,
            speakerIndex: speakerId,
            finalizedSegments: outlierFinalizedSegments,
            tentativeSegments: outlierTentativeSegments,
            finalizedClusters: self.finalizedOutliers,
            tentativeClusters: self.tentativeOutliers,
            cannotLink: cannotLink
        )
        
        self.finalizedOutliers.removeAll()
        self.tentativeOutliers.removeAll()
        
        return result
    }
    
    public func finalize() {
        if !tentativeClusters.isEmpty {
            self.finalizedClusters = self.tentativeClusters
            self.tentativeClusters.removeAll()
        }
        if !finalizedOutliers.isEmpty {
            self.finalizedOutliers = self.tentativeOutliers
            self.tentativeOutliers.removeAll()
        }
        if !tentativeSegments.isEmpty {
            self.finalizedSegments.append(contentsOf: self.tentativeSegments)
            self.tentativeSegments.removeAll()
        }
    }
    
    public func absorbAndFinalize(_ other: SpeakerProfile) {
        self.finalize()
        other.finalize()
        
        typealias ClusterAssignment = (src: SpeakerClusterCentroid, dst: SpeakerClusterCentroid)

        // Update segments
        self.finalizedSegments.append(contentsOf: other.finalizedSegments)
        self.tentativeSegments.append(contentsOf: other.tentativeSegments)
        finalizedSegments.sort()
        tentativeSegments.sort()
        
        // Merge similar clusters
        
        var finalizedMatches: [UUID : SpeakerClusterCentroid] = [:]
        var tentativeMatches: [UUID : SpeakerClusterCentroid] = [:]
        
        finalizedMatches.reserveCapacity(self.finalizedClusters.count)
        tentativeMatches.reserveCapacity(self.tentativeClusters.count)
        
        for cluster in other.finalizedClusters {
            finalizedMatches[cluster.id] = Self.findCluster(
                for: cluster,
                in: self.finalizedClusters,
                maxDistance: config.clusteringThreshold
            )?.cluster
        }
        
        // Update/assign outliers
        
        var finalizedClusterAssignments: [ClusterAssignment] = []
        var tentativeClusterAssignments: [ClusterAssignment] = []
        finalizedClusterAssignments.reserveCapacity(self.finalizedOutliers.count + other.finalizedOutliers.count)
        finalizedClusterAssignments.reserveCapacity(self.tentativeOutliers.count + other.tentativeOutliers.count)

        var newFinalizedOutliers: [SpeakerClusterCentroid] = []
        var newTentativeOutliers: [SpeakerClusterCentroid] = []
        newFinalizedOutliers.reserveCapacity(finalizedClusterAssignments.capacity)
        newTentativeOutliers.reserveCapacity(tentativeClusterAssignments.capacity)
        
        func updateAssignmentsAndOutliers(
            outliers: [SpeakerClusterCentroid],
            clusters: [SpeakerClusterCentroid],
            assignments: inout [ClusterAssignment],
            remainingOutliers: inout [SpeakerClusterCentroid]
        ) {
            for outlier in outliers {
                if let (cluster, _) = Self.findCluster(
                    for: outlier,
                    in: clusters,
                    maxDistance: config.clusteringThreshold
                ) {
                    assignments.append((outlier, cluster))
                } else {
                    remainingOutliers.append(outlier)
                }
            }
        }
        
        updateAssignmentsAndOutliers(outliers: self.finalizedOutliers,
                                     clusters: other.finalizedClusters,
                                     assignments: &finalizedClusterAssignments,
                                     remainingOutliers: &newFinalizedOutliers)
        updateAssignmentsAndOutliers(outliers: other.finalizedOutliers,
                                     clusters: self.finalizedClusters,
                                     assignments: &finalizedClusterAssignments,
                                     remainingOutliers: &newFinalizedOutliers)
        updateAssignmentsAndOutliers(outliers: self.tentativeOutliers,
                                     clusters: other.tentativeClusters,
                                     assignments: &tentativeClusterAssignments,
                                     remainingOutliers: &newTentativeOutliers)
        updateAssignmentsAndOutliers(outliers: other.tentativeOutliers,
                                     clusters: self.tentativeClusters,
                                     assignments: &tentativeClusterAssignments,
                                     remainingOutliers: &newTentativeOutliers)
        
        // Update clusters from outlier assignments
        for (outlier, cluster) in finalizedClusterAssignments {
            cluster.update(with: outlier)
        }
        
        for (outlier, cluster) in tentativeClusterAssignments {
            cluster.update(with: outlier)
        }
        
        // Combine clusters
        self.finalizedClusters.append(contentsOf: other.finalizedClusters)
        self.tentativeClusters.append(contentsOf: other.tentativeClusters)
        
        // Combine bullshits
        self.cannotLink.formUnion(other.cannotLink)
        self.cannotLink.remove(speakerId)
    }
}

public class SpeakerClusterCentroid: Identifiable {
    public var id: UUID { embedding.id }
    public let embedding: SpeakerEmbedding
    public var weight: Float
    public var segments: [SpeakerSegment]
    public var isFinalized: Bool
    
    public var buffer: UnsafeBufferPointer<Float> { embedding.bufferView }
    public var baseAddress: UnsafePointer<Float>? { embedding.baseAddress }
    
    public init(id: UUID = UUID(), segments: [SpeakerSegment] = [], weight: Float, isFinalized: Bool = false) {
        // Create a deep copy of the embedding
        self.embedding = SpeakerEmbedding(id: id, startFrame: 0, endFrame: 0)
        self.segments = segments
        self.isFinalized = isFinalized
        self.weight = weight
    }
    
    public init(id: UUID = UUID(), embedding: SpeakerEmbedding, segments: [SpeakerSegment] = [], weight: Float, isFinalized: Bool = false) {
        // Create a deep copy of the embedding
        self.embedding = SpeakerEmbedding(id: id, embedding: embedding.bufferView, startFrame: embedding.startFrame, endFrame: embedding.endFrame)
        self.segments = segments
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
        
        self.segments.append(contentsOf: centroid.segments)
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
            segments: self.segments + centroid.segments,
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
            segments: self.segments,
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
