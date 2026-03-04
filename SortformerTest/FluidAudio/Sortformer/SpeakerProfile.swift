//
//  SpeakerProfile.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 2/18/26.
//

import Foundation
import Accelerate
import OrderedCollections
import AHCClustering

public class SpeakerProfile: Hashable {
    public let id = UUID()
    public let config: ClusteringConfig
    public private(set) var finalizedClusters: [SpeakerClusterCentroid] = []
    public private(set) var tentativeClusters: [SpeakerClusterCentroid] = []
    public private(set) var finalizedSegments: [TimelineSegment] = []
    public private(set) var tentativeSegments: [TimelineSegment] = []
    
    /// All speaker segments
    public var segments: [SpeakerSegment] {
        (finalizedSegments + tentativeSegments).map(\.speakerSegment)
    }
    
    /// Set of speaker IDs with which this profile cannot link
    public private(set) var cannotLink: Set<Int> = []
    
    /// The speaker's original ID, assigned at creation. Used as a fallback
    /// when no inactive speaker profile matches.
    public let defaultSpeakerId: Int
    
    /// Speaker ID
    public var speakerId: Int {
        didSet {
            guard oldValue != speakerId else { return }
            for i in finalizedSegments.indices {
                finalizedSegments[i].speakerId = speakerId
            }
            for i in tentativeSegments.indices {
                tentativeSegments[i].speakerId = speakerId
            }
        }
    }
    
    /// Whether this speaker is finalized
    public private(set) var isFinalized: Bool = false
    
    /// Whether this speaker has any outliers
    public private(set) var hasOutliers: Bool = false
    
    /// Whether this speaker has any clusters
    public var hasClusters: Bool {
        !(finalizedClusters.isEmpty && tentativeClusters.isEmpty)
    }
    
    /// Whether this speaker has any segments
    public var hasSegments: Bool {
        !(finalizedSegments.isEmpty && tentativeSegments.isEmpty)
    }
    
    /// The combined weight of all confirmed clusters
    public var finalizedWeight: Float {
        finalizedClusters.reduce(0) { $0 + $1.weight }
    }
    
    /// The combined weight of all tentative clusters
    public var tentativeWeight: Float {
        tentativeClusters.reduce(0) { $0 + $1.weight }
    }
    
    public var finalizedSegmentCount: Int { finalizedSegments.count }
    public var tentativeSegmentCount: Int { tentativeSegments.count }
    public var finalizedClusterCount: Int { finalizedClusters.count }
    public var tentativeClusterCount: Int { tentativeClusters.count }
    
    public var isDroppable: Bool {
        !hasOutliers
    }
    
    var lastActiveFrame: Int {
        return (tentativeSegments.last?.endFrame ??
                finalizedSegments.last?.endFrame ?? .min)
    }

    var firstActiveFrame: Int {
        return (finalizedSegments.first?.startFrame ??
                tentativeSegments.first?.startFrame ?? .max)
    }
    
    // MARK: - Init
    
    init(config: ClusteringConfig, speakerId: Int) {
        self.config = config
        self.defaultSpeakerId = speakerId
        self.speakerId = speakerId
    }
    
    convenience init(config: ClusteringConfig, speakerIndex: Int) {
        self.init(config: config, speakerId: speakerIndex)
    }
    
    private init(
        config: ClusteringConfig,
        speakerId: Int,
        finalizedSegments: [TimelineSegment],
        tentativeSegments: [TimelineSegment],
        finalizedClusters: [SpeakerClusterCentroid],
        tentativeClusters: [SpeakerClusterCentroid],
        cannotLink: Set<Int>
    ) {
        self.config = config
        self.defaultSpeakerId = speakerId
        self.finalizedClusters = finalizedClusters
        self.tentativeClusters = tentativeClusters
        self.finalizedSegments = finalizedSegments
        self.tentativeSegments = tentativeSegments
        self.speakerId = speakerId
        self.cannotLink = cannotLink
    }
    
    // MARK: - Segment Updates
    
    @inline(__always)
    public func appendFinalizedSegment(_ segment: TimelineSegment) {
        let newSegment = segment.reassigned(toSpeaker: speakerId)
        finalizedSegments.append(newSegment)
    }
    
    @inline(__always)
    public func appendTentativeSegment(_ segment: TimelineSegment) {
        let newSegment = segment.reassigned(toSpeaker: speakerId)
        tentativeSegments.append(newSegment)
    }
    
    @inline(__always)
    public func appendSegment(_ segment: TimelineSegment) {
        if segment.isFinalized {
            appendFinalizedSegment(segment)
        } else {
            appendTentativeSegment(segment)
        }
    }
    
    @inline(__always) @discardableResult
    public func popFinalizedSegment() -> TimelineSegment? {
        guard !finalizedSegments.isEmpty else { return nil }
        return finalizedSegments.removeLast()
    }
    
    @inline(__always) @discardableResult
    public func popTentativeSegment() -> TimelineSegment? {
        return tentativeSegments.popLast()
    }
    
    @inline(__always) @discardableResult
    public func popSegment(finalized: Bool) -> TimelineSegment? {
        return finalized ? popFinalizedSegment() : popTentativeSegment()
    }
    
    @inline(__always)
    func clearTentativeSegments() {
        return tentativeSegments.removeAll(keepingCapacity: true)
    }
    
    // MARK: - Embedding Updates
    
    public func stream(
        newFinalized: [EmbeddingSegment],
        newTentative: [EmbeddingSegment],
        updateOutliers: Bool = false
    ) {
        hasOutliers = false
        let updateOutliers = updateOutliers && hasClusters
        
        func update(
            clusters: inout [SpeakerClusterCentroid],
            with segments: [EmbeddingSegment],
            oldClusters: borrowing [SpeakerClusterCentroid]
        ) {
            for segment in segments {
                guard let centroid = segment.centroid else { continue }
                
                // Find the best match
                if let (cluster, distance) = findCluster(for: centroid, in: clusters) {
                    cluster.update(
                        with: centroid,
                        updateVector: distance <= config.updateThreshold
                    )
                } else if !updateOutliers || hasMatchingCluster(for: centroid, in: oldClusters) {
                    clusters.append(centroid.deepCopy())
                } else {
                    hasOutliers = true
                    debugPrint("There was an outlier in speaker \(speakerId).")
                }
            }
        }
        
        // Update finalized clusters
        var oldClusters = finalizedClusters.isEmpty ? tentativeClusters : []

        update(clusters: &finalizedClusters,
               with: newFinalized,
               oldClusters: oldClusters)
        
        // Update tentative clusters
        if finalizedClusters.isEmpty {
            oldClusters = tentativeClusters
            tentativeClusters.removeAll(keepingCapacity: true)
        } else {
            oldClusters = finalizedClusters
            tentativeClusters = finalizedClusters.map {
                $0.deepCopy(keepingId: true)
            }
        }
        
        update(clusters: &tentativeClusters,
               with: newTentative,
               oldClusters: oldClusters)
        
        self.isFinalized = false
    }
    
    public func findCluster<E>(
        for embedding: E,
        in clusters: [SpeakerClusterCentroid],
        maxDistance: Float? = nil
    ) -> (cluster: SpeakerClusterCentroid, distance: Float)?
    where E: EmbeddingVector {
        guard !clusters.isEmpty else { return nil }
        
        var bestDistance = (maxDistance ?? config.clusteringThreshold).nextUp
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
    public func hasMatchingCluster<E>(
        for embedding: E,
        in clusters: [SpeakerClusterCentroid],
        maxDistance: Float? = nil
    ) -> Bool where E: EmbeddingVector {
        guard !clusters.isEmpty else { return false }
        let threshold = maxDistance ?? config.clusteringThreshold
        return clusters.contains {
            $0.cosineDistance(to: embedding) <= threshold
        }
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
        
        let clustersA = (useTentative && !self.isFinalized)
            ? self.tentativeClusters : self.finalizedClusters
        let clustersB = (useTentative && !other.isFinalized)
            ? other.tentativeClusters : other.finalizedClusters
        
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

    /// Finalize the speaker
    public func finalize() {
        guard !isFinalized else { return }
        
        if !tentativeClusters.isEmpty {
            finalizedClusters = tentativeClusters
            tentativeClusters.removeAll()
            finalizedClusters.sort { $0.weight > $1.weight }
            debugPrint("Cluster weights for speaker \(speakerId): \(finalizedClusters.map(\.weight))")
        }
        
        if !tentativeSegments.isEmpty {
            finalizedSegments.append(contentsOf: tentativeSegments)
            tentativeSegments.removeAll()
        }
        
        hasOutliers = false
        isFinalized = true
    }
    
    public func updateCannotLink(with speakerIds: Set<Int>) {
        cannotLink.formUnion(speakerIds)
        cannotLink.remove(self.speakerId)
    }
    
    public func updateCannotLink(with speakerId: Int) {
        guard speakerId != self.speakerId else { return }
        cannotLink.insert(speakerId)
    }
    
    /// Absorb a speaker profile and finalize both this Speaker and the other being absorbed.
    /// - Parameters:
    ///   - other: The speaker to absorb
    public func absorbAndFinalize(_ other: SpeakerProfile) {
        self.finalize()
        other.finalize()
        other.speakerId = self.speakerId
        
        // 1. Compress clusters and outliers
        
        // Initialize and fill pair-wise distance matrix
        var matrix = EmbeddingDistanceMatrix(.upgma)
        matrix.reserve(self.finalizedClusterCount + other.finalizedClusterCount)
        
        for cluster in self.finalizedClusters {
            matrix.append(cluster.cppView)
        }
        for cluster in other.finalizedClusters {
            matrix.append(cluster.cppView)
        }
        
        // Extract clusters
        let dendrogram = matrix.dendrogram()
        let clusters = dendrogram.extractClusters(config.clusteringThreshold)
        
        let newClusterCentroids = clusters.map { cluster in
            SpeakerClusterCentroid(
                cluster: cluster,
                dendrogram: dendrogram,
                matrix: matrix,
                isFinalized: true
            )
        }
        
        // Free the matrix so its embeddings don't become dangling pointers
        matrix.free()
        
        self.finalizedClusters = newClusterCentroids
        
        // 2. Update segments
        if !other.finalizedSegments.isEmpty {
            let wasEmpty = finalizedSegments.isEmpty
            finalizedSegments.append(contentsOf: other.finalizedSegments)
            if !wasEmpty { finalizedSegments.sort() }
        }
        
        // 3. Inherit cannot-link constraints
        self.cannotLink.formUnion(other.cannotLink)
        self.cannotLink.remove(speakerId)
    }
    
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(id)
    }
    
    public static func == (lhs: SpeakerProfile, rhs: SpeakerProfile) -> Bool {
        return lhs.id == rhs.id
    }
}

public class SpeakerClusterCentroid: EmbeddingVector {
    public var id: UUID { embedding.id }
    public let embedding: SpeakerEmbedding
    public var weight: Float
    public var isFinalized: Bool
    
    public var bufferView: UnsafeBufferPointer<Float> { embedding.bufferView }
    public var buffer: UnsafeBufferPointer<Float> { embedding.bufferView }
    public var baseAddress: UnsafePointer<Float>? { embedding.baseAddress }
    public var magnitude: Float { embedding.magnitude }
    
    var cppView: SpeakerEmbeddingWrapper {
        embedding.withUnsafeMutableBufferPointer { embeddingBuf in
            SpeakerEmbeddingWrapper.init(
                embeddingBuf.baseAddress,
                weight
            )
        }
    }
    
    public init(
        id: UUID = UUID(),
        weight: Float,
        isFinalized: Bool = true
    ) {
        self.embedding = SpeakerEmbedding(id: id, startFrame: 0, endFrame: 0)
        self.isFinalized = isFinalized
        self.weight = weight
    }
    
    public init(
        id: UUID = UUID(),
        embedding: SpeakerEmbedding,
        weight: Float,
        isFinalized: Bool = true
    ) {
        // Create a deep copy of the embedding
        self.embedding = SpeakerEmbedding(id: id, embedding: embedding.bufferView, startFrame: embedding.startFrame, endFrame: embedding.endFrame)
        self.isFinalized = isFinalized
        self.weight = weight
    }
    
    init(
        cluster: Cluster,
        dendrogram: borrowing Dendrogram,
        matrix: borrowing EmbeddingDistanceMatrix,
        isFinalized: Bool = true
    ) {
        self.weight = cluster.weight()
        self.embedding = SpeakerEmbedding(startFrame: 0, endFrame: 0)
        self.isFinalized = isFinalized
        
        embedding.withUnsafeMutableBufferPointer { embeddingBuf in
            var embeddingView = SpeakerEmbeddingWrapper(embeddingBuf.baseAddress)
            matrix.computeUnitCentroidOf(cluster, &embeddingView)
        }
        
        embedding.setMagnitudeUnsafe(to: 1.0)
    }
    
    public func update(with centroid: SpeakerClusterCentroid, updateVector: Bool = true) {
        self.weight += centroid.weight
        
        if updateVector {
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
        }
    }
    
    /// Update the centroid in place
    @inline(__always)
    public func update(with segment: EmbeddingSegment) {
        guard let centroid = segment.centroid else {
            return
        }
        update(with: centroid)
    }
    
    public func setTo(_ other: SpeakerClusterCentroid) {
        self.embedding.withUnsafeMutableBufferPointer { muPtr in
            _ = muPtr.initialize(fromContentsOf: other.embedding.bufferView)
        }
        self.weight = other.weight
        self.isFinalized = other.isFinalized
    }
    
    @inline(__always)
    public func deepCopy(keepingId: Bool = false) -> SpeakerClusterCentroid {
        SpeakerClusterCentroid(
            id: keepingId ? self.id : UUID(),
            embedding: self.embedding,
            weight: self.weight,
            isFinalized: self.isFinalized
        )
    }
    
    /// Get the cosine distance to another embedding vector
    /// - Parameter other: Another embedding vector
    /// - Returns: The cosine distance between this centroid to the speaker embedding
    @inline(__always)
    public func cosineDistance<E>(to other: E) -> Float where E: EmbeddingVector {
        return embedding.cosineDistance(to: other)
    }

    public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Float>) throws -> R) rethrows -> R {
        return try embedding.withUnsafeBufferPointer(body)
    }
    
    public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Float>) throws -> R) rethrows -> R {
        return try embedding.withUnsafeMutableBufferPointer(body)
    }
    
    public static func ==(lhs: SpeakerClusterCentroid, rhs: SpeakerClusterCentroid) -> Bool {
        return lhs.id == rhs.id
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(self.id)
    }
}
