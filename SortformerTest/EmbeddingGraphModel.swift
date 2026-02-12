//
//  EmbeddingGraphModel.swift
//  SortformerTest
//
//  Real-time embedding graph model with PCA projection and kNN edges.
//

import Foundation
import Accelerate
import simd
import Combine

#if canImport(UMAP)
import UMAP
#endif

#if canImport(UMAP)
import UMAP
#endif

/// Configuration for the embedding graph
public struct EmbeddingGraphConfig {
    /// Number of nearest neighbors for kNN graph
    public var k: Int = 30
    public var numNeg: Int = 5
    
    // UMAP Parameters
    public var umapNeighbors: Int = 15
    public var umapMinDist: Float = 0.1
    public var umapEpochs: Int = 200
    public var showAllEdges: Bool = false
    
    /// Cosine distance threshold - edges below this are drawn
    /// Also determines attract vs repel behavior
    public var distanceThreshold: Float = 0.3
    
    /// Force strength for the simulation (lower = smoother)
    public var forceStrength: Float = 0.005
    
    /// Damping factor for velocities (higher = less jittery)
    public var damping: Float = 0.85
    
    /// Position scaling factor (higher = more zoomed in)
    public var positionScale: Float = 2.0
    
    /// Minimum number of embeddings needed to show graph
    public var minEmbeddingCount: Int = 1
    
    public init() {}
}

/// A node in the embedding graph
public struct GraphNode: Identifiable {
    public let id: Int
    public var position: SIMD2<Float>
    public var embedding: [Float]
    public var speakerIndex: Int
    public var segmentIndex: Int
    public var startFrame: Int
    public var endFrame: Int
    public var clusterLabel: Int?
    /// UUID of the original TitaNetEmbedding for cross-view linking
    public var embeddingId: UUID
}

/// An edge in the embedding graph (stored as adjacency)
public struct GraphEdge: Hashable {
    public let u: Int
    public let v: Int
    public let distance: Float
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(min(u, v))
        hasher.combine(max(u, v))
    }
    
    public static func == (lhs: GraphEdge, rhs: GraphEdge) -> Bool {
        (lhs.u == rhs.u && lhs.v == rhs.v) || (lhs.u == rhs.v && lhs.v == rhs.u)
    }
}

/// Clustering method selection
public enum ClusteringMethod: String, CaseIterable {
    case spectralKMeans = "Spectral + K-Means"
    case constrainedAHC = "Constrained AHC"
}

/// Observable model for the embedding graph visualization
@MainActor
public class EmbeddingGraphModel: ObservableObject {
    
    // MARK: - Published State
    
    @Published public var nodes: [GraphNode] = []
    @Published public var edges: [GraphEdge] = []
    @Published public var selectedNodeIndex: Int? = nil
    @Published public var hoveredNodeIndex: Int? = nil
    @Published public var config: EmbeddingGraphConfig = EmbeddingGraphConfig()
    
    /// Auto-zoom scale factor calculated by simulation to fit graph in view
    @Published public var autoScaleFactor: Float = 1.0
    
    /// Selected embedding UUID for cross-view linking (timeline ↔ graph)
    @Published public var selectedEmbeddingId: UUID? = nil
    
    /// Whether the cluster plot is visible
    @Published public var isClusterPlotVisible: Bool = false
    
    /// User-selected number of clusters (nil = use eigengap heuristic)
    @Published public var selectedClusterCount: Int? = nil
    
    /// Eigengap-optimal cluster count (computed after clustering)
    @Published public var eigengapOptimalK: Int = 2
    
    /// Maximum number of clusters to show in slider
    @Published public var maxClusterCount: Int = 10
    
    // MARK: - Private State
    
    /// Cache of precomputed edges per node (adjacency list)
    private var adjacencyList: [[Int]] = []
    
    /// Cache of edge distances
    private var edgeDistances: [Int: [Int: Float]] = [:]
    
    /// Last update hash to avoid redundant recomputation
    private var lastEmbeddingHash: Int = 0
    
    /// Node velocities for smooth animation
    private var velocities: [SIMD2<Float>] = []
    
    /// Pairwise cosine distances (upper triangle, flattened)
    private var pairwiseDistances: [Float] = []
    
    /// Number of nodes for indexing pairwiseDistances
    private var numNodes: Int = 0
    
    /// Whether the simulation has stabilized (velocities below threshold)
    private var isStable: Bool = false
    
    /// Frames of stability before stopping
    private var stabilityCounter: Int = 0
    private let stabilityThreshold: Int = 30  // Stop after 30 frames of low velocity
    
    // MARK: - Initialization
    
    public init() {}
    
    // MARK: - Selection API
    
    /// Select a node by its embedding UUID (called from timeline when clicking embedding stroke)
    public func selectByEmbeddingId(_ id: UUID?) {
        selectedEmbeddingId = id
        if let id = id {
            selectedNodeIndex = nodes.firstIndex { $0.embeddingId == id }
        } else {
            selectedNodeIndex = nil
        }
    }
    
    /// Select a node by index (called from graph when clicking node)
    public func selectByNodeIndex(_ index: Int?) {
        selectedNodeIndex = index
        if let idx = index, idx < nodes.count {
            selectedEmbeddingId = nodes[idx].embeddingId
        } else {
            selectedEmbeddingId = nil
        }
    }
    
    /// Get node for a given embedding ID
    public func nodeForEmbeddingId(_ id: UUID) -> GraphNode? {
        nodes.first { $0.embeddingId == id }
    }
    
    // MARK: - Public API
    
    /// Update the graph from a list of embedding segments
    /// This initializes positions and computes pairwise distances
    public func update(from segments: [EmbeddingSegment]) {
        // Collect all embeddings from all segments
        var allEmbeddings: [(embedding: [Float], speakerIndex: Int, segmentIndex: Int, startFrame: Int, endFrame: Int, embeddingId: UUID)] = []
        
        for (segIdx, segment) in segments.enumerated() {
            for emb in segment.embeddings {
                allEmbeddings.append((
                    embedding: emb.embedding,
                    speakerIndex: segment.speakerIndex,
                    segmentIndex: segIdx,
                    startFrame: emb.startFrame,
                    endFrame: emb.endFrame,
                    embeddingId: emb.id
                ))
            }
        }
        
        // Check if we need to recompute
        let newHash = computeHash(allEmbeddings.map { $0.embedding })
        guard newHash != lastEmbeddingHash else { return }
        lastEmbeddingHash = newHash
        
        // Need minimum embeddings
        guard allEmbeddings.count >= config.minEmbeddingCount else {
            nodes = []
            edges = []
            adjacencyList = []
            edgeDistances = [:]
            velocities = []
            pairwiseDistances = []
            numNodes = 0
            return
        }
        
        let n = allEmbeddings.count
        numNodes = n
        
        // L2-normalize embeddings for cosine distance
        let normalizedEmbeddings = allEmbeddings.map { l2Normalize($0.embedding) }
        
        // Compute ALL pairwise cosine distances (for force simulation)
        // Store upper triangle only: index = i * n + j where i < j
        pairwiseDistances = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in (i+1)..<n {
                let dist = cosineDistance(normalizedEmbeddings[i], normalizedEmbeddings[j])
                pairwiseDistances[i * n + j] = dist
                pairwiseDistances[j * n + i] = dist // Mirror for easy access
            }
        }
        
        // Build kNN graph for edge visualization
        buildKNNGraph(embeddings: normalizedEmbeddings, k: config.k)
        
        // Check for existing nodes with same embeddingId to preserve positions
        var existingPositions: [UUID: SIMD2<Float>] = [:]
        for node in nodes {
            existingPositions[node.embeddingId] = node.position
        }
        
        // Build nodes with random or preserved positions
        nodes = allEmbeddings.enumerated().map { idx, data in
            let position: SIMD2<Float>
            if let existing = existingPositions[data.embeddingId] {
                // Preserve existing position
                position = existing
            } else {
                // Random position in unit square centered at origin
                position = SIMD2<Float>(
                    Float.random(in: -0.5...0.5),
                    Float.random(in: -0.5...0.5)
                )
            }
            
            return GraphNode(
                id: idx,
                position: position,
                embedding: normalizedEmbeddings[idx],
                speakerIndex: data.speakerIndex,
                segmentIndex: data.segmentIndex,
                startFrame: data.startFrame,
                endFrame: data.endFrame,
                clusterLabel: nil,
                embeddingId: data.embeddingId
            )
        }
        
        // Initialize velocities and reset stability
        velocities = [SIMD2<Float>](repeating: .zero, count: n)
        isStable = false
        stabilityCounter = 0
    }
    
    /// Get cosine similarity (1 - distance) for edge thickness
    public func cosineSimilarity(u: Int, v: Int) -> Float {
        guard u < numNodes && v < numNodes && u != v else { return 0 }
        let dist = pairwiseDistances[u * numNodes + v]
        return max(0, 1.0 - dist)
    }
    
    /// Get edges for a specific node (including reverse edges)
    public func edgesForNode(_ nodeIndex: Int) -> [GraphEdge] {
        guard nodeIndex < adjacencyList.count else { return [] }
        
        var result: [GraphEdge] = []
        for neighbor in adjacencyList[nodeIndex] {
            if let dist = edgeDistances[nodeIndex]?[neighbor] {
                result.append(GraphEdge(u: nodeIndex, v: neighbor, distance: dist))
            }
        }
        
        // Also find edges where this node is the target
        for (srcIdx, neighbors) in adjacencyList.enumerated() {
            if neighbors.contains(nodeIndex) && srcIdx != nodeIndex {
                if let dist = edgeDistances[srcIdx]?[nodeIndex] {
                    let edge = GraphEdge(u: srcIdx, v: nodeIndex, distance: dist)
                    if !result.contains(edge) {
                        result.append(edge)
                    }
                }
            }
        }
        
        return result
    }
    
    /// Get all edges below the threshold
    public var thresholdedEdges: [GraphEdge] {
        edges.filter { $0.distance <= config.distanceThreshold }
    }
    
    // MARK: - Private Methods
    
    /// Compute a simple hash of embeddings for change detection
    private func computeHash(_ embeddings: [[Float]]) -> Int {
        var hasher = Hasher()
        hasher.combine(embeddings.count)
        for emb in embeddings.prefix(10) {
            hasher.combine(emb.prefix(8).map { Int($0 * 1000) })
        }
        return hasher.finalize()
    }
    
    /// L2-normalize an embedding vector using Accelerate
    private func l2Normalize(_ vec: [Float]) -> [Float] {
        var norm: Float = 0
        vDSP_svesq(vec, 1, &norm, vDSP_Length(vec.count))
        norm = sqrt(norm)
        
        guard norm > 0 else { return vec }
        
        var result = [Float](repeating: 0, count: vec.count)
        var divisor = norm
        vDSP_vsdiv(vec, 1, &divisor, &result, 1, vDSP_Length(vec.count))
        return result
    }
    
    /// Compute cosine distance between two L2-normalized vectors
    /// d = 1 - cos(θ) = 1 - dot(a, b) for normalized vectors
    private func cosineDistance(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count else { return 2.0 }
        
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        return 1.0 - dot
    }
    
    /// Normalize positions to [-1, 1] range
    private func normalizePositions(_ positions: [SIMD2<Float>]) -> [SIMD2<Float>] {
        guard !positions.isEmpty else { return [] }
        
        var minX: Float = .infinity, maxX: Float = -.infinity
        var minY: Float = .infinity, maxY: Float = -.infinity
        
        for p in positions {
            minX = min(minX, p.x)
            maxX = max(maxX, p.x)
            minY = min(minY, p.y)
            maxY = max(maxY, p.y)
        }
        
        let rangeX = maxX - minX
        let rangeY = maxY - minY
        let range = max(rangeX, rangeY, 0.001)
        
        let centerX = (minX + maxX) / 2
        let centerY = (minY + maxY) / 2
        
        return positions.map { p in
            SIMD2<Float>(
                (p.x - centerX) / range * 2,
                (p.y - centerY) / range * 2
            )
        }
    }
    
    /// Build kNN graph efficiently
    private func buildKNNGraph(embeddings: [[Float]], k: Int) {
        let n = embeddings.count
        guard n > 1 else {
            adjacencyList = []
            edges = []
            edgeDistances = [:]
            return
        }
        
        let actualK = min(k, n - 1)
        adjacencyList = Array(repeating: [], count: n)
        edgeDistances = [:]
        var allEdges = Set<GraphEdge>()
        
        // For each node, find k nearest neighbors
        // This is O(n^2 * d) but with early termination for large n
        for i in 0..<n {
            // Compute distances to all other nodes
            var distances: [(idx: Int, dist: Float)] = []
            distances.reserveCapacity(n - 1)
            
            for j in 0..<n where j != i {
                let dist = cosineDistance(embeddings[i], embeddings[j])
                distances.append((j, dist))
            }
            
            // Partial sort to get top k (O(n) instead of O(n log n))
            distances.sort { $0.dist < $1.dist }
            let topK = distances.prefix(actualK)
            
            // Store adjacency and distances
            adjacencyList[i] = topK.map { $0.idx }
            edgeDistances[i] = Dictionary(uniqueKeysWithValues: topK.map { ($0.idx, $0.dist) })
            
            // Add edges (deduplicated via Set)
            for (neighbor, dist) in topK {
                allEdges.insert(GraphEdge(u: i, v: neighbor, distance: dist))
            }
        }
        
        edges = Array(allEdges).sorted { $0.distance < $1.distance }
    }
    
    // MARK: - Clustering
    
    /// Perform clustering on the embeddings using the specified method
    /// - Parameters:
    ///   - method: Clustering method to use
    ///   - numClusters: Number of clusters (nil = auto-detect using eigengap heuristic)
    ///   - maxClusters: Maximum number of clusters for auto-detection
    /// - Returns: True if clustering was successful
    @discardableResult
    public func performClustering(
        method: ClusteringMethod = .constrainedAHC,
        numClusters: Int? = nil,
        maxClusters: Int = 20
    ) async -> Bool {
        let n = nodes.count
        guard n > 2 else { return false }
        
        // Build constraints from node information
        let (cannotLink, mustLinkGroups) = buildConstraints()
        
        await Task.yield()
        
        var k: Int = 0
        let labels: [Int]
        
        switch method {
        case .spectralKMeans:
            // For Spectral, use Eigengap heuristic if k is not provided
            if let num = numClusters {
                k = min(num, n)
            } else {
                k = computeOptimalK_Eigengap(maxClusters: maxClusters, cannotLink: cannotLink)
            }
            labels = performSpectralKMeansInternal(k: k, cannotLink: cannotLink, mustLinkGroups: mustLinkGroups)
            
        case .constrainedAHC:
            // For AHC, use Elbow method on merge distances
            let result = performConstrainedAHCAndDetectK(
                targetK: numClusters,
                maxClusters: maxClusters,
                cannotLink: cannotLink,
                mustLinkGroups: mustLinkGroups
            )
            k = result.detectedK
            labels = result.labels
        }
        
        eigengapOptimalK = k
        maxClusterCount = max(k + 2, min(maxClusters, 12))
        
        guard k > 0 && k <= n else { return false }
        
        await Task.yield()
        
        // Permute labels by arrival order (0 = first appearing speaker)
        let permutedLabels = permuteLabelsByArrivalOrder(labels: labels, k: k)
        
        // Assign cluster labels to nodes
        for i in 0..<n {
            nodes[i].clusterLabel = permutedLabels[i]
        }
        
        // Compute UMAP positions for visualization
        await computeUMAPPositions()
        
        return true
        
    }
    
    /// Permute cluster labels based on arrival order (first appearing cluster gets label 0)
    private func permuteLabelsByArrivalOrder(labels: [Int], k: Int) -> [Int] {
        let n = nodes.count
        guard n == labels.count && k > 0 else { return labels }
        
        // Find first occurrence frame for each cluster
        var firstFrame = [Int: Int]()
        
        for i in 0..<n {
            let cluster = labels[i]
            let frame = nodes[i].startFrame
            if let existing = firstFrame[cluster] {
                if frame < existing { firstFrame[cluster] = frame }
            } else {
                firstFrame[cluster] = frame
            }
        }
        
        // Sort clusters by start frame
        let sortedClusters = firstFrame.keys.sorted { firstFrame[$0]! < firstFrame[$1]! }
        
        // Map old cluster ID to new label (0, 1, 2...)
        var clusterToNewLabel = [Int: Int]()
        for (idx, cluster) in sortedClusters.enumerated() {
            clusterToNewLabel[cluster] = idx
        }
        
        // Assign remaining clusters to unused labels
        var nextLabel = sortedClusters.count
        for cluster in 0..<k {
            if clusterToNewLabel[cluster] == nil {
                clusterToNewLabel[cluster] = nextLabel
                nextLabel += 1
            }
        }
        
        // Apply mapping
        return labels.map { clusterToNewLabel[$0] ?? $0 }
    }
    
    // MARK: - Constraint Building
    
    /// Build cannot-link and must-link constraints from nodes
    private func buildConstraints() -> (cannotLink: Set<[Int]>, mustLinkGroups: [[Int]]) {
        let n = nodes.count
        
        // Cannot-link: pairs from different speaker slots
        var cannotLink = Set<[Int]>()
        for i in 0..<n {
            for j in (i + 1)..<n {
                if nodes[i].speakerIndex != nodes[j].speakerIndex {
                    cannotLink.insert([i, j])
                }
            }
        }
        
        // Must-link: group by segmentIndex (embeddings from same segment)
        var segmentGroups: [Int: [Int]] = [:]
        for i in 0..<n {
            let segIdx = nodes[i].segmentIndex
            segmentGroups[segIdx, default: []].append(i)
        }
        
        let mustLinkGroups = segmentGroups.values.filter { $0.count > 1 }
        
        return (cannotLink, Array(mustLinkGroups))
    }
    
    /// Compute optimal number of clusters using eigengap heuristic (Spectral only)
    private func computeOptimalK_Eigengap(maxClusters: Int, cannotLink: Set<[Int]>) -> Int {
        let n = nodes.count
        let threshold = config.distanceThreshold
        let sigma: Float = threshold / 2.0
        let sigmaSq2 = 2.0 * sigma * sigma
        
        var affinity = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                if i == j {
                    affinity[i * n + j] = 1.0
                } else if cannotLink.contains([min(i, j), max(i, j)]) {
                    affinity[i * n + j] = 0.0
                } else {
                    let dist = pairwiseDistances[i * numNodes + j]
                    if dist <= threshold * 1.5 {
                        affinity[i * n + j] = exp(-dist * dist / sigmaSq2)
                    }
                }
            }
        }
        
        var degree = [Float](repeating: 0, count: n)
        for i in 0..<n {
            for j in 0..<n {
                degree[i] += affinity[i * n + j]
            }
        }
        
        var normalizedAffinity = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            let di = degree[i] > 1e-6 ? 1.0 / sqrt(degree[i]) : 0.0
            for j in 0..<n {
                let dj = degree[j] > 1e-6 ? 1.0 / sqrt(degree[j]) : 0.0
                normalizedAffinity[i * n + j] = affinity[i * n + j] * di * dj
            }
        }
        
        let maxK = min(maxClusters, n - 1)
        let (eigenvalues, _) = computeTopKEigenvectorsWithValues(matrix: normalizedAffinity, n: n, k: maxK)
        
        let minK = Set(nodes.map { $0.speakerIndex }).count
        var bestGap: Float = 0
        var bestK = minK
        
        for i in minK..<eigenvalues.count {
            let gap = eigenvalues[i - 1] - eigenvalues[i]
            if gap > bestGap {
                bestGap = gap
                bestK = i
            }
        }
        
        print("[Clustering] Auto-detected \(bestK) clusters (eigengap=\(String(format: "%.4f", bestGap)))")
        return max(minK, min(bestK, n))
    }
    
    // MARK: - Spectral Clustering with Constrained K-Means
    
    private func performSpectralKMeansInternal(k: Int, cannotLink: Set<[Int]>, mustLinkGroups: [[Int]]) -> [Int] {
        let n = nodes.count
        let threshold = config.distanceThreshold
        let sigma: Float = threshold / 2.0
        let sigmaSq2 = 2.0 * sigma * sigma
        
        var affinity = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                if i == j {
                    affinity[i * n + j] = 1.0
                } else if cannotLink.contains([min(i, j), max(i, j)]) {
                    affinity[i * n + j] = 0.0
                } else {
                    let dist = pairwiseDistances[i * numNodes + j]
                    if dist <= threshold * 1.5 {
                        affinity[i * n + j] = exp(-dist * dist / sigmaSq2)
                    }
                }
            }
        }
        
        var degree = [Float](repeating: 0, count: n)
        for i in 0..<n { for j in 0..<n { degree[i] += affinity[i * n + j] } }
        
        var normalizedAffinity = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            let di = degree[i] > 1e-6 ? 1.0 / sqrt(degree[i]) : 0.0
            for j in 0..<n {
                let dj = degree[j] > 1e-6 ? 1.0 / sqrt(degree[j]) : 0.0
                normalizedAffinity[i * n + j] = affinity[i * n + j] * di * dj
            }
        }
        
        let eigenvectors = computeTopKEigenvectors(matrix: normalizedAffinity, n: n, k: k)
        guard eigenvectors.count == k else { return Array(0..<n) }
        
        var Y = [[Float]](repeating: [Float](repeating: 0, count: k), count: n)
        for i in 0..<n {
            var row = [Float](repeating: 0, count: k)
            for j in 0..<k { row[j] = eigenvectors[j][i] }
            var norm: Float = 0
            vDSP_svesq(row, 1, &norm, vDSP_Length(k))
            norm = sqrt(norm)
            if norm > 1e-6 { for j in 0..<k { row[j] /= norm } }
            Y[i] = row
        }
        
        return constrainedKMeans(data: Y, k: k, cannotLink: cannotLink, mustLinkGroups: mustLinkGroups)
    }
    
    private func constrainedKMeans(data: [[Float]], k: Int, cannotLink: Set<[Int]>, mustLinkGroups: [[Int]], maxIterations: Int = 50) -> [Int] {
        let n = data.count
        guard n > 0 && k > 0 else { return [] }
        let d = data[0].count
        
        var centroids = [[Float]]()
        var usedIndices = Set<Int>()
        centroids.append(data[0])
        usedIndices.insert(0)
        
        while centroids.count < k {
            var bestIdx = -1
            var bestMinDist: Float = -1
            for i in 0..<n where !usedIndices.contains(i) {
                var minDist: Float = .infinity
                for c in centroids {
                    var dist: Float = 0
                    for j in 0..<d { let diff = data[i][j] - c[j]; dist += diff * diff }
                    minDist = min(minDist, sqrt(dist))
                }
                if minDist > bestMinDist && minDist.isFinite { bestMinDist = minDist; bestIdx = i }
            }
            if bestIdx >= 0 { centroids.append(data[bestIdx]); usedIndices.insert(bestIdx) } else { break }
        }
        
        var labels = [Int](repeating: 0, count: n)
        var mustLinkTo = [Int: Int]()
        for group in mustLinkGroups { let rep = group[0]; for idx in group { mustLinkTo[idx] = rep } }
        
        for _ in 0..<maxIterations {
            var changed = false
            for i in 0..<n {
                if let rep = mustLinkTo[i], rep != i { if labels[i] != labels[rep] { labels[i] = labels[rep]; changed = true }; continue }
                var bestDist: Float = .infinity
                var bestLabel = labels[i]
                for (c, centroid) in centroids.enumerated() {
                    var violates = false
                    for j in 0..<n where labels[j] == c && j != i { if cannotLink.contains([min(i, j), max(i, j)]) { violates = true; break } }
                    if violates { continue }
                    var dist: Float = 0
                    for j in 0..<d { let diff = data[i][j] - centroid[j]; dist += diff * diff }
                    if dist < bestDist { bestDist = dist; bestLabel = c }
                }
                if labels[i] != bestLabel { labels[i] = bestLabel; changed = true }
            }
            if !changed { break }
            var counts = [Int](repeating: 0, count: k)
            var newCentroids = [[Float]](repeating: [Float](repeating: 0, count: d), count: k)
            for i in 0..<n { let l = labels[i]; counts[l] += 1; for j in 0..<d { newCentroids[l][j] += data[i][j] } }
            for c in 0..<k { if counts[c] > 0 { for j in 0..<d { newCentroids[c][j] /= Float(counts[c]) }; centroids[c] = newCentroids[c] } }
        }
        return labels
    }
    
    // MARK: - Constrained AHC on Cosine Distances
    
    /// Run Constrained AHC and optionally detect the best K using the Elbow method
    private func performConstrainedAHCAndDetectK(
        targetK: Int?,
        maxClusters: Int,
        cannotLink: Set<[Int]>,
        mustLinkGroups: [[Int]]
    ) -> (labels: [Int], detectedK: Int) {
        let n = nodes.count
        guard n > 0 else { return ([], 0) }
        
        // Initialize clusters: pre-merge must-link groups
        var clusterMembers: [Set<Int>] = (0..<n).map { [$0] }
        var activeClusterIds = Set(0..<n)
        
        // Pre-merge must-link groups
        for group in mustLinkGroups where group.count > 1 {
            let target = group[0]
            for i in 1..<group.count {
                let source = group[i]
                if activeClusterIds.contains(source) {
                    clusterMembers[target].formUnion(clusterMembers[source])
                    clusterMembers[source].removeAll()
                    activeClusterIds.remove(source)
                }
            }
        }
        
        func canMerge(_ c1: Int, _ c2: Int) -> Bool {
            for i in clusterMembers[c1] {
                for j in clusterMembers[c2] {
                    if cannotLink.contains([min(i, j), max(i, j)]) { return false }
                }
            }
            return true
        }
        
        func clusterDistance(_ c1: Int, _ c2: Int) -> Float {
            var sum: Float = 0; var count = 0
            for i in clusterMembers[c1] {
                for j in clusterMembers[c2] { sum += pairwiseDistances[i * numNodes + j]; count += 1 }
            }
            return count > 0 ? sum / Float(count) : Float.infinity
        }
        
        // Store merge history: (clusterToKeep, clusterToRemove, distance at merge)
        var mergeHistory: [(Int, Int, Float)] = []
        
        // Loop until 1 cluster or constraints prevent further merges
        while activeClusterIds.count > 1 {
            var bestI = -1, bestJ = -1, bestDist: Float = .infinity
            let activeList = Array(activeClusterIds).sorted()
            
            // Find best merge
            for (idx1, c1) in activeList.enumerated() {
                for c2 in activeList[(idx1 + 1)...] {
                    guard canMerge(c1, c2) else { continue }
                    let dist = clusterDistance(c1, c2)
                    if dist < bestDist { bestDist = dist; bestI = c1; bestJ = c2 }
                }
            }
            
            guard bestI >= 0 && bestJ >= 0 else { break }
            
            // Record merge
            mergeHistory.append((bestI, bestJ, bestDist))
            
            // Execute merge
            clusterMembers[bestI].formUnion(clusterMembers[bestJ])
            clusterMembers[bestJ].removeAll()
            activeClusterIds.remove(bestJ)
            
            // Check if we hit targetK
            if let target = targetK, activeClusterIds.count == target {
                var labels = [Int](repeating: 0, count: n)
                let finalClusters = Array(activeClusterIds).sorted()
                for (newLabel, clusterId) in finalClusters.enumerated() {
                    for pointIdx in clusterMembers[clusterId] { labels[pointIdx] = newLabel }
                }
                return (labels, target)
            }
        }
        
        // If targetK was set but unreachable, return current state
        if let target = targetK {
            var labels = [Int](repeating: 0, count: n)
            let finalClusters = Array(activeClusterIds).sorted()
            for (newLabel, clusterId) in finalClusters.enumerated() {
                for pointIdx in clusterMembers[clusterId] { labels[pointIdx] = newLabel }
            }
            return (labels, activeClusterIds.count)
        }
        
        // Auto-detection using Elbow Method on merge distances
        let nMerges = mergeHistory.count
        if nMerges == 0 { return ([Int](repeating: 0, count: n), 1) }

        // Initial K after must-link merges
        let initialK = n - (mustLinkGroups.reduce(0) { $0 + max(0, $1.count - 1) })
        let minK = Set(nodes.map { $0.speakerIndex }).count
        
        var bestCutIdx = nMerges
        var maxJump: Float = 0
        
        // Analyze jumps (looking backwards from last merge towards start)
        // We want to stop BEFORE a big jump in distance.
        // History: i=0 (first merge) ... i=nMerges-1 (last merge)
        // K decreases as i increases.
        
        for i in 0..<(nMerges - 1) {
            let currentK = initialK - (i + 1)
            
            // Only evaluate cuts that result in valid k range
            if currentK > maxClusters { continue }
            if currentK < minK { break }
            
            let distCurrent = mergeHistory[i].2
            let distNext = mergeHistory[i+1].2
            
            let jump = distNext - distCurrent
            if jump > maxJump {
                maxJump = jump
                bestCutIdx = i + 1 // Ensure we keep state after merge i, before merge i+1
            }
        }
        
        let detectedK = initialK - bestCutIdx
        print("[Clustering] Auto-detected \(detectedK) clusters (AHC Elbow, maxJump=\(String(format: "%.4f", maxJump)))")
        
        // Reconstruct state at bestCutIdx by replaying merges
        var replayClusterMembers: [Set<Int>] = (0..<n).map { [$0] }
        var replayActiveIds = Set(0..<n)
        
        // Re-apply must links
        for group in mustLinkGroups where group.count > 1 {
            let target = group[0]
            for i in 1..<group.count {
                let source = group[i]
                if replayActiveIds.contains(source) {
                    replayClusterMembers[target].formUnion(replayClusterMembers[source])
                    replayClusterMembers[source].removeAll()
                    replayActiveIds.remove(source)
                }
            }
        }
        
        // Replay history
        for i in 0..<bestCutIdx {
            let (keep, remove, _) = mergeHistory[i]
            replayClusterMembers[keep].formUnion(replayClusterMembers[remove])
            replayClusterMembers[remove].removeAll()
            replayActiveIds.remove(remove)
        }
        
        var labels = [Int](repeating: 0, count: n)
        let finalClusters = Array(replayActiveIds).sorted()
        for (newLabel, clusterId) in finalClusters.enumerated() {
            for pointIdx in replayClusterMembers[clusterId] { labels[pointIdx] = newLabel }
        }
        
        return (labels, replayActiveIds.count)
    }
    
    // MARK: - True UMAP Visualization
    

    private func computeUMAPPositions() async {
        let n = nodes.count
        guard n > 1 else { return }
        
        // Compute kNN graph (needed for both implementations)
        let k = min(config.umapNeighbors, n - 1)
        var knnIndices = [[Int]](repeating: [], count: n)
        var knnDists = [[Float]](repeating: [], count: n)
        
        for i in 0..<n {
            var dists: [(idx: Int, dist: Float)] = []
            for j in 0..<n where j != i {
                dists.append((j, pairwiseDistances[i * numNodes + j]))
            }
            dists.sort { $0.dist < $1.dist }
            let neighbors = dists.prefix(k)
            knnIndices[i] = neighbors.map { $0.idx }
            knnDists[i] = neighbors.map { $0.dist }
        }

#if canImport(UMAP)
        print("[UMAP] Using C++ Layout (UMAPPP)")
        
        // Flatten for C API
        var flatIndices = [Int32](repeating: -1, count: n * k)
        var flatDists = [Float](repeating: 0, count: n * k)
        
        for i in 0..<n {
            let count = knnIndices[i].count
            for j in 0..<k {
                if j < count {
                    flatIndices[i*k + j] = Int32(knnIndices[i][j])
                    flatDists[i*k + j] = knnDists[i][j]
                }
            }
        }
        
        var output = [Float](repeating: 0, count: n * 2)
        
        let status = umap_initialize(
            Int32(n), 2, Int32(k), 
            config.umapMinDist, 
            &flatIndices, &flatDists, 
            &output, 
            42
        )
        
        var positions = [SIMD2<Float>](repeating: .zero, count: n)
        
        if let status = status {
            // Run to completion
            umap_run(status, &output, Int32(config.umapEpochs))
            umap_free(status)
            
            for i in 0..<n {
                positions[i] = SIMD2<Float>(output[i*2], output[i*2+1])
            }
        }
#else
        print("[UMAP] Using Swift Fallback (Module not found)")
        
        let nNeighbors = k
        let nEpochs = config.umapEpochs
        
        // Compute sigmas
        var sigmas = [Float](repeating: 1.0, count: n)
        var rhos = [Float](repeating: 0.0, count: n)
        
        for i in 0..<n {
            guard !knnDists[i].isEmpty else { continue }
            rhos[i] = knnDists[i][0] // distance to 1st nearest neighbor
            
            // Binary search for sigma such that sum(exp(-d/sigma)) = log2(k)
            let target = log2(Float(nNeighbors))
            var lo: Float = 0.0
            var hi: Float = 100.0
            var mid: Float = 1.0
            
            for _ in 0..<20 {
                mid = (lo + hi) / 2.0
                var sum: Float = 0
                for d in knnDists[i] {
                    sum += exp(-(max(0, d - rhos[i])) / mid)
                }
                if sum < target { lo = mid } else { hi = mid }
            }
            sigmas[i] = mid
        }
        
        // Construct sparse weighted graph (symmetrized)
        var edgeWeights: [Int: [Int: Float]] = [:]
        for i in 0..<n { edgeWeights[i] = [:] }
        
        for i in 0..<n {
            for (idx, j) in knnIndices[i].enumerated() {
                let d = knnDists[i][idx]
                if d == 0 { continue }
                let w = exp(-(max(0, d - rhos[i])) / sigmas[i])
                
                // Union rule: w_sym = w_ij + w_ji - w_ij*w_ji
                let existing = edgeWeights[i]?[j] ?? 0
                let sym = w + existing - w * existing
                edgeWeights[i]![j] = sym
                edgeWeights[j]![i] = sym
            }
        }
        
        // Flatten weights for initialization
        var highDimWeights = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            for (j, w) in edgeWeights[i] ?? [:] {
                highDimWeights[i * n + j] = w
            }
        }
        
        // Initialize with spectral embedding (Laplacian eigenmaps)
        var positions = projectLaplacianEigenmaps(weights: highDimWeights, n: n)
        
        // UMAP parameters from config
        let (a, b) = umapParameters(minDist: config.umapMinDist)
        
        // Stochastic gradient descent
        for epoch in 0..<nEpochs {
            let lr = 1.0 - Float(epoch) / Float(nEpochs)
            var gradients = [SIMD2<Float>](repeating: .zero, count: n)
            
            for i in 0..<n {
                for (j, w) in edgeWeights[i] ?? [:] {
                    if w < 0.01 { continue }
                    
                    let delta = positions[j] - positions[i]
                    let distSq = simd_length_squared(delta) + 0.001
                    let dist = sqrt(distSq)
                    
                    let dPow2b = pow(distSq, b)
                    let q = 1.0 / (1.0 + a * dPow2b)
                    
                    // Attractive gradient
                    let attractMag = 2.0 * a * b * pow(distSq, b - 1) * q * w
                    gradients[i] += (delta / dist) * attractMag * lr
                    
                    // Repulsive (sample random non-neighbors)
                    for _ in 0..<config.numNeg {
                        let negJ = Int.random(in: 0..<n)
                        let isNeighbor = (edgeWeights[i]?[negJ] != nil) || (i == negJ)
                        if !isNeighbor {
                            let negDelta = positions[negJ] - positions[i]
                            let negDistSq = simd_length_squared(negDelta) + 0.001
                            let negDist = sqrt(negDistSq)
                            let negQ = 1.0 / (1.0 + a * pow(negDistSq, b))
                            let repulseMag = 2.0 * b * (1.0 - negQ) / (negDistSq + 0.01)
                            gradients[i] -= (negDelta / negDist) * repulseMag * 0.1 * lr
                        }
                    }
                }
            }
            
            // Apply gradients
            for i in 0..<n {
                positions[i] += gradients[i]
            }
            
            if epoch % 50 == 0 { await Task.yield() }
        }
#endif
        
        // Normalize and assign
        let normalized = normalizePositions(positions)
        for i in 0..<n {
            nodes[i].position = normalized[i]
        }
        
        // Update edges for display (same for both methods)
        var allEdges = Set<GraphEdge>()
        let threshold = config.distanceThreshold
        
        for i in 0..<n {
            for (idx, j) in knnIndices[i].enumerated() {
                let dist = knnDists[i][idx]
                if dist <= threshold {
                    allEdges.insert(GraphEdge(u: i, v: j, distance: dist))
                }
            }
        }
        
        edges = Array(allEdges).sorted { $0.distance < $1.distance }
    }
    
    private func projectLaplacianEigenmaps(weights: [Float], n: Int) -> [SIMD2<Float>] {
        // Build normalized Laplacian
        var degree = [Float](repeating: 0, count: n)
        for i in 0..<n { for j in 0..<n { degree[i] += weights[i * n + j] } }
        
        var normalizedL = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            let di = degree[i] > 1e-6 ? 1.0 / sqrt(degree[i]) : 0.0
            for j in 0..<n {
                let dj = degree[j] > 1e-6 ? 1.0 / sqrt(degree[j]) : 0.0
                normalizedL[i * n + j] = weights[i * n + j] * di * dj
            }
        }
        
        let eigenvectors = computeTopKEigenvectors(matrix: normalizedL, n: n, k: 3)
        guard eigenvectors.count >= 2 else {
            return (0..<n).map { _ in SIMD2<Float>(Float.random(in: -1...1), Float.random(in: -1...1)) }
        }
        
        // Use 2nd and 3rd eigenvectors (skip first trivial one)
        let ev1 = eigenvectors.count > 1 ? eigenvectors[1] : eigenvectors[0]
        let ev2 = eigenvectors.count > 2 ? eigenvectors[2] : eigenvectors[0]
        
        return (0..<n).map { i in SIMD2<Float>(ev1[i] * 10, ev2[i] * 10) }
    }
    
    /// Compute UMAP hyperparameters a and b based on min_dist
    private func umapParameters(minDist: Float) -> (Float, Float) {
        if minDist <= 0.05 { return (1.8, 0.95) }
        if minDist <= 0.15 { return (1.58, 0.89) }
        if minDist <= 0.25 { return (1.28, 0.82) }
        if minDist <= 0.50 { return (0.85, 0.70) }
        return (0.6, 0.6)
    }
    
    
    /// Compute top-k eigenvectors using power iteration with deflation
    private func computeTopKEigenvectors(matrix: [Float], n: Int, k: Int) -> [[Float]] {
        var eigenvectors: [[Float]] = []
        var deflatedMatrix = matrix
        
        for eigIdx in 0..<k {
            // Deterministic initialization: use alternating pattern based on eigenvector index
            var v = (0..<n).map { i in Float((i + eigIdx) % 2 == 0 ? 1 : -1) / sqrt(Float(n)) }
            v = l2Normalize(v)
            
            // Power iteration with convergence check
            let maxIter = 100
            let tolerance: Float = 1e-6
            
            for _ in 0..<maxIter {
                // v_new = M * v
                var newV = [Float](repeating: 0, count: n)
                for i in 0..<n {
                    for j in 0..<n {
                        newV[i] += deflatedMatrix[i * n + j] * v[j]
                    }
                }
                newV = l2Normalize(newV)
                
                // Check convergence: ||v_new - v|| < tolerance
                var diff: Float = 0
                for i in 0..<n {
                    let d = newV[i] - v[i]
                    diff += d * d
                }
                
                v = newV
                
                if sqrt(diff) < tolerance {
                    break
                }
            }
            
            eigenvectors.append(v)
            
            // Deflate: M = M - lambda * v * v^T
            // Compute eigenvalue (Rayleigh quotient)
            var Mv = [Float](repeating: 0, count: n)
            for i in 0..<n {
                for j in 0..<n {
                    Mv[i] += deflatedMatrix[i * n + j] * v[j]
                }
            }
            var lambda: Float = 0
            vDSP_dotpr(v, 1, Mv, 1, &lambda, vDSP_Length(n))
            
            // Subtract outer product
            for i in 0..<n {
                for j in 0..<n {
                    deflatedMatrix[i * n + j] -= lambda * v[i] * v[j]
                }
            }
        }
        
        return eigenvectors
    }
    
    /// Compute top-k eigenvectors and eigenvalues using power iteration with deflation
    /// Returns eigenvalues sorted in descending order
    private func computeTopKEigenvectorsWithValues(matrix: [Float], n: Int, k: Int) -> (eigenvalues: [Float], eigenvectors: [[Float]]) {
        var eigenvectors: [[Float]] = []
        var eigenvalues: [Float] = []
        var deflatedMatrix = matrix
        
        for eigIdx in 0..<k {
            // Deterministic initialization: use alternating pattern based on eigenvector index
            var v = (0..<n).map { i in Float((i + eigIdx) % 2 == 0 ? 1 : -1) / sqrt(Float(n)) }
            v = l2Normalize(v)
            
            // Power iteration with convergence check
            let maxIter = 100
            let tolerance: Float = 1e-6
            
            for _ in 0..<maxIter {
                // v_new = M * v
                var newV = [Float](repeating: 0, count: n)
                for i in 0..<n {
                    for j in 0..<n {
                        newV[i] += deflatedMatrix[i * n + j] * v[j]
                    }
                }
                newV = l2Normalize(newV)
                
                // Check convergence: ||v_new - v|| < tolerance
                var diff: Float = 0
                for i in 0..<n {
                    let d = newV[i] - v[i]
                    diff += d * d
                }
                
                v = newV
                
                if sqrt(diff) < tolerance {
                    break
                }
            }
            
            eigenvectors.append(v)
            
            // Compute eigenvalue (Rayleigh quotient)
            var Mv = [Float](repeating: 0, count: n)
            for i in 0..<n {
                for j in 0..<n {
                    Mv[i] += deflatedMatrix[i * n + j] * v[j]
                }
            }
            var lambda: Float = 0
            vDSP_dotpr(v, 1, Mv, 1, &lambda, vDSP_Length(n))
            eigenvalues.append(lambda)
            
            // Deflate: M = M - lambda * v * v^T
            for i in 0..<n {
                for j in 0..<n {
                    deflatedMatrix[i * n + j] -= lambda * v[i] * v[j]
                }
            }
        }
        
        return (eigenvalues, eigenvectors)
    }
}
