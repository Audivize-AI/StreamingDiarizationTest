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

/// Configuration for the embedding graph
public struct EmbeddingGraphConfig {
    /// Number of nearest neighbors for kNN graph
    public var k: Int = 30
    public var numNeg: Int = 5
    
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
    
    /// Step the force simulation forward by one frame
    /// Call this from a display link or timer for smooth animation
    /// Optimized: O(n * (k + samples)) instead of O(n²)
    public func stepSimulation() {
        let n = nodes.count
        guard n > 1 else { return }
        
        // Skip if already stable
//        guard !isStable else { return }
        
        // UMAP Parameters controlled by slider
        // separation threshold -> min_dist
        let minDist = max(0.01, config.distanceThreshold)
        
        // Approximate 'a' and 'b' parameters for the given min_dist
        // (Empirically inverse to spread)
        let a = 1.6 * (0.1 / minDist)
        let b: Float = 1.0
        
        var forces = [SIMD2<Float>](repeating: .zero, count: n)
        
        // 1. ATTRACTION: Only between kNN neighbors where similarity is high
        for edge in edges {
            let i = edge.u
            let j = edge.v
            guard i < n && j < n else { continue }
            
            // Only attract if within distance threshold (semantically close)
            if edge.distance > config.distanceThreshold { continue }
            
            // Weight based on similarity
            let similarity = 1.0 - edge.distance
            let w = max(0.01, similarity)
            
            let delta = nodes[j].position - nodes[i].position
            let distSq = simd_length_squared(delta)
            
            guard distSq > 1e-6 else { continue }
            
            // UMAP Attraction Gradient: 2ab * d^(2b-2) / (1 + a*d^2b) * w
            // Assuming b=1 for simplicity: 2a / (1 + a*d^2) * w
            
            let num = 2.0 * a * b * pow(distSq, b - 1.0)
            let den = 1.0 + a * pow(distSq, b)
            let mag = (num / den) * w
            
            let force = delta * mag * config.forceStrength * 5.0 // Scale up for visibility
            
            forces[i] += force
            forces[j] -= force
        }
        
        // 2. REPULSION: K-Nearest Euclidean Neighbors
        let repulsionK = min(config.numNeg, n - 1) // Using numNeg as K for repulsion
        
        for i in 0..<n {
            // Find K nearest Euclidean neighbors
            var euclideanDists: [(idx: Int, distSq: Float)] = []
            euclideanDists.reserveCapacity(n)
            
            let posI = nodes[i].position
            
            for j in 0..<n where j != i {
                let delta = nodes[j].position - posI
                let distSq = simd_length_squared(delta)
                euclideanDists.append((j, distSq))
            }
            
            euclideanDists.sort { $0.distSq < $1.distSq }
            
            for (j, plotDistSq) in euclideanDists.prefix(repulsionK) {
                let cosineD = pairwiseDistances[i * numNodes + j]
                
                // Repulse if semantically distant
                if cosineD > config.distanceThreshold {
                    let delta = nodes[i].position - nodes[j].position // Away from J
                    
                    // UMAP Repulsion: 2b / ((1 + a*d^2b) * d) * (1-w)
                    // Simplified: Strong near 0, decays
                    
                    let den = (0.001 + plotDistSq) * (1.0 + a * pow(plotDistSq, b))
                    let mag = (2.0 * b) / den
                    
                    // Limit max repulsion
                    let clampedMag = min(mag, 10.0)
                    
                    forces[i] += delta * clampedMag * config.forceStrength * 0.5
                }
            }
        }
        
        // 3. GRAVITY: Pull to center to keep graph bounded and fit on screen
        // Low strength harmonic potential (F = -k*x)
        let gravity = config.forceStrength * 0.1
        for i in 0..<n {
            forces[i] -= nodes[i].position * gravity
        }
        
        // Apply forces and damping, track max velocity
        var maxSpeed: Float = 0
        for i in 0..<n {
            velocities[i] = velocities[i] * config.damping + forces[i]
            
            // Clamp velocity
            let speed = simd_length(velocities[i])
            maxSpeed = max(maxSpeed, speed)
            
            if speed > 0.05 {
                velocities[i] *= (0.05 / speed)
            }
            
            nodes[i].position += velocities[i]
        }
        
        // 4. POST-STEP CENTERING: Inspect centroid and shift to origin
        // This ensures the camera is always focused on the "mass" of the graph
        if n > 0 {
            var centroid: SIMD2<Float> = .zero
            for node in nodes {
                centroid += node.position
            }
            centroid /= Float(n)
            
            // Apply shift if significant
            if simd_length_squared(centroid) > 1e-6 {
                for i in 0..<n {
                    nodes[i].position -= centroid
                }
            }
            
            // 5. UPDATE AUTO-SCALE: Calculate bounds and update scale factor
            // This ensures the graph always fits in the view
            var maxD: Float = 0
            for node in nodes {
                maxD = max(maxD, abs(node.position.x))
                maxD = max(maxD, abs(node.position.y))
            }
            
            if maxD > 1e-6 {
                // Target: fit within 0.9 of normalized range [-1, 1]
                let targetScale = 0.9 / maxD
                // Smooth interpolation to prevent jitter (camera smoothing)
                autoScaleFactor = autoScaleFactor * 0.7 + targetScale * 0.3
            }
        }
        
        // Output max speed for stability check (though often ignored in continuous mode)
        if maxSpeed < 0.001 {
            stabilityCounter += 1
            if stabilityCounter >= stabilityThreshold {
                isStable = true
            }
        } else {
            stabilityCounter = 0
        }
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
    
    /// Project embeddings to 2D using PCA
    private func projectPCA(embeddings: [[Float]]) -> [SIMD2<Float>] {
        guard !embeddings.isEmpty else { return [] }
        
        let n = embeddings.count
        let d = embeddings[0].count
        
        // Center the data
        var mean = [Float](repeating: 0, count: d)
        for emb in embeddings {
            vDSP_vadd(mean, 1, emb, 1, &mean, 1, vDSP_Length(d))
        }
        var scale = Float(n)
        vDSP_vsdiv(mean, 1, &scale, &mean, 1, vDSP_Length(d))
        
        var centered: [[Float]] = embeddings.map { emb in
            var result = [Float](repeating: 0, count: d)
            vDSP_vsub(mean, 1, emb, 1, &result, 1, vDSP_Length(d))
            return result
        }
        
        // For efficiency, use power iteration to find top 2 principal components
        let pc1 = powerIteration(data: centered, numIterations: 20)
        
        // Deflate data for second component
        for i in 0..<n {
            var proj: Float = 0
            vDSP_dotpr(centered[i], 1, pc1, 1, &proj, vDSP_Length(d))
            var scaledPC1 = [Float](repeating: 0, count: d)
            var projScalar = proj
            vDSP_vsmul(pc1, 1, &projScalar, &scaledPC1, 1, vDSP_Length(d))
            vDSP_vsub(scaledPC1, 1, centered[i], 1, &centered[i], 1, vDSP_Length(d))
        }
        
        let pc2 = powerIteration(data: centered, numIterations: 20)
        
        // Project to 2D
        var positions: [SIMD2<Float>] = []
        positions.reserveCapacity(n)
        
        for emb in embeddings {
            var centeredEmb = [Float](repeating: 0, count: d)
            vDSP_vsub(mean, 1, emb, 1, &centeredEmb, 1, vDSP_Length(d))
            
            var x: Float = 0, y: Float = 0
            vDSP_dotpr(centeredEmb, 1, pc1, 1, &x, vDSP_Length(d))
            vDSP_dotpr(centeredEmb, 1, pc2, 1, &y, vDSP_Length(d))
            
            positions.append(SIMD2<Float>(x, y))
        }
        
        // Normalize positions to [-1, 1]
        return normalizePositions(positions)
    }
    
    /// Power iteration to find dominant eigenvector
    private func powerIteration(data: [[Float]], numIterations: Int) -> [Float] {
        guard !data.isEmpty else { return [] }
        let d = data[0].count
        
        // Random initialization
        var v = (0..<d).map { _ in Float.random(in: -1...1) }
        v = l2Normalize(v)
        
        for _ in 0..<numIterations {
            // v = X^T * X * v
            var intermediate = [Float](repeating: 0, count: data.count)
            for (i, row) in data.enumerated() {
                vDSP_dotpr(row, 1, v, 1, &intermediate[i], vDSP_Length(d))
            }
            
            var newV = [Float](repeating: 0, count: d)
            for (i, row) in data.enumerated() {
                var scaled = [Float](repeating: 0, count: d)
                var scalar = intermediate[i]
                vDSP_vsmul(row, 1, &scalar, &scaled, 1, vDSP_Length(d))
                vDSP_vadd(newV, 1, scaled, 1, &newV, 1, vDSP_Length(d))
            }
            
            v = l2Normalize(newV)
        }
        
        return v
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
    
    // MARK: - Spectral Clustering
    
    /// Perform spectral clustering on the embeddings using eigengap heuristic for automatic k detection
    /// - Parameters:
    ///   - numClusters: Number of clusters to create (if nil, uses eigengap heuristic to auto-detect)
    ///   - affinityThreshold: Distance threshold for affinity (edges above this are weak)
    ///   - maxClusters: Maximum number of clusters to consider when auto-detecting (default: 10)
    /// - Returns: True if clustering was successful
    @discardableResult
    public func performSpectralClustering(numClusters: Int? = nil, affinityThreshold: Float? = nil, maxClusters: Int = 10) -> Bool {
        let n = nodes.count
        guard n > 2 else { return false }
        
        let threshold = affinityThreshold ?? config.distanceThreshold
        
        // 1. Build affinity matrix using Gaussian kernel on cosine distances
        // A[i,j] = exp(-d[i,j]^2 / (2 * sigma^2)) for close points, 0 otherwise
        let sigma: Float = threshold / 2.0
        let sigmaSq2 = 2.0 * sigma * sigma
        
        var affinity = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            for j in 0..<n {
                if i == j {
                    affinity[i * n + j] = 1.0  // Self-affinity
                } else {
                    let dist = pairwiseDistances[i * numNodes + j]
                    if dist <= threshold * 1.5 {  // Only consider relatively close points
                        affinity[i * n + j] = exp(-dist * dist / sigmaSq2)
                    }
                }
            }
        }
        
        // 2. Compute degree matrix D (diagonal containing row sums)
        var degree = [Float](repeating: 0, count: n)
        for i in 0..<n {
            var sum: Float = 0
            for j in 0..<n {
                sum += affinity[i * n + j]
            }
            degree[i] = sum
        }
        
        // 3. Build normalized Laplacian: L_sym = D^(-1/2) * L * D^(-1/2)
        // Where L = D - A (unnormalized Laplacian)
        // This is equivalent to: L_sym = I - D^(-1/2) * A * D^(-1/2)
        var normalizedAffinity = [Float](repeating: 0, count: n * n)
        for i in 0..<n {
            let di = degree[i] > 1e-6 ? 1.0 / sqrt(degree[i]) : 0.0
            for j in 0..<n {
                let dj = degree[j] > 1e-6 ? 1.0 / sqrt(degree[j]) : 0.0
                normalizedAffinity[i * n + j] = affinity[i * n + j] * di * dj
            }
        }
        
        // 4. Determine number of clusters using eigengap heuristic (if not specified)
        let k: Int
        if let numClusters = numClusters {
            k = min(numClusters, n)
        } else {
            // Minimum clusters = number of distinct speaker slots with embeddings
            // This ensures we don't merge speakers that the diarizer already separated
            let minK = Set(nodes.map { $0.speakerIndex }).count
            
            // Compute eigenvalues to find the eigengap
            let maxK = min(maxClusters, n - 1)
            let (eigenvalues, _) = computeTopKEigenvectorsWithValues(matrix: normalizedAffinity, n: n, k: maxK)
            
            // Find largest eigengap starting from minK
            // (eigenvalues are sorted descending for normalized affinity)
            // The optimal k is where the gap between consecutive eigenvalues is largest
            var bestGap: Float = 0
            var bestK = minK  // Start with minimum = number of active speakers
            
            for i in minK..<eigenvalues.count {
                let gap = eigenvalues[i - 1] - eigenvalues[i]
                if gap > bestGap {
                    bestGap = gap
                    bestK = i
                }
            }
            
            k = max(minK, min(bestK, n))
            print("[SpectralClustering] Auto-detected \(k) clusters (minK=\(minK), eigengap=\(String(format: "%.4f", bestGap)), allEigens=\(eigenvalues))")
        }
        
        guard k > 0 && k <= n else { return false }
        
        // 5. Find top-k eigenvectors of normalized affinity
        let eigenvectors = computeTopKEigenvectors(matrix: normalizedAffinity, n: n, k: k)
        
        guard eigenvectors.count == k else { return false }
        
        // 6. Form matrix Y where rows are the k-dimensional embeddings for each point
        // Normalize rows to unit length for k-means
        var Y = [[Float]](repeating: [Float](repeating: 0, count: k), count: n)
        for i in 0..<n {
            var row = [Float](repeating: 0, count: k)
            for j in 0..<k {
                row[j] = eigenvectors[j][i]
            }
            // Normalize row
            var norm: Float = 0
            vDSP_svesq(row, 1, &norm, vDSP_Length(k))
            norm = sqrt(norm)
            if norm > 1e-6 {
                for j in 0..<k {
                    row[j] /= norm
                }
            }
            Y[i] = row
        }
        
        // 7. Apply k-means clustering to rows of Y
        var labels = kMeansClustering(data: Y, k: k, maxIterations: 50)
        
        // 8. Permute cluster labels to align with original speaker indices
        labels = permuteLabelsToMatchSpeakers(labels: labels, k: k)
        
        // 9. Assign cluster labels to nodes
        for i in 0..<n {
            nodes[i].clusterLabel = labels[i]
        }
        
        return true
    }
    
    /// Permute cluster labels to maximize alignment with original speaker indices
    /// Uses greedy assignment: for each speaker, find the cluster with most of their embeddings
    private func permuteLabelsToMatchSpeakers(labels: [Int], k: Int) -> [Int] {
        let n = nodes.count
        guard n == labels.count && k > 0 else { return labels }
        
        // Build confusion matrix: confusionMatrix[speaker][cluster] = count
        let numSpeakers = Set(nodes.map { $0.speakerIndex }).count
        var confusionMatrix = [[Int]](repeating: [Int](repeating: 0, count: k), count: numSpeakers)
        
        for i in 0..<n {
            let speaker = nodes[i].speakerIndex
            let cluster = labels[i]
            if speaker < numSpeakers && cluster < k {
                confusionMatrix[speaker][cluster] += 1
            }
        }
        
        // Greedy assignment: map clusters to speakers to maximize overlap
        var clusterToNewLabel = [Int: Int]()
        var usedNewLabels = Set<Int>()
        
        // Sort by total count in each speaker to prioritize speakers with more embeddings
        let speakerOrder = (0..<numSpeakers).sorted { sp1, sp2 in
            confusionMatrix[sp1].reduce(0, +) > confusionMatrix[sp2].reduce(0, +)
        }
        
        for speaker in speakerOrder {
            // Find the cluster that has most embeddings from this speaker and isn't assigned yet
            var bestCluster = -1
            var bestCount = 0
            for cluster in 0..<k {
                if !clusterToNewLabel.keys.contains(cluster) {
                    if confusionMatrix[speaker][cluster] > bestCount {
                        bestCount = confusionMatrix[speaker][cluster]
                        bestCluster = cluster
                    }
                }
            }
            
            if bestCluster >= 0 && !usedNewLabels.contains(speaker) {
                clusterToNewLabel[bestCluster] = speaker
                usedNewLabels.insert(speaker)
            }
        }
        
        // Assign remaining clusters to unused labels
        var nextLabel = 0
        for cluster in 0..<k {
            if clusterToNewLabel[cluster] == nil {
                while usedNewLabels.contains(nextLabel) {
                    nextLabel += 1
                }
                clusterToNewLabel[cluster] = nextLabel
                usedNewLabels.insert(nextLabel)
                nextLabel += 1
            }
        }
        
        // Apply permutation
        return labels.map { clusterToNewLabel[$0] ?? $0 }
    }
    
    /// Simple k-means clustering
    private func kMeansClustering(data: [[Float]], k: Int, maxIterations: Int) -> [Int] {
        let n = data.count
        guard n > 0 && k > 0 else { return [] }
        
        let d = data[0].count
        
        // Initialize centroids using k-means++ style: pick first at random, then weighted by distance
        var centroids = [[Float]]()
        var usedIndices = Set<Int>()
        
        // First centroid is random
        let firstIdx = Int.random(in: 0..<n)
        centroids.append(data[firstIdx])
        usedIndices.insert(firstIdx)
        
        // Pick remaining centroids
        while centroids.count < k {
            // Calculate squared distance to nearest centroid for each unused point
            var distances = [Float](repeating: 0, count: n)
            for i in 0..<n {
                if usedIndices.contains(i) {
                    distances[i] = 0  // Already a centroid, distance = 0
                    continue
                }
                var minDist: Float = .infinity
                for c in centroids {
                    var dist: Float = 0
                    for j in 0..<d {
                        let diff = data[i][j] - c[j]
                        dist += diff * diff
                    }
                    minDist = min(minDist, dist)
                }
                // Clamp to avoid infinity in the sum
                distances[i] = minDist.isFinite ? minDist : 0
            }
            
            // Pick next centroid with probability proportional to distance squared
            let totalDist = distances.reduce(0, +)
            if totalDist < 1e-6 || !totalDist.isFinite {
                // All points are on centroids or distances are degenerate, just pick any unused
                for i in 0..<n where !usedIndices.contains(i) {
                    centroids.append(data[i])
                    usedIndices.insert(i)
                    break
                }
            } else {
                var threshold = Float.random(in: 0..<totalDist)
                var found = false
                for i in 0..<n {
                    if usedIndices.contains(i) { continue }
                    threshold -= distances[i]
                    if threshold <= 0 {
                        centroids.append(data[i])
                        usedIndices.insert(i)
                        found = true
                        break
                    }
                }
                // Fallback if we didn't find one (shouldn't happen, but safety)
                if !found {
                    for i in 0..<n where !usedIndices.contains(i) {
                        centroids.append(data[i])
                        usedIndices.insert(i)
                        break
                    }
                }
            }
        }
        
        var labels = [Int](repeating: 0, count: n)
        
        for _ in 0..<maxIterations {
            // Assignment step
            var changed = false
            for i in 0..<n {
                var bestDist: Float = .infinity
                var bestLabel = 0
                for (c, centroid) in centroids.enumerated() {
                    var dist: Float = 0
                    for j in 0..<d {
                        let diff = data[i][j] - centroid[j]
                        dist += diff * diff
                    }
                    if dist < bestDist {
                        bestDist = dist
                        bestLabel = c
                    }
                }
                if labels[i] != bestLabel {
                    labels[i] = bestLabel
                    changed = true
                }
            }
            
            if !changed { break }
            
            // Update step
            var counts = [Int](repeating: 0, count: k)
            var newCentroids = [[Float]](repeating: [Float](repeating: 0, count: d), count: k)
            
            for i in 0..<n {
                let label = labels[i]
                counts[label] += 1
                for j in 0..<d {
                    newCentroids[label][j] += data[i][j]
                }
            }
            
            for c in 0..<k {
                if counts[c] > 0 {
                    for j in 0..<d {
                        newCentroids[c][j] /= Float(counts[c])
                    }
                    centroids[c] = newCentroids[c]
                }
            }
        }
        
        return labels
    }
    
    /// Compute top-k eigenvectors using power iteration with deflation
    private func computeTopKEigenvectors(matrix: [Float], n: Int, k: Int) -> [[Float]] {
        var eigenvectors: [[Float]] = []
        var deflatedMatrix = matrix
        
        for _ in 0..<k {
            // Power iteration to find dominant eigenvector
            var v = (0..<n).map { _ in Float.random(in: -1...1) }
            v = l2Normalize(v)
            
            for _ in 0..<30 {  // Power iteration steps
                // v = M * v
                var newV = [Float](repeating: 0, count: n)
                for i in 0..<n {
                    for j in 0..<n {
                        newV[i] += deflatedMatrix[i * n + j] * v[j]
                    }
                }
                v = l2Normalize(newV)
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
        
        for _ in 0..<k {
            // Power iteration to find dominant eigenvector
            var v = (0..<n).map { _ in Float.random(in: -1...1) }
            v = l2Normalize(v)
            
            for _ in 0..<30 {  // Power iteration steps
                // v = M * v
                var newV = [Float](repeating: 0, count: n)
                for i in 0..<n {
                    for j in 0..<n {
                        newV[i] += deflatedMatrix[i * n + j] * v[j]
                    }
                }
                v = l2Normalize(newV)
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
    
    // MARK: - UMAP Implementation
    
    /// Run UMAP-style optimization on the layout
    private func runUMAP(
        initialPositions: [SIMD2<Float>],
        embeddings: [[Float]],
        iterations: Int
    ) -> [SIMD2<Float>] {
        var pos = initialPositions
        let n = pos.count
        guard n > 1 else { return pos }
        
        // UMAP-style parameters
        let initialLR: Float = 1.0
        let minDist: Float = 0.1
        
        // a, b for t-distribution-like kernel: Q(d) = 1 / (1 + a * d^(2b))
        // For min_dist=0.1: a ≈ 1.58, b ≈ 0.895
        let a: Float = 1.58
        let b: Float = 0.895
        
        // Build adjacency set for quick neighbor lookup
        var neighborSet: [Set<Int>] = Array(repeating: Set(), count: n)
        for edge in edges {
            neighborSet[edge.u].insert(edge.v)
            neighborSet[edge.v].insert(edge.u)
        }
        
        // Compute edge weights based on high-dimensional similarity
        // Using exponential decay from cosine distance, normalized per-node
        var edgeWeights: [Int: [Int: Float]] = [:]
        for i in 0..<n {
            edgeWeights[i] = [:]
            let neighbors = Array(neighborSet[i])
            guard !neighbors.isEmpty else { continue }
            
            // Compute local sigma (distance to k-th neighbor / 3 for smooth decay)
            var distances: [Float] = []
            for j in neighbors {
                distances.append(cosineDistance(embeddings[i], embeddings[j]))
            }
            let sigma = max(0.01, (distances.max() ?? 0.1) / 3.0)
            
            // Compute weights
            for j in neighbors {
                let d = cosineDistance(embeddings[i], embeddings[j])
                let w = exp(-max(0, d - distances.min()!) / sigma)
                edgeWeights[i]![j] = w
            }
        }
        
        // Number of negative samples per positive edge
        let numNegativeSamples = 10
        
        // Optimization loop
        for iter in 0..<iterations {
            // Learning rate with warm restart decay
            let progress = Float(iter) / Float(iterations)
            let lr = initialLR * (1.0 - progress)
            
            var gradients = [SIMD2<Float>](repeating: .zero, count: n)
            
            // Attractive forces (neighbors only)
            for edge in edges {
                let i = edge.u
                let j = edge.v
                
                let delta = pos[j] - pos[i]
                let distSq = simd_length_squared(delta)
                
                // Skip if points are on top of each other
                guard distSq > 1e-6 else { continue }
                
                let dist = sqrt(distSq)
                
                // Attractive gradient: pull neighbors together
                // grad = 2ab * d^(2b-2) / (1 + a*d^2b) * direction
                let dPow2b = pow(distSq, b)
                let q = 1.0 / (1.0 + a * dPow2b)
                
                // Weight from high-D similarity
                let w = edgeWeights[i]?[j] ?? 0.5
                
                let attractMag = 2.0 * a * b * pow(distSq, b - 1.0) * q * w
                let attractForce = (delta / dist) * attractMag
                
                gradients[i] += attractForce * lr
                gradients[j] -= attractForce * lr
            }
            
            // Repulsive forces (negative sampling)
            for i in 0..<n {
                for _ in 0..<numNegativeSamples {
                    // Sample a random non-neighbor
                    var j = Int.random(in: 0..<n)
                    var attempts = 0
                    while (j == i || neighborSet[i].contains(j)) && attempts < 10 {
                        j = Int.random(in: 0..<n)
                        attempts += 1
                    }
                    if j == i { continue }
                    
                    let delta = pos[i] - pos[j]
                    let distSq = simd_length_squared(delta) + 0.001
                    let dist = sqrt(distSq)
                    
                    // Repulsive gradient: push non-neighbors apart
                    // Stronger when points are close, weaker when far
                    let dPow2b = pow(distSq, b)
                    let q = 1.0 / (1.0 + a * dPow2b)
                    
                    // Repulsion is scaled by (1-q) to be stronger for close non-neighbors
                    let repulseMag = 2.0 * b * (1.0 - q) / (distSq + 0.01)
                    let repulseForce = (delta / dist) * repulseMag * 0.1  // Scale down repulsion
                    
                    gradients[i] += repulseForce * lr
                }
            }
            
            // Apply gradients with clipping
            for i in 0..<n {
                let len = simd_length(gradients[i])
                if len > 4.0 {
                    gradients[i] *= (4.0 / len)
                }
                pos[i] += gradients[i]
            }
        }
        
        return normalizePositions(pos)
    }
}
