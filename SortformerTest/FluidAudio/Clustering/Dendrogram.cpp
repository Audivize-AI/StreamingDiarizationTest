//
// Created by Benjamin Lee on 1/29/26.
//
#include "Dendrogram.hpp"
#include "Cluster.hpp"
#include <iostream>
#include <queue>


Dendrogram::Dendrogram(CDistanceMatrix const& distMat, bool ignoreTentative) {
    // Setup
    long nodesRemaining = distMat.embeddingCount() - (ignoreTentative ? 0L : distMat.tentativeCount());
    if (nodesRemaining == 0) return;
    
    this->count = nodesRemaining * 2 - 1;
    this->_nodes = std::make_shared<DendrogramNode[]>(std::max(count, distMat._size));

    Aux aux = {
            .linkagePolicy          = distMat.linkagePolicy,
            .matrix                 = new float[distMat._matrixEndIndex],
            .activeFlags            = new bool[distMat._size],
            .matrixToNode           = new long[distMat._size],
            .freeIndices            = distMat._freeIndices,
            .size                   = distMat._size,
            .numClustersRemaining   = nodesRemaining
    };

    bool foundActive = false;
    int numActive = 0;

    // Fill _nodes and aux
    for (int i = 0; i < distMat._size; ++i) {
        bool isActive = distMat._embeddings[i].hasVector();
        aux.matrixToNode[i] = i;
        aux.activeFlags[i] = isActive;
        this->_nodes[i].matrixIndex = i;
        this->_nodes[i].weight = distMat._embeddings[i].weight();
        numActive += isActive;
        if (!foundActive && isActive) {
            aux.firstActiveMatrixIndex = i;
            foundActive = true;
        }
    }
    
    if (numActive == 1) {
        this->_rootId = aux.firstActiveMatrixIndex;
        return;
    }
    
    if (ignoreTentative) {
        for (auto [_, index]: distMat._tentativeIndices) {
            aux.freeIndices.push_back(index);
            aux.activeFlags[index] = false;
            --numActive;
        }
        
        if (numActive <= 0) return;
        foundActive = aux.activeFlags[aux.firstActiveMatrixIndex];
        while (!foundActive) {
            foundActive = aux.activeFlags[++aux.firstActiveMatrixIndex];
        }
    }

    if (numActive != nodesRemaining) {
        std::cerr << "ERROR: numActive != nodesRemaining" << std::endl;
    }
    
    // Copy the distance matrix
    std::copy(distMat.matrix, distMat.matrix + distMat._matrixEndIndex, aux.matrix);

    // Build the _nodes
    buildDendrogram(aux);
}


Dendrogram::Dendrogram(Dendrogram&& other) noexcept: 
        count(other.count),
        _rootId(other._rootId),
        elbowLinkage(other.elbowLinkage),
        _nodes(std::move(other._nodes)) {
    other.count = 0;
    other._rootId = 0;
    other.elbowLinkage = 0;
}


std::vector<Cluster> Dendrogram::extractSubClusters(long root, float linkageThreshold, long maxClusters) const {
    auto clusterRootIds = extractSubClusterRoots(root, linkageThreshold, maxClusters);
    std::vector<Cluster> results;
    results.reserve(clusterRootIds.size());

    for (auto nodeId: clusterRootIds) {
        results.emplace_back(collectClusterLeaves(nodeId));
    }

    return results;
}


std::vector<long> Dendrogram::extractSubClusterRoots(long root, float linkageThreshold, long maxClusters) const {
    if (root == -1) root = this->_rootId;
    if (root == -1) return {};

    // Use a Priority Queue to track candidate clusters.
    // We order by 'mergeDistance' so we always split the "least cohesive" cluster first.
    // Pair: <MergeDistance, NodeID>
    std::priority_queue<std::pair<float, long>> queue;

    if (linkageThreshold < 0) linkageThreshold = this->elbowLinkage;
    if (maxClusters < 1) maxClusters = std::numeric_limits<long>::max();

    std::vector<long> results;
    results.reserve(std::min(maxClusters, this->_nodes[root].count));

    // Initialize with the root
    queue.emplace(this->_nodes[root].mergeDistance, root);
    long currentClusterCount = 1;

    while (!queue.empty() && currentClusterCount < maxClusters) {
        // Get the worst offender (highest variance/distance)
        auto [dist, nodeId] = queue.top();

        if (dist <= linkageThreshold)
            break; // All remaining clusters will fail this check, so we're done  

        const auto& node = this->_nodes[nodeId];
        bool isLeaf = node.leftChild == -1;

        // Don't split leaves
        if (isLeaf) {
            queue.pop();
            results.emplace_back(nodeId);
            continue;
        }

        // Swap top node with its children
        queue.pop();
        queue.emplace(this->_nodes[node.leftChild].mergeDistance, node.leftChild);
        queue.emplace(this->_nodes[node.rightChild].mergeDistance, node.rightChild);

        ++currentClusterCount;
    }

    // Finalize
    while (!queue.empty()) {
        auto id = queue.top().second;
        results.emplace_back(id);
        queue.pop();
    }

    return results;
}


Cluster Dendrogram::collectClusterLeaves(long root) const {
    auto& clusterRoot = _nodes[root]; 
    auto cluster = Cluster(clusterRoot);
    long i = 0;

    std::vector<const DendrogramNode*> stack;
    stack.reserve(clusterRoot.count);
    stack.push_back(&clusterRoot);

    while (!stack.empty()) {
        const auto* node = stack.back();
        stack.pop_back();

        // Check if Leaf DendrogramNode (children are -1)
        if (node->leftChild < 0) {
            // For leaf nodes, matrixIndex holds the original input index. 
            // This can be used to look up the embedding in the distance matrix.
            cluster[i++] = node->matrixIndex;
        } else {
            stack.push_back(&this->_nodes[node->rightChild]);
            stack.push_back(&this->_nodes[node->leftChild]);
        }
    }
    return cluster;
}


void Dendrogram::buildDendrogram(Aux& aux) {
    if (aux.numClustersRemaining < 1) 
        return;
    
    long lastSurvivor = aux.firstActiveMatrixIndex; 
    const auto stack = new std::ptrdiff_t[aux.numClustersRemaining];
    long topIndex = 0;
    stack[0] = lastSurvivor;
    
    while (aux.numClustersRemaining > 1) {
        if (topIndex < 0)
            stack[++topIndex] = lastSurvivor;

        auto top = stack[topIndex];
        auto next = nearestNeighbor(top, aux);

        if (topIndex > 0 && next == stack[topIndex - 1]) {
            lastSurvivor = merge(top, next, aux);
            topIndex -= 2;
        } else {
            stack[++topIndex] = next;
        }
    }
    delete[] stack;
    
    this->_rootId = aux.matrixToNode[lastSurvivor];
}


std::ptrdiff_t Dendrogram::merge(std::ptrdiff_t leftIndex, std::ptrdiff_t rightIndex, Aux& aux) {
    std::ptrdiff_t matrixIndex;
    
    if (leftIndex > rightIndex) 
        std::swap(leftIndex, rightIndex);
    
    const auto leftId = aux.matrixToNode[leftIndex];
    const auto rightId = aux.matrixToNode[rightIndex];
    const auto wA = this->_nodes[leftId].weight; 
    const auto wB = this->_nodes[rightId].weight; 
    const auto wAB = wA + wB;
    const auto distAB = distance(rightIndex, leftIndex, aux);
    aux.activeFlags[leftIndex] = false;
    
    // Update the matrix
    const auto matrixRow = rightIndex;
    bool foundActive = false;

    // Update each (matrixRow, col)
    matrixIndex = matrixRow * (matrixRow - 1) / 2 + aux.firstActiveMatrixIndex;
    for (auto col = aux.firstActiveMatrixIndex; col < matrixRow; ++col, ++matrixIndex) {
        if (!aux.activeFlags[col]) continue;
        if (!foundActive) {
            aux.firstActiveMatrixIndex = col;
            foundActive = true;
        }

        const auto wC = this->_nodes[aux.matrixToNode[col]].weight;
        const auto distAC = distance(leftIndex, col, aux);
        const auto distBC = distance(rightIndex, col, aux);
        aux.matrix[matrixIndex] = aux.linkagePolicy->distance(distAC, wA, distBC, wB, distAB, wC);
    }
    
    // Update each (row, matrixRow) squaredDistanceTo for row > matrixRow
    // Start at (r, c) = (matrixRow + 1, matrixRow)
    // [i+1]([i+1]-1) / 2 + i = i(i+3) / 2
    matrixIndex = matrixRow * (matrixRow + 3) / 2;
    for (auto col = matrixRow + 1; col < aux.size; matrixIndex += col++) {
        if (!aux.activeFlags[col]) continue;
        
        const auto wC = this->_nodes[aux.matrixToNode[col]].weight;
        const auto distAC = distance(leftIndex, col, aux);
        const auto distBC = distance(rightIndex, col, aux);
        aux.matrix[matrixIndex] = aux.linkagePolicy->distance(distAC, wA, distBC, wB, distAB, wC);
    }
    
    long clusterRow;
    if (aux.freeIndices.empty()) {
        clusterRow = aux.size + aux.numMerged++;
    } else {
        clusterRow = aux.freeIndices.back();
        aux.freeIndices.pop_back();
    }
    
    // Write the node
    aux.matrixToNode[matrixRow] = clusterRow;
    this->_nodes[clusterRow].matrixIndex = matrixRow;
    this->_nodes[clusterRow].weight = wAB;
    this->_nodes[clusterRow].count = this->_nodes[leftId].count + this->_nodes[rightId].count;
    this->_nodes[clusterRow].mergeDistance = distAB;
    this->_nodes[clusterRow].leftChild = leftId;
    this->_nodes[clusterRow].rightChild = rightId;
    --aux.numClustersRemaining;
    
    // Update max gap
    auto childDist = std::max(this->_nodes[leftId].mergeDistance, this->_nodes[rightId].mergeDistance);
    if (!std::isfinite(childDist)) // Max gap must be finite
        return matrixRow;
    auto gap = distAB - childDist;
    if (gap > aux.maxGap) {
        aux.maxGap = gap;
        this->elbowLinkage = (distAB + childDist) / 2.f;
    }
    
    return matrixRow;
}


std::ptrdiff_t Dendrogram::nearestNeighbor(std::ptrdiff_t index, Aux& aux) {
    float minDist = std::numeric_limits<float>::infinity();
    std::ptrdiff_t nearestIndex = -1;

    // Update each (index, col)
    auto matrixIndex = index * (index - 1) / 2 + aux.firstActiveMatrixIndex;
    for (auto row = aux.firstActiveMatrixIndex; row < index; ++row, ++matrixIndex) {
        if (!aux.activeFlags[row]) continue;
        auto dist = distance(row, index, aux);
        if (dist < minDist) {
            minDist = dist;
            nearestIndex = row;
        }
    }

    // Scan each (row, index) for row > index, starting at (r, c) = (index + 1, index)
    matrixIndex = index * (index + 3) / 2;
    for (auto col = index + 1; col < aux.size; matrixIndex += col++) {
        if (!aux.activeFlags[col]) continue;
        auto dist = distance(index, col, aux);
        if (dist < minDist) {
            minDist = dist;
            nearestIndex = col;
        }
    }
    
    return nearestIndex;
}

Dendrogram& Dendrogram::operator=(Dendrogram&& other) noexcept {
    count = other.count;
    elbowLinkage = other.elbowLinkage;
    _rootId = other._rootId;
    _nodes = std::move(other._nodes);
    
    other.count = 0;
    other._rootId = 0;
    other.elbowLinkage = 0;
    
    return *this;
}
