//
// Created by Benjamin Lee on 1/29/26.
//
#include "Dendrogram.hpp"
#include "Cluster.hpp"
#include "ClusterLinkage.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <queue>

template<LinkagePolicy LP>
Dendrogram<LP>::Dendrogram(EmbeddingDistanceMatrix<LP>& distMat, ClusteringFilter filter) {
    // Setup
    bool includeFinalized = (filter & ClusteringFilter::finalized) != ClusteringFilter::none;
    bool includeTentative = (filter & ClusteringFilter::tentative) != ClusteringFilter::none;
    
    long nodesRemaining;
    switch (filter) {
        case ClusteringFilter::all:
            nodesRemaining = distMat._embeddingCount; 
            break;
        case ClusteringFilter::finalized:
            nodesRemaining = distMat._embeddingCount - static_cast<long>(distMat.tentativeIndices.size()); 
            break;
        case ClusteringFilter::tentative:
            nodesRemaining = static_cast<long>(distMat.tentativeIndices.size()); 
            break;
        case ClusteringFilter::none:
            return;
    }
    
    nodesRemaining = std::max<long>(0, nodesRemaining);
    this->count = std::max(nodesRemaining * 2 - 1, distMat._size);
    this->dendrogram = std::make_shared<DendrogramNode[]>(count);
    if (nodesRemaining == 0) return;

    Aux aux = {
            .matrix                 = new float[distMat.matrixEndIndex],
            .activeFlags            = new bool[distMat._size],
            .matrixToNode           = new long[distMat._size],
            .freeIndices            = distMat.freeIndices,
            .size                   = distMat._size,
            .matrixEndIndex         = distMat.matrixEndIndex,
            .numClustersRemaining   = nodesRemaining
    };
    
    bool foundActive = false;

    // Fill dendrogram and aux
    for (int i = 0; i < distMat._size; ++i) {
        bool isActive = distMat.embeddings[i].hasVector();
        aux.matrixToNode[i] = i;
        aux.activeFlags[i] = isActive;
        this->dendrogram[i].matrixIndex = i;
        this->dendrogram[i].weight = distMat.embeddings[i].weight();
        this->dendrogram[i].mergeDistance = distMat.embeddings[i].spread();

        if (!foundActive && isActive) {
            aux.firstActiveMatrixIndex = i;
            foundActive = true;
        }
    }
    
    // Mask out excluded indices
    if (includeFinalized) {
        applyConstraints(distMat.mustLinkIndices, aux);
        if (!includeTentative) {
            for (auto index: distMat.tentativeIndices) {
                aux.freeIndices.push_back(index);
                aux.activeFlags[index] = false;
            }
        } else {
            applyConstraints(distMat.tentativeMustLinkIndices, aux);
        }
        
        while (aux.activeFlags[aux.firstActiveMatrixIndex] == false) {
            ++aux.firstActiveMatrixIndex;
            if (aux.firstActiveMatrixIndex >= distMat._size)
                return;
        }
    } else { // It must be tentative only
        auto& tentativeIndices = distMat.tentativeIndices;
        std::sort(tentativeIndices.begin(), tentativeIndices.end());
        
        long lower = 0;
        foundActive = false;
        for (auto upper : tentativeIndices) {
            if (!foundActive && aux.activeFlags[upper]) {
                aux.firstActiveMatrixIndex = upper;
                foundActive = true;
            }
                
            for (long i = lower; i < upper; ++i) {
                aux.activeFlags[i] = false;
                aux.freeIndices.push_back(i);
            }
            lower = upper + 1;
        }
        
        for (long i = lower; i < distMat._size; ++i) {
            aux.activeFlags[i] = false;
            aux.freeIndices.push_back(i);
        }
        
        if (!foundActive)
            return;
        
        applyConstraints(distMat.tentativeMustLinkIndices, aux);
    }

    long activeCount = 0;
    foundActive = false;
    for (long i = 0; i < aux.size; ++i) {
        if (!aux.activeFlags[i]) continue;
        if (!foundActive) {
            aux.firstActiveMatrixIndex = i;
            foundActive = true;
        }
        ++activeCount;
    }

    aux.numClustersRemaining = activeCount;
    if (activeCount <= 0 || !foundActive) return;
    if (activeCount == 1) {
        this->rootId = aux.matrixToNode[aux.firstActiveMatrixIndex];
        return;
    }

    // Copy the distance matrix
    std::memcpy(aux.matrix, distMat.matrix, distMat.matrixEndIndex * sizeof(float));
    
    // Build the dendrogram
    buildDendrogram(aux);
}

template<LinkagePolicy LP>
Dendrogram<LP>::Dendrogram(EmbeddingDistanceMatrix<LP>& distMat, bool ignoreTentative) {
    // Setup
    long nodesRemaining = distMat._embeddingCount - (ignoreTentative ? static_cast<long>(distMat.tentativeIndices.size()) : 0);
    nodesRemaining = std::max<long>(0, nodesRemaining);
    this->count = std::max(nodesRemaining * 2 - 1, distMat._size);
    this->dendrogram = std::make_shared<DendrogramNode[]>(count);
    if (nodesRemaining == 0) return;

    Aux aux = {
            .matrix                 = new float[distMat.matrixEndIndex],
            .activeFlags            = new bool[distMat._size],
            .matrixToNode           = new long[distMat._size],
            .freeIndices            = distMat.freeIndices,
            .size                   = distMat._size,
            .matrixEndIndex         = distMat.matrixEndIndex,
            .numClustersRemaining   = nodesRemaining
    };

    bool foundActive = false;
    int numActive = 0;

    // Fill dendrogram and aux
    for (int i = 0; i < distMat._size; ++i) {
        bool isActive = distMat.embeddings[i].hasVector();
        aux.matrixToNode[i] = i;
        aux.activeFlags[i] = isActive;
        this->dendrogram[i].matrixIndex = i;
        this->dendrogram[i].weight = distMat.embeddings[i].weight();
        this->dendrogram[i].mergeDistance = distMat.embeddings[i].spread();
        numActive += isActive;
        if (!foundActive && isActive) {
            aux.firstActiveMatrixIndex = i;
            foundActive = true;
        }
    }
    
    if (ignoreTentative) {
        for (auto index: distMat.tentativeIndices) {
            aux.freeIndices.push_back(index);
            aux.activeFlags[index] = false;
            --numActive;
        }
        
        if (numActive <= 0) return;
        foundActive = aux.activeFlags[aux.firstActiveMatrixIndex];
        while (!foundActive) {
            foundActive = aux.activeFlags[++aux.firstActiveMatrixIndex];
        }
    } else {
        applyConstraints(distMat.tentativeMustLinkIndices, aux);
    }

    if (numActive != nodesRemaining) {
        std::cerr << "ERROR: numActive != nodesRemaining" << std::endl;
    }

    applyConstraints(distMat.mustLinkIndices, aux);

    long activeCount = 0;
    foundActive = false;
    for (long i = 0; i < aux.size; ++i) {
        if (!aux.activeFlags[i]) continue;
        if (!foundActive) {
            aux.firstActiveMatrixIndex = i;
            foundActive = true;
        }
        ++activeCount;
    }

    aux.numClustersRemaining = activeCount;
    if (activeCount <= 0 || !foundActive) return;
    if (activeCount == 1) {
        this->rootId = aux.matrixToNode[aux.firstActiveMatrixIndex];
        return;
    }

    // Copy the distance matrix
    std::memcpy(aux.matrix, distMat.matrix, distMat.matrixEndIndex * sizeof(float));

    // Build the dendrogram
    buildDendrogram(aux);
}

template<LinkagePolicy LP>
Dendrogram<LP>::Dendrogram(Dendrogram&& clustering) noexcept: 
        count(clustering.count),
        rootId(clustering.rootId), 
        elbowLinkage(clustering.elbowLinkage), 
        dendrogram(std::move(clustering.dendrogram)) {}
        
template<LinkagePolicy LP>
std::vector<Cluster> Dendrogram<LP>::extractClusters(float linkageThreshold, long maxClusters) const {
    if (this->rootId == -1) return {};
    
    // Use a Priority Queue to track candidate clusters.
    // We order by 'mergeDistance' so we always split the "least cohesive" cluster first.
    // Pair: <MergeDistance, NodeID>
    std::priority_queue<std::pair<float, long>> queue;
    
    if (linkageThreshold < 0) linkageThreshold = this->elbowLinkage;
    if (maxClusters < 1) maxClusters = std::numeric_limits<long>::max();
    
    std::vector<Cluster> results;
    results.reserve(std::min(maxClusters, this->dendrogram[this->rootId].count));

    // Initialize with the root
    queue.emplace(this->dendrogram[this->rootId].mergeDistance, this->rootId);
    long currentClusterCount = 1;

    while (!queue.empty() && currentClusterCount < maxClusters) {
        // Get the worst offender (highest variance/distance)
        auto [dist, nodeId] = queue.top();
        
        if (dist <= linkageThreshold)
            break; // All remaining clusters will fail this check, so we're done  

        const auto& node = this->dendrogram[nodeId];
        bool isLeaf = (node.leftChild == -1 || node.mustLink);
        
        // Don't split leaves
        if (isLeaf) {
            queue.pop();
            results.emplace_back(collectCluster(node));
            continue;
        }
        
        // Swap top node with its children
        queue.pop();
        queue.emplace(this->dendrogram[node.leftChild].mergeDistance, node.leftChild);
        queue.emplace(this->dendrogram[node.rightChild].mergeDistance, node.rightChild);

        ++currentClusterCount;
    }

    // Finalize
    while (!queue.empty()) {
        auto id = queue.top().second;
        results.emplace_back(collectCluster(this->dendrogram[id]));
        queue.pop();
    }

    return results;
}

template<LinkagePolicy LP>
Cluster Dendrogram<LP>::collectCluster(const DendrogramNode &root) const {
    auto cluster = Cluster(root);
    long i = 0;

    std::vector<const DendrogramNode*> stack;
    stack.reserve(root.count);
    stack.push_back(&root);

    while (!stack.empty()) {
        const auto* node = stack.back();
        stack.pop_back();

        // Check if Leaf DendrogramNode (children are -1)
        if (node->leftChild < 0) {
            // For leaf nodes, matrixIndex holds the original input index. 
            // This can be used to look up the embedding in the distance matrix.
            cluster[i++] = node->matrixIndex;
        } else {
            stack.push_back(&this->dendrogram[node->rightChild]);
            stack.push_back(&this->dendrogram[node->leftChild]);
        }
    }
    return cluster;
}

template<LinkagePolicy LP>
void Dendrogram<LP>::buildDendrogram(Aux& aux) {
    if (aux.numClustersRemaining < 1)
        return;

    if (aux.firstActiveMatrixIndex < 0 || aux.firstActiveMatrixIndex >= aux.size || !aux.activeFlags[aux.firstActiveMatrixIndex]) {
        for (long i = 0; i < aux.size; ++i) {
            if (aux.activeFlags[i]) {
                aux.firstActiveMatrixIndex = i;
                break;
            }
        }
        if (aux.firstActiveMatrixIndex < 0 || aux.firstActiveMatrixIndex >= aux.size || !aux.activeFlags[aux.firstActiveMatrixIndex]) {
            return;
        }
    }

    long lastSurvivor = aux.firstActiveMatrixIndex;
    std::vector<std::ptrdiff_t> stack;
    stack.reserve(static_cast<std::size_t>(std::max<long>(2, aux.numClustersRemaining)));
    stack.push_back(lastSurvivor);

    while (aux.numClustersRemaining > 1) {
        if (stack.empty()) {
            if (lastSurvivor < 0 || lastSurvivor >= aux.size || !aux.activeFlags[lastSurvivor]) {
                long nextActive = -1;
                for (long i = 0; i < aux.size; ++i) {
                    if (aux.activeFlags[i]) {
                        nextActive = i;
                        break;
                    }
                }
                if (nextActive < 0) break;
                lastSurvivor = nextActive;
            }
            stack.push_back(lastSurvivor);
        }

        auto top = stack.back();
        if (top < 0 || top >= aux.size || !aux.activeFlags[top]) {
            stack.pop_back();
            continue;
        }

        auto next = nearestNeighbor(top, aux);
        if (next < 0 || next == top) {
            break;
        }

        if (stack.size() > 1 && next == stack[stack.size() - 2]) {
            auto merged = merge(top, next, aux);
            if (merged < 0) break;
            lastSurvivor = merged;
            stack.pop_back();
            stack.pop_back();
        } else {
            stack.push_back(next);
        }
    }

    if (lastSurvivor >= 0 && lastSurvivor < aux.size) {
        this->rootId = aux.matrixToNode[lastSurvivor];
    }
}

template<LinkagePolicy LP>
void Dendrogram<LP>::applyConstraints(std::vector<MustLinkConstraint> const& constraints, Aux& aux) {
    for (auto& constraint: constraints) {
        std::ptrdiff_t survivor = -1;
        for (std::size_t i = 0; i < constraint.count; ++i) {
            auto index = constraint.indices[i];
            if (index < 0 || index >= aux.size || !aux.activeFlags[index]) continue;
            survivor = index;
            break;
        }
        if (survivor < 0) continue;

        for (std::size_t i = 1; i < constraint.count; ++i) {
            auto index = constraint.indices[i];
            if (index < 0 || index >= aux.size || !aux.activeFlags[index]) continue;
            survivor = merge(survivor, constraint.indices[i], aux, true);
            if (survivor < 0) break;
        }
    }
}

template<LinkagePolicy LP>
std::ptrdiff_t Dendrogram<LP>::merge(std::ptrdiff_t leftIndex, std::ptrdiff_t rightIndex, Aux& aux, bool mustLink) {
    std::ptrdiff_t matrixIndex;

    if (leftIndex == rightIndex) return leftIndex;
    if (leftIndex < 0 || rightIndex < 0 || leftIndex >= aux.size || rightIndex >= aux.size) {
        return -1;
    }
    if (!aux.activeFlags[leftIndex] && !aux.activeFlags[rightIndex]) {
        return -1;
    }
    if (!aux.activeFlags[leftIndex]) return rightIndex;
    if (!aux.activeFlags[rightIndex]) return leftIndex;
    
    if (leftIndex > rightIndex) 
        std::swap(leftIndex, rightIndex);
    
    const auto leftId = aux.matrixToNode[leftIndex];
    const auto rightId = aux.matrixToNode[rightIndex];
    const auto wA = this->dendrogram[leftId].weight; 
    const auto wB = this->dendrogram[rightId].weight; 
    const auto wAB = wA + wB;
    auto distAB = distance(rightIndex, leftIndex, aux);
    if (!std::isfinite(distAB)) {
        distAB = 2.f;
    } else {
        distAB = std::clamp(distAB, 0.f, 2.f);
    }
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

        auto wC = this->dendrogram[aux.matrixToNode[col]].weight;
        auto distAC = distance(leftIndex, col, aux);
        auto distBC = distance(rightIndex, col, aux);
        aux.matrix[matrixIndex] = LP::distance(distAC, wA, distBC, wB, distAB, wC);
    }
    
    // Update each (row, matrixRow) squaredDistanceTo for row > matrixRow
    // Start at (r, c) = (matrixRow + 1, matrixRow)
    // [i+1]([i+1]-1) / 2 + i = i(i+3) / 2
    matrixIndex = matrixRow * (matrixRow + 3) / 2;
    for (auto col = matrixRow + 1; col < aux.size; matrixIndex += col++) {
        if (!aux.activeFlags[col]) continue;
        
        // dist(AB, C) = ((wA + nC) * dist(A, C) + (wB + nC) * dist(B, C) - nC * dist(A, B)) / (wA + wB + nC)
        const auto wC = this->dendrogram[aux.matrixToNode[col]].weight;
        auto distAC = distance(leftIndex, col, aux);
        auto distBC = distance(rightIndex, col, aux);
        aux.matrix[matrixIndex] = LP::distance(distAC, wA, distBC, wB, distAB, wC);
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
    this->dendrogram[clusterRow].matrixIndex = matrixRow;
    this->dendrogram[clusterRow].weight = wAB;
    this->dendrogram[clusterRow].count = this->dendrogram[leftId].count + this->dendrogram[rightId].count;
    this->dendrogram[clusterRow].mergeDistance = std::clamp(distAB, 0.f, 2.f);
    this->dendrogram[clusterRow].leftChild = leftId;
    this->dendrogram[clusterRow].rightChild = rightId;
    this->dendrogram[clusterRow].mustLink = mustLink;
    --aux.numClustersRemaining;
    
    if (!mustLink) {
        // Compute gap
        auto childDist = std::max(this->dendrogram[leftId].mergeDistance, this->dendrogram[rightId].mergeDistance);
        auto gap = distAB - childDist;
        if (gap > aux.maxGap) {
            aux.maxGap = gap;
            this->elbowLinkage = std::clamp((distAB + childDist) / 2.f, 0.f, 2.f);
        }
    }
    
    return matrixRow;
}

template<LinkagePolicy LP>
std::ptrdiff_t Dendrogram<LP>::nearestNeighbor(std::ptrdiff_t index, Aux& aux) {
    if (index < 0 || index >= aux.size || !aux.activeFlags[index]) {
        return -1;
    }

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

template class Dendrogram<WardLinkage>;
template class Dendrogram<CosineAverageLinkage>;
