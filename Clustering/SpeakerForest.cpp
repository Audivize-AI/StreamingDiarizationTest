//
// Created by Benjamin Lee on 1/29/26.
//

#include "SpeakerForest.hpp"
#include "ClusterLinkage.hpp"
#include <algorithm>
#include <cmath>

SpeakerForest::SpeakerForest(long numRepresentatives, long minEmbeddings, long maxEmbeddings):
    maxRepresentatives(numRepresentatives), minEmbeddings(minEmbeddings), maxEmbeddings(maxEmbeddings) {}

void SpeakerForest::streamEmbeddingSegments(std::vector<EmbeddingSegment>& newFinalized,
                                            std::vector<EmbeddingSegment>& newTentative) {
    this->matrix.stream(newFinalized, newTentative);
    this->dendrogram = Dendrogram<LP>(this->matrix);
    this->oldRepresentatives = std::move(this->representatives);
    this->representatives = generateRepresentatives();
    if (this->matrix.finalizedCount() > this->maxEmbeddings) {
        this->compress(this->minEmbeddings);
    }
}

SpeakerForest::DendrogramSnapshot SpeakerForest::dendrogramSnapshot() const {
    DendrogramSnapshot snapshot{};
    snapshot.rootIndex = this->dendrogram.rootIndex();
    snapshot.nodeCount = this->dendrogram.nodeCount();
    snapshot.activeLeafCount = this->matrix.embeddingCount();

    if (snapshot.nodeCount <= 0 || snapshot.rootIndex < 0) {
        return snapshot;
    }

    auto dendrogramNodes = this->dendrogram.nodes();
    if (!dendrogramNodes) {
        snapshot.rootIndex = -1;
        snapshot.nodeCount = 0;
        snapshot.activeLeafCount = 0;
        return snapshot;
    }

    snapshot.nodes.reserve(static_cast<size_t>(snapshot.nodeCount));
    for (long i = 0; i < snapshot.nodeCount; ++i) {
        const auto &node = dendrogramNodes[i];
        DendrogramSnapshotNode snapshotNode{};
        snapshotNode.matrixIndex = node.matrixIndex;
        snapshotNode.leftChild = node.leftChild;
        snapshotNode.rightChild = node.rightChild;
        snapshotNode.count = node.count;
        snapshotNode.weight = node.weight;
        if (!std::isfinite(node.mergeDistance)) {
            snapshotNode.mergeDistance = 2.f;
        } else {
            snapshotNode.mergeDistance = std::clamp(node.mergeDistance, 0.f, 2.f);
        }
        snapshotNode.mustLink = node.mustLink;

        const bool isLeaf = node.leftChild < 0 || node.rightChild < 0;
        if (isLeaf && node.matrixIndex >= 0 && node.matrixIndex < this->matrix.size()) {
            snapshotNode.speakerIndex = static_cast<long>(this->matrix.embeddings[node.matrixIndex].speakerId());
        }

        snapshot.nodes.push_back(snapshotNode);
    }
    return snapshot;
}

void SpeakerForest::finalize() {
    this->compress(this->maxRepresentatives, minSeparation, false);
}

void SpeakerForest::compress(long maxSize, float mergeThreshold, bool keepTentative) {
    std::vector<Cluster> clusters;
    if (keepTentative) {
        clusters = Dendrogram<LP>(this->matrix, true)
                .extractClusters(mergeThreshold, maxSize);
    } else {
        clusters = dendrogram.extractClusters(mergeThreshold, maxSize);
    }
    
    const auto embeddings = matrix.embeddings;
    auto newMatrix = EmbeddingDistanceMatrix<LP>();
    newMatrix.reserve((long)clusters.size() * 2);
    
    // Add compressed clusters
    for (auto& cluster: clusters) {
        auto centroid = LP::combine(cluster, embeddings);
        newMatrix.insert(centroid);
    }
    
    if (keepTentative) {
        // Add the tentative embeddings back
        for (auto index: matrix.tentativeIndices) {
            newMatrix.insert(embeddings[index], true);
        }
    }
    
    // Re-cluster
    this->matrix = std::move(newMatrix);
    this->dendrogram = Dendrogram<LP>(this->matrix);
}

std::vector<Embedding> SpeakerForest::generateRepresentatives() const {
    std::vector<Cluster> clusters = dendrogram.extractClusters(minSeparation, maxRepresentatives);
    std::vector<Embedding> newRepresentatives;
    newRepresentatives.reserve(maxRepresentatives);
    const auto embeddings = matrix.embeddings;

    // Add compressed clusters
    for (auto& cluster: clusters) {
        newRepresentatives.emplace_back(LP::combine(cluster, embeddings));
    }
    
    return newRepresentatives;
}


float SpeakerForest::chamferDistance(const std::vector<Embedding>& r1, const std::vector<Embedding>& r2) {
    std::vector<float> best2(r2.size(), std::numeric_limits<float>::max());
    float sum1 = 0;
    for (const auto & e1 : r1) {
        auto min1 = std::numeric_limits<float>::max();
        for (int j = 0; j < r2.size(); ++j) {
            auto dist = LP::distance(e1, r2[j]);
            min1 = std::min(min1, dist);
            best2[j] = std::min(best2[j], dist);
        }
        sum1 += min1;
    }
    float sum2 = 0;
    for (auto d: best2) sum2 += d;
    return (sum1 / float(r1.size()) + sum2 / float(r2.size())) / 2;
}
