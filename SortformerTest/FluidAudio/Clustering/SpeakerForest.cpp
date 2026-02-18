//
// Created by Benjamin Lee on 1/29/26.
//

#include "SpeakerForest.hpp"
#include "LinkagePolicy.hpp"
#include <ranges>

template<LinkagePolicy LP>
SpeakerForest<LP>::SpeakerForest(std::shared_ptr<ClusteringConfig> config):
    config(std::move(config)) {}

template<LinkagePolicy LP>
SpeakerForest<LP>::SpeakerForest(
        std::shared_ptr<ClusteringConfig> config,
        EmbeddingDistanceMatrix<LP>&& matrix,
        std::vector<ClusterRepresentative> representatives
): config(std::move(config)), matrix(std::move(matrix)), representatives(std::move(representatives)) {
    this->dendrogram = matrix.dendrogram();
}

template<LinkagePolicy LP>
void SpeakerForest<LP>::streamEmbeddingSegments(std::vector<EmbeddingSegmentWrapper<LP>>& newFinalized,
                                                std::vector<EmbeddingSegmentWrapper<LP>>& newTentative) {
    this->matrix.stream(newFinalized, newTentative);
    this->dendrogram = matrix.dendrogram();
    
    auto oldRepresentatives = std::move(this->representatives);
    this->updateRepresentatives();
    this->updateOutliers(oldRepresentatives, this->config->mergeThreshold);
    
//    if (this->matrix.finalizedCount() > this->config->maxEmbeddings) {
//        this->compress(this->config->minEmbeddings);
//    }
}

template<LinkagePolicy LP>
void SpeakerForest<LP>::finalize() {
    this->compress(this->config->maxRepresentatives, config->minSeparation, false);
}

template<LinkagePolicy LP>
void SpeakerForest<LP>::compress(long maxSize, float linkageThreshold, bool keepTentative) {
    std::vector<Cluster> clusters;
    if (keepTentative) {
        clusters = this->matrix.dendrogram(true).extractClusters(linkageThreshold, maxSize);
    } else {
        clusters = dendrogram.extractClusters(linkageThreshold, maxSize);
    }
    
    const auto embeddings = matrix.embeddings;
    auto newMatrix = EmbeddingDistanceMatrix<LP>();
    newMatrix.reserve((long)clusters.size() * 2);
    
    // Add compressed clusters
    for (auto& cluster: clusters) {
        auto centroid = LP::computeCentroid(embeddings, cluster);
        newMatrix.insert(centroid);
    }
    
    if (keepTentative) {
        // Add the tentative embeddings back
        for (auto [_, index]: matrix.tentativeIndices) {
            newMatrix.insert(embeddings[index], true);
        }
    }
    
    // Re-cluster
    this->matrix = std::move(newMatrix);
    this->dendrogram = this->matrix.dendrogram();
}

template<LinkagePolicy LP>
SpeakerForest<LP> SpeakerForest<LP>::isolateOutliers() {
    std::vector<long> leafIndices{};
    std::vector<ClusterRepresentative> isolatedRepresentatives;
    leafIndices.reserve(matrix.embeddingCount());
    isolatedRepresentatives.reserve(outlierRepresentativeIndices.size());
    
    for (auto index: outlierRepresentativeIndices | std::views::reverse) {
        const auto& cluster = representatives[index].cluster;
        leafIndices.insert(leafIndices.end(), cluster.begin(), cluster.end());
        isolatedRepresentatives.emplace_back(std::move(representatives[index]));
        representatives.erase(representatives.begin() + index);
    }
    
    auto resultMatrix = matrix.gatherAndPop(leafIndices);
    return {config, std::move(resultMatrix), isolatedRepresentatives};
}

template<LinkagePolicy LP>
void SpeakerForest<LP>::updateRepresentatives() {
    auto clusters = dendrogram.extractClusters(config->minSeparation, config->maxRepresentatives);
    representatives.clear();
    representatives.reserve(clusters.size());

    for (auto& cluster: clusters) {
        representatives.emplace_back(LP::computeCentroid(matrix.embeddings, cluster), std::move(cluster));
    }
}

template<LinkagePolicy LP>
float SpeakerForest<LP>::distanceTo(const SpeakerForest<LP>& other) const {
    const auto& repsA = this->representatives;
    const auto& repsB = other.representatives;
    
    std::vector<float> bestB(repsB.size(), std::numeric_limits<float>::max());
    float sumA = 0;
    for (const auto &[embA, _] : repsA) {
        auto minA = std::numeric_limits<float>::max();
        for (int j = 0; j < repsB.size(); ++j) {
            auto dist = LP::distance(embA, repsB[j].centroid);
            minA = std::min(minA, dist);
            bestB[j] = std::min(bestB[j], dist);
        }
        sumA += minA;
    }
    float sumB = 0;
    for (auto dist: bestB) sumB += dist;
    return (sumA / float(repsA.size()) + sumB / float(repsB.size())) / 2;
}

template<LinkagePolicy LP>
void SpeakerForest<LP>::updateOutliers(std::vector<ClusterRepresentative> const& oldRepresentatives, float linkageThreshold) {
    outlierRepresentativeIndices.clear();

    for (long i = 0; i < representatives.size(); ++i) {
        bool isOutlier = true;
        const auto& embA = representatives[i].centroid;
        
        // If there is at least one old representative that is close enough, it's not an outlier 
        for (auto &[embB, _]: oldRepresentatives) {
            if (LP::distance(embA, embB) <= linkageThreshold) {
                isOutlier = false;
                break;
            }
        }
        if (isOutlier)
            outlierRepresentativeIndices.push_back(i);
    }
}

template<LinkagePolicy LP>
void SpeakerForest<LP>::absorb(const SpeakerForest& other) {
    this->matrix.absorb(other.matrix);
    this->dendrogram = this->matrix.dendrogram(false);
    this->updateRepresentatives();
}

template class SpeakerForest<WardLinkage>;
template class SpeakerForest<CosineAverageLinkage>;
template class SpeakerForest<WeightedAverageLinkage>;
template class SpeakerForest<WeightedCosineAverageLinkage>;
