//
// Created by Benjamin Lee on 1/29/26.
//

#pragma once
#include "EmbeddingDistanceMatrix.hpp"
#include "Dendrogram.hpp"
#include "LinkagePolicy.hpp"
#include "ClusteringConfig.hpp"

struct ClusterRepresentative {
    SpeakerEmbeddingWrapper centroid;
    Cluster cluster;
    
    ClusterRepresentative(SpeakerEmbeddingWrapper&& embedding, Cluster&& cluster) : 
            centroid(std::move(embedding)), cluster(std::move(cluster)) {}

    ClusterRepresentative(const ClusterRepresentative& other) noexcept = default;
    
    ClusterRepresentative(ClusterRepresentative&& other) noexcept :
            centroid(std::move(other.centroid)), cluster(std::move(other.cluster)) {}

    ClusterRepresentative& operator=(ClusterRepresentative&& other) noexcept {
        centroid = std::move(other.centroid);
        cluster = std::move(other.cluster);
        return *this;
    }

    ClusterRepresentative& operator=(const ClusterRepresentative& other) noexcept = default;
};

template<LinkagePolicy LP>
class SpeakerForest {
private:
    EmbeddingDistanceMatrix<LP> matrix;
    Dendrogram<LP> dendrogram;
    std::vector<ClusterRepresentative> representatives;
    std::vector<long> outlierRepresentativeIndices;
    std::vector<UUIDWrapper> tentativeSortformerSegmentIds;
    std::vector<UUIDWrapper> sortformerSegmentIds;
    long numClusters{0};

    std::shared_ptr<ClusteringConfig> config;

    SpeakerForest(std::shared_ptr<ClusteringConfig> config,
                  EmbeddingDistanceMatrix<LP>&& matrix,
                  std::vector<ClusterRepresentative> representatives);

    void updateRepresentatives();
    void updateOutliers(std::vector<ClusterRepresentative> const& oldRepresentatives, float linkageThreshold);
    
public:
    explicit SpeakerForest(std::shared_ptr<ClusteringConfig> config);
    
    /** 
     * @brief Compress the embeddings into cluster centroids
     * @param maxSize Maximum number of embeddings to keep
     * @param linkageThreshold Minimum linkage distance to split a cluster
     * @param keepTentative Whether to preserve the tentative embeddings. If false, they will be compressed.
     */
    void compress(long maxSize, float linkageThreshold = -1.f, bool keepTentative=true);

    /** 
     * @brief Finalize tentative embeddings and compress the embeddings into representative centroids
     */
    void finalize();

    void streamEmbeddingSegments(std::vector<EmbeddingSegmentWrapper<LP>>& newFinalized, 
                                 std::vector<EmbeddingSegmentWrapper<LP>>& newTentative);
    
    SpeakerForest isolateOutliers();
    
    void absorb(const SpeakerForest& other);
    
    [[nodiscard]] float distanceTo(const SpeakerForest& other) const;

    // Expose read-only clustering state for bridge snapshot serialization.
    [[nodiscard]] inline const EmbeddingDistanceMatrix<LP>& distanceMatrix() const { return matrix; }
    [[nodiscard]] inline const Dendrogram<LP>& currentDendrogram() const { return dendrogram; }
};
