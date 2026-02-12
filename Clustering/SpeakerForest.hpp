//
// Created by Benjamin Lee on 1/29/26.
//

#pragma once
#include "EmbeddingDistanceMatrix.hpp"
#include "Dendrogram.hpp"
#include "ClusterLinkage.hpp"

// TODO: REFACTOR THIS SO THAT IT DOES OUTLIER DETECTION WHEN A NEW CLUSTER APPEARS 
//  THAT DOESN'T MATCH THE OLD REPRESENTATIVES
//template<LinkagePolicy LP>
class SpeakerForest {
public:
    using LP = CosineAverageLinkage;

    struct DendrogramSnapshotNode {
        long matrixIndex{-1};
        long leftChild{-1};
        long rightChild{-1};
        long count{1};
        float weight{1};
        float mergeDistance{0};
        bool mustLink{false};
        long speakerIndex{-1};
    };

    struct DendrogramSnapshot {
        long rootIndex{-1};
        long nodeCount{0};
        long activeLeafCount{0};
        std::vector<DendrogramSnapshotNode> nodes{};
    };
    
    explicit SpeakerForest(long numRepresentatives, long minEmbeddings, long maxEmbeddings);
    
    /** 
     * @brief Compress the embeddings into cluster centroids
     * @param maxSize Maximum number of embeddings to keep
     * @param mergeThreshold Minimum linkage distance to split a cluster
     * @param keepTentative Whether to preserve the tentative embeddings. If false, they will be compressed.
     */
    void compress(long maxSize, float mergeThreshold = -1.f, bool keepTentative=true);

    /** 
     * @brief Finalize tentative embeddings and compress the embeddings into representative centroids
     */
    void finalize();

    void streamEmbeddingSegments(std::vector<EmbeddingSegment>& newFinalized, 
                                 std::vector<EmbeddingSegment>& newTentative);

    [[nodiscard]] DendrogramSnapshot dendrogramSnapshot() const;
    
    [[nodiscard]] static float chamferDistance(const std::vector<Embedding>& r1, const std::vector<Embedding>& r2);
private:
    EmbeddingDistanceMatrix<LP> matrix;
    Dendrogram<LP> dendrogram;
    std::vector<Embedding> oldRepresentatives;
    std::vector<Embedding> representatives;
    long maxEmbeddings;
    long minEmbeddings;
    long maxRepresentatives;
    long numClusters;
    float minSeparation = 0.1;
    float mergeThreshold = 0.3;

    [[nodiscard]] std::vector<Embedding> generateRepresentatives() const;
};
