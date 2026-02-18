#pragma once
#include "SpeakerEmbeddingWrapper.hpp"
#include "Cluster.hpp"
#include <Accelerate/Accelerate.h>
#include <concepts>
#include <vector>
#include <iostream>

template<typename T>
concept LinkagePolicy = requires(
        const SpeakerEmbeddingWrapper& a, const SpeakerEmbeddingWrapper& b,
        float distAC, float wA, float distBC, float wB, float distAB, float wC,
        const Cluster& cluster, const SpeakerEmbeddingWrapper* embeddings, long count, UUIDWrapper id,
        std::vector<UUIDWrapper>&& segmentIds
) {
    { T::distance(distAC, wA, distBC, wB, distAB, wC) } -> std::convertible_to<float>;
    { T::distance(a, b) } -> std::convertible_to<float>;
    { T::computeCentroid(embeddings, cluster) } -> std::convertible_to<SpeakerEmbeddingWrapper>;
    { T::computeCentroid(id, embeddings, count, std::move(segmentIds)) } -> std::convertible_to<SpeakerEmbeddingWrapper>;
};

namespace {
    template<bool ShouldNormalizeWeights, bool ShouldNormalizeVector>
    SpeakerEmbeddingWrapper computeCentroidHelper(const SpeakerEmbeddingWrapper *embeddings, const Cluster &cluster);

    template<bool ShouldNormalizeWeights, bool ShouldNormalizeVector>
    SpeakerEmbeddingWrapper computeCentroidHelper(
            UUIDWrapper id,
            const SpeakerEmbeddingWrapper *embeddings,
            long count,
            std::vector<UUIDWrapper> &&segmentIds
    );
    
    constexpr bool NormalizeWeights = true;
    constexpr bool KeepRawWeights = false;
    constexpr bool NormalizeCentroid = true;
    constexpr bool KeepRawCentroid = false;
}

struct WardLinkage {
    static inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) {
        return ((wA + wC) * distAC + (wB + wC) * distBC - wC * distAB) / (wA + wB + wC);
    }

    static inline float distance(const SpeakerEmbeddingWrapper& a, const SpeakerEmbeddingWrapper& b) {
        return a.wardDistanceTo(b);
    }

    static inline SpeakerEmbeddingWrapper computeCentroid(const SpeakerEmbeddingWrapper* embeddings, const Cluster& cluster) {
        return computeCentroidHelper<KeepRawWeights, NormalizeCentroid>(embeddings, cluster);
    }
    
    static inline SpeakerEmbeddingWrapper computeCentroid(
            UUIDWrapper id,
            const SpeakerEmbeddingWrapper* embeddings, 
            long count, 
            std::vector<UUIDWrapper>&& segmentIds
    ) {
        return computeCentroidHelper<KeepRawWeights, NormalizeCentroid>(id, embeddings, count, std::move(segmentIds));
    }
};


struct CosineAverageLinkage {
    static inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) {
        return (wA * distAC + wB * distBC) / (wA + wB);
    }

    static inline float distance(const SpeakerEmbeddingWrapper& a, const SpeakerEmbeddingWrapper& b) {
        return a.unitCosineDistanceTo(b);
    }

    static inline SpeakerEmbeddingWrapper computeCentroid(const SpeakerEmbeddingWrapper* embeddings, const Cluster& cluster) {
        return computeCentroidHelper<NormalizeWeights, KeepRawCentroid>(embeddings, cluster);
    }

    static inline SpeakerEmbeddingWrapper computeCentroid(
            UUIDWrapper id,
            const SpeakerEmbeddingWrapper* embeddings,
            long count,
            std::vector<UUIDWrapper>&& segmentIds
    ) {
        return computeCentroidHelper<NormalizeWeights, KeepRawCentroid>(id, embeddings, count, std::move(segmentIds));
    }
};

struct WeightedAverageLinkage {
    static inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) {
        return std::max(distAC, distBC);
    }

    static inline float distance(const SpeakerEmbeddingWrapper& a, const SpeakerEmbeddingWrapper& b) {
        return a.cosineDistanceTo(b); //(b) / 2;
    }

    static inline SpeakerEmbeddingWrapper computeCentroid(const SpeakerEmbeddingWrapper* embeddings, const Cluster& cluster) {
        return computeCentroidHelper<NormalizeWeights, KeepRawCentroid>(embeddings, cluster);
    }

    static inline SpeakerEmbeddingWrapper computeCentroid(
            UUIDWrapper id,
            const SpeakerEmbeddingWrapper* embeddings,
            long count,
            std::vector<UUIDWrapper>&& segmentIds
    ) {
        return computeCentroidHelper<NormalizeWeights, KeepRawCentroid>(id, embeddings, count, std::move(segmentIds));
    }
};

struct WeightedCosineAverageLinkage {
    static inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) {
        return 0.625 * distAC + 0.625 * distBC - 0.25 * distAB;
    }

    static inline float distance(const SpeakerEmbeddingWrapper& a, const SpeakerEmbeddingWrapper& b) {
        return a.cosineDistanceTo(b);
    }

    static inline SpeakerEmbeddingWrapper computeCentroid(const SpeakerEmbeddingWrapper* embeddings, const Cluster& cluster) {
        return computeCentroidHelper<NormalizeWeights, KeepRawCentroid>(embeddings, cluster);
    }

    static inline SpeakerEmbeddingWrapper computeCentroid(
            UUIDWrapper id,
            const SpeakerEmbeddingWrapper* embeddings,
            long count,
            std::vector<UUIDWrapper>&& segmentIds
    ) {
        return computeCentroidHelper<NormalizeWeights, KeepRawCentroid>(id, embeddings, count, std::move(segmentIds));
    }
};

namespace {
    template<bool ShouldNormalizeWeights, bool ShouldNormalizeVector>
    SpeakerEmbeddingWrapper computeCentroidHelper(const SpeakerEmbeddingWrapper *embeddings, const Cluster &cluster) {
        if (cluster.count() == 0) return {};
        if (cluster.count() == 1) return embeddings[cluster[0]];

        auto iter = cluster.begin();
        const auto &e0 = embeddings[*(iter++)];
        const auto &e1 = embeddings[*(iter++)];

        auto centroid = SpeakerEmbeddingWrapper(e0.id() + e1.id(), cluster.weight());

        float normalizer, totalWeight, w0, w1, weight;
        if constexpr (ShouldNormalizeWeights) {
            normalizer = 1.f / cluster.weight();
            w0 = e0.weight() * normalizer;
            w1 = e1.weight() * normalizer;
        } else {
            w0 = e0.weight();
            w1 = e1.weight();
        }

        // Centroid = emb0 * w0 + emb1 * w1
        vDSP_vsmsma(
                e0.vector(), 1, &w0,
                e1.vector(), 1, &w1,
                centroid.vector(), 1,
                SpeakerEmbeddingWrapper::dims
        );
        
        // Track segment IDs
        auto& segmentIds = centroid.segmentIds();
        segmentIds.reserve(cluster.segmentCount());
        segmentIds.insert(segmentIds.end(), e0.segmentIds().begin(), e0.segmentIds().end());
        segmentIds.insert(segmentIds.end(), e1.segmentIds().begin(), e1.segmentIds().end());
        
        for (; iter != cluster.end(); ++iter) {
            const auto &embedding = embeddings[*iter];
            if constexpr (ShouldNormalizeWeights)
                weight = embedding.weight() * normalizer;
            else
                weight = embedding.weight();
            
            // Centroid <- emb * w + centroid
            vDSP_vsma(
                    embedding.vector(), 1, &weight,
                    centroid.vector(), 1,
                    centroid.vector(), 1,
                    SpeakerEmbeddingWrapper::dims
            );
            
            centroid.id() += embedding.id();
            
            // Update segment IDs
            auto& newSegmentIds = embedding.segmentIds();
            segmentIds.insert(segmentIds.end(), newSegmentIds.begin(), newSegmentIds.end());
        }
        
        std::sort(segmentIds.begin(), segmentIds.end());
        segmentIds.erase(std::unique(segmentIds.begin(), segmentIds.end()), segmentIds.end());

        if constexpr (ShouldNormalizeVector) {
            return centroid.normalizedInPlace();
        } else {
            return centroid;
        }
    }

    template<bool ShouldNormalizeWeights, bool ShouldNormalizeVector>
    SpeakerEmbeddingWrapper computeCentroidHelper(
            UUIDWrapper id,
            const SpeakerEmbeddingWrapper *embeddings,
            long count,
            std::vector<UUIDWrapper> &&segmentIds
    ) {
        if (count == 0) return {};
        if (count == 1) {
            auto result = embeddings[0];
            result.segmentIds() = std::move(segmentIds);
            return result;
        }
        
        float normalizer, totalWeight, w0, w1, weight;
        if constexpr (ShouldNormalizeWeights) {
            totalWeight = 0;
            for (auto i = 0; i < count; ++i)
                totalWeight += embeddings[i].weight();
            normalizer = 1.f / totalWeight;
            w0 = embeddings[0].weight() * normalizer;
            w1 = embeddings[1].weight() * normalizer;
        } else {
            w0 = embeddings[0].weight();
            w1 = embeddings[1].weight();
            totalWeight = w0 + w1;
        }

        auto centroid = SpeakerEmbeddingWrapper(id, totalWeight, std::move(segmentIds));
        
        // Centroid = emb0 * w0 + emb1 * w1 for the first two embeddings 
        vDSP_vsmsma(
                embeddings[0].vector(), 1, &w0,
                embeddings[1].vector(), 1, &w1,
                centroid.vector(), 1,
                SpeakerEmbeddingWrapper::dims
        );

        for (auto i = 2; i < count; ++i) {
            const auto &embedding = embeddings[i];
            if constexpr (ShouldNormalizeWeights) {
                weight = embedding.weight() * normalizer;
            } else {
                weight = embedding.weight();
                centroid.weight() += weight;
            }
            
            // Centroid <- emb * w + centroid
            vDSP_vsma(
                    embedding.vector(), 1, &weight,
                    centroid.vector(), 1,
                    centroid.vector(), 1,
                    SpeakerEmbeddingWrapper::dims
            );
        }

        if constexpr (ShouldNormalizeVector) {
            return centroid.normalizedInPlace();
        } else {
            return centroid;
        }
    }
}
