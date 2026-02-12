//
// Created by Benjamin Lee on 1/30/26.
//

#pragma once
#include "Embedding.hpp"
#include "Cluster.hpp"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <cmath>
#include <concepts>

template<typename T>
concept LinkagePolicy = requires(
        const Embedding& a, const Embedding& b, 
        float distAC, float wA, float distBC, float wB, float distAB, float wC,
        const Cluster& cluster, Embedding* embeddings, long count
) {
    { T::distance(distAC, wA, distBC, wB, distAB, wC) } -> std::convertible_to<float>;
    { T::distance(a, b) } -> std::convertible_to<float>;
    { T::combine(cluster, embeddings) } -> std::convertible_to<Embedding>;
};

struct WardLinkage {
    static inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) {
        return ((wA + wC) * distAC + (wB + wC) * distBC - wC * distAB) / (wA + wB + wC);
    }

    static inline float distance(const Embedding& a, const Embedding& b) {
        auto wA = a.weight(), wB = b.weight();
        return (wA * wB) / (wA + wB) * a.squaredDistanceTo(b); 
    }
    
    static Embedding combine(const Cluster& cluster, Embedding* embeddings) {
        if (cluster.count() == 0) return {};

        const auto& firstEmb = embeddings[cluster[0]];
        auto result = firstEmb * firstEmb.weight();
        
        for (auto iter = ++cluster.begin(); iter != cluster.end(); ++iter) {
            const auto& embedding = embeddings[*iter];
            auto weight = embedding.weight(); 
            
            vDSP_vsma(
                    embedding.vector().get(), 1,
                    &weight,
                    result.vector().get(), 1,
                    result.vector().get(), 1,
                    Embedding::dims
            );
        }
        
        result.setSpread(cluster.spread());
        result.setWeight(cluster.weight());
        return result.normalize();
    }
};


struct CosineAverageLinkage {
    static inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) {
        (void)distAB;
        (void)wC;
        const auto denom = wA + wB;
        if (!std::isfinite(distAC) || !std::isfinite(distBC) || !std::isfinite(denom) || denom <= 1e-6f) {
            return 2.f;
        }
        const auto merged = (wA * distAC + wB * distBC) / denom;
        return std::clamp(merged, 0.f, 2.f);
    }

    static inline float distance(const Embedding& a, const Embedding& b) {
        const auto dist = a.unitCosineDistanceTo(b);
        if (!std::isfinite(dist)) {
            return 2.f;
        }
        return std::clamp(dist, 0.f, 2.f);
    }

    static Embedding combine(const Cluster& cluster, Embedding* embeddings) {
        if (cluster.count() == 0) return {};
        const auto clusterWeight = cluster.weight();
        if (!std::isfinite(clusterWeight) || clusterWeight <= 1e-6f) return {};

        const auto normalizer = 1.f / clusterWeight;

        const auto& firstEmb = embeddings[cluster[0]];
        auto result = firstEmb * (firstEmb.weight() * normalizer);

        for (auto iter = ++cluster.begin(); iter != cluster.end(); ++iter) {
            const auto& embedding = embeddings[*iter];
            auto weight = embedding.weight() * normalizer;

            vDSP_vsma(
                    embedding.vector().get(), 1,
                    &weight,
                    result.vector().get(), 1,
                    result.vector().get(), 1,
                    Embedding::dims
            );
        }
        result.setSpread(cluster.spread());
        result.setWeight(clusterWeight);
        return result.normalize();
    }
};
