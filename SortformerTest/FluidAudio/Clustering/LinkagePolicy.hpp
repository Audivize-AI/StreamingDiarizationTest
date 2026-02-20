#pragma once
#include "SpeakerEmbeddingWrapper.hpp"
#include "Cluster.hpp"
#include <Accelerate/Accelerate.h>
#include <vector>

class WardLinkage;
class UPGMALinkage;
class CompleteLinkage;
class SingleLinkage;

enum class LinkagePolicyType {
    wardLinkage,
    completeLinkage,
    singleLinkage,
    upgma,
};

class EmbeddingDistanceMatrix;

class LinkagePolicy {
public:
    static const WardLinkage wardLinkage;
    static const UPGMALinkage upgmaLinkage;
    static const CompleteLinkage completeLinkage;
    static const SingleLinkage singleLinkage;
    
    [[nodiscard]] virtual LinkagePolicyType getPolicyType() const = 0;
    
    [[nodiscard]] virtual float distance(float distAC, float wA,
                                         float distBC, float wB,
                                         float distAB, float wC) const = 0;
    
    [[nodiscard]] virtual float distance(const SpeakerEmbeddingWrapper &a,
                                         const SpeakerEmbeddingWrapper &b) const = 0;
    
    [[nodiscard]] virtual SpeakerEmbeddingWrapper computeCentroid(const SpeakerEmbeddingWrapper *embeddings,
                                                                  const Cluster &cluster) const = 0;
    
    [[nodiscard]] virtual SpeakerEmbeddingWrapper computeCentroid(UUIDWrapper id,
                                                                  const SpeakerEmbeddingWrapper *embeddings,
                                                                  long count,
                                                                  std::vector<UUIDWrapper> &&segmentIds) const = 0;
    
    [[nodiscard]] static const LinkagePolicy *getPolicy(LinkagePolicyType forType);
    
    [[nodiscard]] static float distance(const LinkagePolicy* policy,
                                        const SpeakerEmbeddingWrapper &a,
                                        const SpeakerEmbeddingWrapper &b) {
        return policy->distance(a, b);
    }
    
    [[nodiscard]] static SpeakerEmbeddingWrapper computeCentroid(const LinkagePolicy* policy,
                                                                 EmbeddingDistanceMatrix matrix,
                                                                 const Cluster &cluster);
    
protected:
    enum class NormalizeBy {
        weight,
        l2Norm
    };
    
    template<NormalizeBy Normalization>
    static SpeakerEmbeddingWrapper computeCentroidHelper(const SpeakerEmbeddingWrapper *embeddings, const Cluster &cluster);
    
    template<NormalizeBy Normalization>
    static SpeakerEmbeddingWrapper computeCentroidHelper(UUIDWrapper id, const SpeakerEmbeddingWrapper *embeddings, long count, std::vector<UUIDWrapper> &&segmentIds);
};

class WardLinkage: public LinkagePolicy {
public:
    [[nodiscard]] LinkagePolicyType getPolicyType() const final {
        return LinkagePolicyType::wardLinkage;
    }
    
    [[nodiscard]] inline float
    distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) const final {
        return ((wA + wC) * distAC + (wB + wC) * distBC - wC * distAB) / (wA + wB + wC);
    }

    [[nodiscard]] inline float
    distance(const SpeakerEmbeddingWrapper &a, const SpeakerEmbeddingWrapper &b) const final {
        return a.wardDistanceTo(b);
    }

    [[nodiscard]] inline SpeakerEmbeddingWrapper
    computeCentroid(const SpeakerEmbeddingWrapper *embeddings, const Cluster &cluster) const final {
        return computeCentroidHelper<NormalizeBy::l2Norm>(embeddings, cluster);
    }

    [[nodiscard]] inline SpeakerEmbeddingWrapper computeCentroid(UUIDWrapper id, const SpeakerEmbeddingWrapper *embeddings, long count, std::vector<UUIDWrapper> &&segmentIds) const final {
        return computeCentroidHelper<NormalizeBy::l2Norm>(id, embeddings, count, std::move(segmentIds));
    }
};

class UPGMALinkage: public LinkagePolicy {
public:
    [[nodiscard]] LinkagePolicyType getPolicyType() const final {
        return LinkagePolicyType::upgma;
    }
    
    [[nodiscard]] inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) const final {
        return (wA * distAC + wB * distBC) / (wA + wB);
    }

    [[nodiscard]] inline float distance(const SpeakerEmbeddingWrapper& a, const SpeakerEmbeddingWrapper& b) const final {
        return a.unitCosineDistanceTo(b);
    }

    [[nodiscard]] inline SpeakerEmbeddingWrapper computeCentroid(const SpeakerEmbeddingWrapper* embeddings, const Cluster& cluster) const final {
        return computeCentroidHelper<NormalizeBy::weight>(embeddings, cluster);
    }

    [[nodiscard]] inline SpeakerEmbeddingWrapper computeCentroid(UUIDWrapper id, const SpeakerEmbeddingWrapper *embeddings, long count, std::vector<UUIDWrapper> &&segmentIds) const final {
        return computeCentroidHelper<NormalizeBy::weight>(id, embeddings, count, std::move(segmentIds));
    }
};

class CompleteLinkage: public LinkagePolicy {
public:
    [[nodiscard]] LinkagePolicyType getPolicyType() const final {
        return LinkagePolicyType::completeLinkage;
    }
    
    [[nodiscard]] inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) const final {
        return std::max(distAC, distBC);
    }

    [[nodiscard]] inline float distance(const SpeakerEmbeddingWrapper& a, const SpeakerEmbeddingWrapper& b) const final {
        return a.unitCosineDistanceTo(b);
    }

    [[nodiscard]] inline SpeakerEmbeddingWrapper computeCentroid(const SpeakerEmbeddingWrapper* embeddings, const Cluster& cluster) const final {
        return computeCentroidHelper<NormalizeBy::l2Norm>(embeddings, cluster);
    }

    [[nodiscard]] inline SpeakerEmbeddingWrapper computeCentroid(UUIDWrapper id, const SpeakerEmbeddingWrapper *embeddings, long count, std::vector<UUIDWrapper> &&segmentIds) const final {
        return computeCentroidHelper<NormalizeBy::l2Norm>(id, embeddings, count, std::move(segmentIds));
    }
};

class SingleLinkage: public LinkagePolicy {
public:
    [[nodiscard]] LinkagePolicyType getPolicyType() const final {
        return LinkagePolicyType::singleLinkage;
    }
    
    [[nodiscard]] inline float distance(float distAC, float wA, float distBC, float wB, float distAB, float wC) const final {
        return std::min(distAC, distBC);
    }

    [[nodiscard]] inline float distance(const SpeakerEmbeddingWrapper& a, const SpeakerEmbeddingWrapper& b) const final {
        return a.unitCosineDistanceTo(b);
    }

    [[nodiscard]] inline SpeakerEmbeddingWrapper computeCentroid(const SpeakerEmbeddingWrapper* embeddings, const Cluster& cluster) const final {
        return computeCentroidHelper<NormalizeBy::l2Norm>(embeddings, cluster);
    }

    [[nodiscard]] inline SpeakerEmbeddingWrapper computeCentroid(UUIDWrapper id, const SpeakerEmbeddingWrapper *embeddings, long count, std::vector<UUIDWrapper> &&segmentIds) const final {
        return computeCentroidHelper<NormalizeBy::l2Norm>(id, embeddings, count, std::move(segmentIds));
    }
};

