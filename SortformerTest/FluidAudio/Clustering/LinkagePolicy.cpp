#include "LinkagePolicy.hpp"
#include "EmbeddingDistanceMatrix.hpp"

const UPGMALinkage LinkagePolicy::upgmaLinkage{};
const WardLinkage LinkagePolicy::wardLinkage{};
const CompleteLinkage LinkagePolicy::completeLinkage{};
const SingleLinkage LinkagePolicy::singleLinkage{};

const LinkagePolicy* LinkagePolicy::getPolicy(LinkagePolicyType forType) {
    switch (forType) {
        case LinkagePolicyType::wardLinkage:
            return &LinkagePolicy::wardLinkage;
        case LinkagePolicyType::upgma:
            return &LinkagePolicy::upgmaLinkage;
        case LinkagePolicyType::singleLinkage:
            return &LinkagePolicy::singleLinkage;
        case LinkagePolicyType::completeLinkage:
            return &LinkagePolicy::completeLinkage;
        default:
            return nullptr;
    }
}

template<LinkagePolicy::NormalizeBy Normalization>
SpeakerEmbeddingWrapper LinkagePolicy::computeCentroidHelper(const SpeakerEmbeddingWrapper *embeddings, const Cluster &cluster) {
    if (cluster.count() == 0) return {};
    if (cluster.count() == 1) return embeddings[cluster[0]];

    auto iter = cluster.begin();
    const auto &e0 = embeddings[*(iter++)];
    const auto &e1 = embeddings[*(iter++)];

    auto centroid = SpeakerEmbeddingWrapper(e0.id() + e1.id(), cluster.weight());

    float normalizer, w0, w1, weight;
    if constexpr (Normalization == NormalizeBy::weight) {
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
        if constexpr (Normalization == NormalizeBy::weight)
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

    if constexpr (Normalization == NormalizeBy::l2Norm) {
        return centroid.normalizedInPlace();
    } else {
        return centroid;
    }
}

template<LinkagePolicy::NormalizeBy Normalization>
SpeakerEmbeddingWrapper LinkagePolicy::computeCentroidHelper(
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
    if constexpr (Normalization == NormalizeBy::weight) {
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
        if constexpr (Normalization == NormalizeBy::weight) {
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

    if constexpr (Normalization == NormalizeBy::l2Norm) {
        return centroid.normalizedInPlace();
    } else {
        return centroid;
    }
}


SpeakerEmbeddingWrapper LinkagePolicy::computeCentroid(const LinkagePolicy* policy,
                                                       EmbeddingDistanceMatrix matrix,
                                                       const Cluster &cluster) {
    return policy->computeCentroid(matrix.embeddings(), cluster);
}
