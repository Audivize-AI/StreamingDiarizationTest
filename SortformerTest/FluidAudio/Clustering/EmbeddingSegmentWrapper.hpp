#pragma once
#include "SpeakerEmbeddingWrapper.hpp"
#include <unordered_set>
#include <memory>
#include "LinkagePolicy.hpp"

template<LinkagePolicy LP>
class EmbeddingSegmentWrapper {
private:
    std::unique_ptr<SpeakerEmbeddingWrapper[]> embeddings;
    std::vector<UUIDWrapper> _speakerSegmentIds;
    const UUIDWrapper _id;
    const long _speakerId;
    const long _embeddingCount;
    
public:
    /**
     * @param id Embedding Segment ID. Sum of all the embedding IDs
     * @param speakerId Embedding segment speaker ID
     * @param swiftEmbeddingsPtr Pointer to the array of SpeakerEmbeddings
     * @param embeddingCount Number of speaker embeddings
     * @param sortformerSegmentIdsPtr Pointer to the array of sortformer segment IDs
     * @param sortformerSegmentCount Number of sortformer segment IDs
     */
    explicit EmbeddingSegmentWrapper(
            uuid_t id, 
            long speakerId,
            const void** swiftEmbeddingsPtr, 
            long embeddingCount,
            uuid_t* sortformerSegmentIdsPtr,
            long speakerSegmentCount
    ):
            embeddings(std::make_unique<SpeakerEmbeddingWrapper[]>(embeddingCount)),
            _speakerSegmentIds(),
            _id(id),
            _speakerId(speakerId),
            _embeddingCount(embeddingCount)
    {
        for (int i = 0; i < embeddingCount; ++i)
            embeddings[i] = SpeakerEmbeddingWrapper(swiftEmbeddingsPtr[i]);
            _speakerSegmentIds.reserve(speakerSegmentCount);
        for (int i = 0; i < speakerSegmentCount; ++i)
            _speakerSegmentIds.emplace_back(sortformerSegmentIdsPtr[i]);
    }

    EmbeddingSegmentWrapper(const EmbeddingSegmentWrapper&) = delete;
    EmbeddingSegmentWrapper& operator=(const EmbeddingSegmentWrapper&) = delete;
    EmbeddingSegmentWrapper(EmbeddingSegmentWrapper&&) noexcept = default;
    EmbeddingSegmentWrapper& operator=(EmbeddingSegmentWrapper&&) noexcept = delete;
    
    // Segment speaker ID
    [[nodiscard]] inline long speakerId() const { return _speakerId; }
    
    // Number of embeddings
    [[nodiscard]] inline long embeddingCount() const { return _embeddingCount; }
    
    // Segment ID
    [[nodiscard]] inline UUIDWrapper id() const { return _id; }
    
    // Get segment centroid
    [[nodiscard]] inline SpeakerEmbeddingWrapper centroid() { 
        return LP::computeCentroid(_id, embeddings.get(), _embeddingCount, std::move(_speakerSegmentIds));
    }
};

template class EmbeddingSegmentWrapper<WardLinkage>;
template class EmbeddingSegmentWrapper<CosineAverageLinkage>;
template class EmbeddingSegmentWrapper<WeightedAverageLinkage>;
template class EmbeddingSegmentWrapper<WeightedCosineAverageLinkage>;


