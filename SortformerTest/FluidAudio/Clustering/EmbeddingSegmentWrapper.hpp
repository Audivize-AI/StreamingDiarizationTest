#pragma once
#include "SpeakerEmbeddingWrapper.hpp"
#include <unordered_set>
#include <memory>
#include "LinkagePolicy.hpp"

#define MUST_LINK true

class EmbeddingSegmentWrapper {
private:
    std::vector<SpeakerEmbeddingWrapper> _embeddings{};
    std::vector<UUIDWrapper> speakerSegmentIds{};
    UUIDWrapper _id = UUIDWrapper::zero;
    
public:
    /**
     * @param id Embedding Segment ID. Sum of all the embedding IDs
     * @param embeddingCount Number of speaker embeddings
     * @param speakerSegmentCount Number of speaker segment IDs
     */
    explicit EmbeddingSegmentWrapper(const uuid_t id, long embeddingCount, long speakerSegmentCount);
    
    EmbeddingSegmentWrapper() = default;
    EmbeddingSegmentWrapper(const EmbeddingSegmentWrapper&) = default;
    EmbeddingSegmentWrapper(EmbeddingSegmentWrapper&&) noexcept;
    EmbeddingSegmentWrapper& operator=(const EmbeddingSegmentWrapper&) = default;
    EmbeddingSegmentWrapper& operator=(EmbeddingSegmentWrapper&&) noexcept;
    
    // Number of embeddings
    [[nodiscard]] inline long embeddingCount() const { return _embeddings.size(); }
    
    // Segment ID
    [[nodiscard]] inline UUIDWrapper id() const { return _id; }
    
    // Check if no embeddings have been assigned to it
    [[nodiscard]] inline bool isEmpty() const { return _embeddings.empty(); }
    
    [[nodiscard]] std::vector<SpeakerEmbeddingWrapper>& embeddings() {
        return _embeddings;
    }
    
    // Get segment centroid
    [[nodiscard]] inline SpeakerEmbeddingWrapper centroid(const LinkagePolicy* linkagePolicy) {
        return linkagePolicy->computeCentroid(_id, _embeddings.data(), _embeddings.size(), std::move(speakerSegmentIds));
    }
    
    /**
     * @brief Add a segment
     * @param id Pointer to the UUID's memory block
     */
    inline void addSegmentId(const uuid_t id) {
        speakerSegmentIds.emplace_back(id);
    }
    
    /**
     * @brief Add a speaker embedding
     * @param swiftEmbeddingPtr Pointer to a Swift Speaker Embedding object
     */
    inline void addEmbedding(void const* swiftEmbeddingPtr) {
        _embeddings.emplace_back(swiftEmbeddingPtr);
#if (!MUST_LINK)
        _embeddings[_embeddings.size() - 1].segmentIds() = speakerSegmentIds;
#endif
    }
};
