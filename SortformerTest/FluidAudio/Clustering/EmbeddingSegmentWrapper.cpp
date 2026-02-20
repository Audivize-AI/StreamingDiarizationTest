#include "EmbeddingSegmentWrapper.hpp"

EmbeddingSegmentWrapper::EmbeddingSegmentWrapper(
        const uuid_t id,
        long embeddingCount,
        long speakerSegmentCount
):
        _embeddings(),
        speakerSegmentIds(),
        _id(id)
{
    _embeddings.reserve(embeddingCount);
    speakerSegmentIds.reserve(speakerSegmentCount);
}

EmbeddingSegmentWrapper::EmbeddingSegmentWrapper(EmbeddingSegmentWrapper&& other) noexcept:
        _embeddings(std::move(other._embeddings)),
        speakerSegmentIds(std::move(other.speakerSegmentIds)),
        _id(other._id)
{
    other._id = UUIDWrapper::zero;
}

EmbeddingSegmentWrapper& EmbeddingSegmentWrapper::operator=(EmbeddingSegmentWrapper&& other) noexcept {
    this->_embeddings = std::move(other._embeddings);
    this->speakerSegmentIds = std::move(other.speakerSegmentIds);
    this->_id = other._id;
    
    other._id = UUIDWrapper::zero;
    
    return *this;
}
