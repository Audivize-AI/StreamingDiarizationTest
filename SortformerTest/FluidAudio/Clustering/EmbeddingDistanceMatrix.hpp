#pragma once

#include <vector>
#include "SpeakerEmbeddingWrapper.hpp"
#include "EmbeddingSegmentWrapper.hpp"
#include "LinkagePolicy.hpp"

template<LinkagePolicy> class Dendrogram;

template<LinkagePolicy LP>
class EmbeddingDistanceMatrix {
private:
    float* matrix;
    SpeakerEmbeddingWrapper* embeddings;
    std::vector<std::ptrdiff_t> freeIndices{};
    std::unordered_map<UUIDWrapper, long> tentativeIndices{};
    long _embeddingCount{0};
    long _capacity{0};
    long _size{0};
    long matrixEndIndex{0};

    template<LinkagePolicy> friend class Dendrogram;
    template<LinkagePolicy> friend class SpeakerForest;
    
    void remove(long index, bool canBeTentative);
    
public:
    using EmbeddingSegment = EmbeddingSegmentWrapper<LP>;
    EmbeddingDistanceMatrix();
    EmbeddingDistanceMatrix(EmbeddingDistanceMatrix&& other) noexcept;
    EmbeddingDistanceMatrix(const EmbeddingDistanceMatrix&) = delete;
    ~EmbeddingDistanceMatrix();
    
    [[nodiscard]] inline long embeddingCount() const { return _embeddingCount; }
    [[nodiscard]] inline long finalizedCount() const { return _embeddingCount - tentativeIndices.size(); }
    [[nodiscard]] inline long tentativeCount() const { return tentativeIndices.size(); }
    [[nodiscard]] inline long size() const { return _size; }
    [[nodiscard]] inline long capacity() const { return _capacity; }
    
    /**
     * @brief Reserve memory for more embeddings if necessary.
     * @param newCapacity The new maximum number of embeddings that can be stored.
     */
    void reserve(long newCapacity);

    /**
     * @brief Get the spread between the embedding at index row and the embedding at index col 
     * @param row Index of embedding 1
     * @param col Index of embedding 2
     * @return The _spread between the two embeddings
     */
    [[nodiscard]] float distance(long row, long col) const;
    
    // Get the embedding at the row
    [[nodiscard]] inline SpeakerEmbeddingWrapper& embedding(long index) const { return embeddings[index]; }
    
    /**
     * Insert a new embedding to the matrix at the next free slot and records the slot in the embedding
     * @param embedding The embedding to add
     * @param isTentative Whether the embedding is tentative
     */
    void insert(SpeakerEmbeddingWrapper& embedding, bool isTentative = false);

    /**
     * Replace the embedding at a given index with a new one and update the distances
     * @param index The index of the embedding to replace
     * @param embedding The embedding to add
     */
    void replace(long index, SpeakerEmbeddingWrapper& embedding);
    
    /**
     * @brief Remove the embedding at the index without changing any other indices.
     * @param index Index of the embedding to remove from further consideration
     */
    inline void remove(long index) { this->remove(index, true); }
    
    /**
     * @brief Stream new tentative and finalized embeddings to the matrix. 
     * Removes old tentative embeddings and that don't match anything.
     * @param newFinalized New finalized embedding segments
     * @param newTentative New tentative embedding segments
     */
    void stream(std::vector<EmbeddingSegment>& newFinalized, std::vector<EmbeddingSegment>& newTentative);
    
    /**
     * @brief Build dendrogram from the embeddings
     * @param ignoreTentative Whether to exclude tentative embeddings from the dendrogram  
     * @return Dendrogram object
     */
    [[nodiscard]] inline Dendrogram<LP> dendrogram(bool ignoreTentative = false) const {
        return Dendrogram<LP>(*this, ignoreTentative);
    }
    
    /**
     * @brief Builds a new distance matrix from the indices provided and removes them from this matrix.
     * @param indices Indices of the embeddings to gather/isolate 
     * @return The new distance matrix
     */
    EmbeddingDistanceMatrix<LP> gatherAndPop(std::vector<long> const& indices);

    /**
     * @brief Absorb another matrix
     * @param other Indices of the embeddings to gather/isolate
     */
    void absorb(const EmbeddingDistanceMatrix& other);
    
    EmbeddingDistanceMatrix& operator=(const EmbeddingDistanceMatrix&) = delete;
    EmbeddingDistanceMatrix& operator=(EmbeddingDistanceMatrix&&) noexcept;
};

using CosineDistanceMatrix = EmbeddingDistanceMatrix<CosineAverageLinkage>;
using WardDistanceMatrix = EmbeddingDistanceMatrix<WardLinkage>;
