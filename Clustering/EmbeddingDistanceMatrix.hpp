//
// Created by Benjamin Lee on 1/26/26.
//

#pragma once

#include <vector>
#include "Embedding.hpp"
#include "EmbeddingSegment.hpp"
#include "ClusterLinkage.hpp"

template<LinkagePolicy LP>
class EmbeddingDistanceMatrix {
public:
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
     * @brief Get the _spread between the embedding at index row and the embedding at index col 
     * @param row Index of embedding 1
     * @param col Index of embedding 2
     * @return The _spread between the two embeddings
     */
    [[nodiscard]] float distance(long row, long col) const;
    
    // Get the embedding at the row
    [[nodiscard]] inline Embedding& embedding(long index) const { return embeddings[index]; }
    
    /**
     * Insert a new embedding to the matrix at the next free slot and records the slot in the embedding
     * @param embedding The embedding to add
     * @param isTentative Whether the embedding is tentative
     */
    void insert(Embedding& embedding, bool isTentative = false);

    /**
     * Replace the embedding at a given index with a new one and update the distances
     * @param index The index of the embedding to replace
     * @param embedding The embedding to add
     */
    void replace(long index, Embedding& embedding);
    
    /**
     * @brief Remove the embedding at the index without changing any other indices.
     * @param index Index of the embedding to remove from further consideration
     */
    void remove(long index);

    /**
     * @brief Remove the embedding without changing any other indices.
     * @param embedding Embedding to remove
     */
    inline void remove(const Embedding& embedding) { remove(embedding.matrixIndex()); }
    
    /**
     * @brief Stream new tentative and finalized embeddings to the matrix. 
     * Removes old tentative embeddings and that don't match anything.
     * @param newFinalized New finalized embedding segments
     * @param newTentative New tentative embedding segments
     */
    void stream(std::vector<EmbeddingSegment>& newFinalized, std::vector<EmbeddingSegment>& newTentative);

    /**
     * @brief Update must link constraints from embedding segments. 
     * @param finalized Finalized embedding segments
     * @param tentative Tentative embedding segments
     */
    void updateMustLinkConstraints(std::vector<EmbeddingSegment> const& finalized, std::vector<EmbeddingSegment> const& tentative);
    
    EmbeddingDistanceMatrix& operator=(const EmbeddingDistanceMatrix&) = delete;
    EmbeddingDistanceMatrix& operator=(EmbeddingDistanceMatrix&&) noexcept;
private:
    float* matrix;
    Embedding* embeddings;
    std::vector<std::ptrdiff_t> freeIndices{};
    std::vector<std::ptrdiff_t> tentativeIndices{};
    std::unordered_map<UUID, long> idToIndex{};
    std::vector<MustLinkConstraint> mustLinkIndices{};
    std::vector<MustLinkConstraint> tentativeMustLinkIndices{};
    long _embeddingCount{0};
    long _capacity{0};
    long _size{0};
    long matrixEndIndex{0};

    template<LinkagePolicy> friend class Dendrogram;
    //template<LinkagePolicy> 
    friend class SpeakerForest;
};
