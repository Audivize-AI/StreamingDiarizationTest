#pragma once

#include <vector>
#include <span>
#include <ranges>
#include <unordered_map>
#include "SpeakerEmbeddingWrapper.hpp"
#include "EmbeddingSegmentWrapper.hpp"
#include "LinkagePolicy.hpp"


class Dendrogram;

class EmbeddingDistanceMatrix {
private:
    struct ControlBlock {
        std::vector<std::ptrdiff_t> freeIndices{};
        std::unordered_map<UUIDWrapper, long> tentativeIndices{};
        long embeddingCount{0};
        long capacity{0};
        long size{0};
        long matrixEndIndex{0};
        std::size_t refcount{1};
        
        ControlBlock() = default;
    };
    
    float* matrix;
    SpeakerEmbeddingWrapper* _embeddings;
    ControlBlock* data;
    const LinkagePolicy* linkagePolicy{nullptr};

    friend class Dendrogram;
    
    void remove(long index, bool canBeTentative);
    
    inline void inc() const { if (data) ++data->refcount; };
    
    inline void dec() {
        if (!data) return;
        if (--(data->refcount) <= 0) {
            delete[] matrix;
            delete[] _embeddings;
            delete data;
        }
    }
    
public:
    EmbeddingDistanceMatrix() = delete;
    explicit EmbeddingDistanceMatrix(const LinkagePolicy* linkagePolicy);
    EmbeddingDistanceMatrix(EmbeddingDistanceMatrix&& other) noexcept;
    EmbeddingDistanceMatrix(const EmbeddingDistanceMatrix&);
    ~EmbeddingDistanceMatrix();
    
    [[nodiscard]] inline long embeddingCount() const {
        return data->embeddingCount;
    }
    
    [[nodiscard]] inline long finalizedCount() const {
        return data->embeddingCount - static_cast<long>(data->tentativeIndices.size());
    }
    
    [[nodiscard]] inline long tentativeCount() const {
        return static_cast<long>(data->tentativeIndices.size());
    }
    
    [[nodiscard]] inline std::vector<long> tentativeIndices() const {
        auto view = data->tentativeIndices | std::views::values;
        return {view.begin(), view.end()};
    }
    
    [[nodiscard]] inline std::vector<UUIDWrapper> tentativeIDs() const {
        auto view = data->tentativeIndices | std::views::keys;
        return {view.begin(), view.end()};
    }
    
    [[nodiscard]] inline std::vector<std::pair<UUIDWrapper, long>> tentatives() const {
        return {data->tentativeIndices.begin(), data->tentativeIndices.end()};
    }
    
    [[nodiscard]] inline long size() const {
        return data->size;
    }
    
    [[nodiscard]] inline long capacity() const {
        return data->capacity;
    }
    
    [[nodiscard]] inline const SpeakerEmbeddingWrapper* embeddings() const {
        return _embeddings;
    }
    
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
    [[nodiscard]] inline SpeakerEmbeddingWrapper embedding(long index) const {
        return _embeddings[index];
    }
    
    /**
     * Insert a new embedding to the matrix at the next free slot and records the slot in the embedding
     * @param embedding The embedding to add
     * @param isTentative Whether the embedding is tentative
     */
    void insert(SpeakerEmbeddingWrapper const& embedding, bool isTentative = false);

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
    inline void stream(std::vector<EmbeddingSegmentWrapper>& newFinalized,
                       std::vector<EmbeddingSegmentWrapper>& newTentative) {
        stream({newFinalized.begin(), newFinalized.size()},
               {newTentative.begin(), newTentative.size()});
    }
    
    /**
     * @brief Stream new tentative and finalized embeddings to the matrix.
     * Removes old tentative embeddings and that don't match anything.
     * @param newFinalized New finalized embedding segments
     * @param newTentative New tentative embedding segments
     */
    void stream(std::span<EmbeddingSegmentWrapper> newFinalized, std::span<EmbeddingSegmentWrapper> newTentative);
    
    /**
     * @brief Build dendrogram from the embeddings
     * @param ignoreTentative Whether to exclude tentative embeddings from the dendrogram  
     * @return Dendrogram object
     */
    [[nodiscard]] Dendrogram dendrogram(bool ignoreTentative = false) const;
    
    /**
     * @brief Builds a new distance matrix from the indices provided and removes them from this matrix.
     * @param indices Indices of the embeddings to gather/isolate 
     * @return The new distance matrix
     */
    inline EmbeddingDistanceMatrix gatherAndPop(std::vector<long> const& indices) {
        return gatherAndPop(std::span<const long>(indices.begin(), indices.end()));
    }
    
    /**
     * @brief Builds a new distance matrix from the indices provided and removes them from this matrix.
     * @param indices Indices of the embeddings to gather/isolate
     * @return The new distance matrix
     */
    EmbeddingDistanceMatrix gatherAndPop(std::span<const long> indices);

    /**
     * @brief Absorb another distance matrix
     * @param other Matrix to absorb
     */
    void absorb(const EmbeddingDistanceMatrix& other);
    
    EmbeddingDistanceMatrix& operator=(const EmbeddingDistanceMatrix&);
    EmbeddingDistanceMatrix& operator=(EmbeddingDistanceMatrix&&) noexcept;
};
