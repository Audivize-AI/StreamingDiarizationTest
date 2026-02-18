//
// Created by Benjamin Lee on 1/29/26.
//
#include "EmbeddingDistanceMatrix.hpp"
#include "Dendrogram.hpp"

template<LinkagePolicy LP>
EmbeddingDistanceMatrix<LP>::EmbeddingDistanceMatrix(): matrix(nullptr), embeddings(nullptr) {}

template<LinkagePolicy LP>
EmbeddingDistanceMatrix<LP>::EmbeddingDistanceMatrix(EmbeddingDistanceMatrix&& other) noexcept :
    matrix(other.matrix),
    embeddings(other.embeddings),
    freeIndices(std::move(other.freeIndices)),
    tentativeIndices(std::move(other.tentativeIndices)),
    _embeddingCount(other._embeddingCount),
    _capacity(other._capacity),
    _size(other._size),
    matrixEndIndex(other.matrixEndIndex)
{
    other.matrix = nullptr;
    other.embeddings = nullptr;
    other._embeddingCount = 0;
    other._capacity = 0;
    other._size = 0;
    other.matrixEndIndex = 0;
}

template<LinkagePolicy LP>
EmbeddingDistanceMatrix<LP>::~EmbeddingDistanceMatrix() {
    delete[] matrix;
    delete[] embeddings;
}

template<LinkagePolicy LP>
void EmbeddingDistanceMatrix<LP>::reserve(long newCapacity) {
    if (newCapacity <= _capacity)
        return;
    
    // Copy embeddings
    auto newEmbeddings = new SpeakerEmbeddingWrapper[newCapacity]{};
    if (embeddings != nullptr) {
        for (auto i = 0; i < _size; ++i) {
            newEmbeddings[i] = std::move(embeddings[i]);
        }
        delete[] embeddings;
    }
    embeddings = newEmbeddings;

    // Copy matrix
    auto newMatrixCapacity = newCapacity * (newCapacity - 1) / 2;
    auto newMatrix = new float[newMatrixCapacity];
    if (matrix != nullptr) {
        std::memcpy(newMatrix, matrix, matrixEndIndex * sizeof(float));
        delete[] matrix;
    }
    matrix = newMatrix;
    
    _capacity = newCapacity;
}

template<LinkagePolicy LP>
float EmbeddingDistanceMatrix<LP>::distance(long row, long col) const {
    // Diagonal elements are always 0 because cosineDistance(E, E) = 0
    if (row == col)
        return 0.f;

    // Data is stored as a lower triangle matrix without the diagonal, so we need col ≤ row.
    if (row < col)
        std::swap(row, col);

    // n(n+1) / 2 - n = n(n-1) / 2
    return matrix[row * (row - 1) / 2 + col];
}

template<LinkagePolicy LP>
void EmbeddingDistanceMatrix<LP>::insert(SpeakerEmbeddingWrapper& embedding, bool isTentative) {
    long index, matrixIndex;
    if (freeIndices.empty()) {
        index = _size;
        // Ensure there is enough _capacity if appending to the end
        if (_size + 1 > _capacity)
            reserve(std::max(_capacity * 2, _size + 1));
        matrixEndIndex += _size++;
    } else {
        index = freeIndices.back();
        freeIndices.pop_back();

        // Update each (row, index) squaredDistanceTo for row > index
        // Start at (r, c) = (index + 1, index)
        matrixIndex = index * (index + 3) / 2;
        for (auto row = index + 1; row < _size; ++row) {
            matrix[matrixIndex] = LP::distance(embedding, embeddings[row]);
            matrixIndex += row;
        }
    }

    // Update each (index, col)
    matrixIndex = index * (index - 1) / 2;
    for (auto col = 0; col < index; ++col) {
        matrix[matrixIndex++] = LP::distance(embedding, embeddings[col]);
    }

    embeddings[index] = embedding;
    ++_embeddingCount;

    if (isTentative)
        tentativeIndices[embedding.id()] = index;
}

template<LinkagePolicy LP>
void EmbeddingDistanceMatrix<LP>::replace(long index, SpeakerEmbeddingWrapper& embedding) {
    if (embeddings[index] == embedding) return;
    
    // Update each (index, col)
    auto matrixIndex = index * (index - 1) / 2;

    for (auto i = 0; i < index; ++i) {
        matrix[matrixIndex++] = LP::distance(embedding, embeddings[i]);
    }

    if (index < _size) {
        // Update each (r, index) squaredDistanceTo for r > index
        // Start at (r, c) = (index + 1, 0)
        matrixIndex = index * (index + 1) / 2;
        for (auto i = index + 1; i < _size; ++i) {
            // This makes column = index in first iteration and increments the index in subsequent ones
            matrixIndex += i - 1;
            matrix[matrixIndex] = LP::distance(embedding, embeddings[i]);
        }
    }

    embeddings[index] = embedding;
}

template<LinkagePolicy LP>
void EmbeddingDistanceMatrix<LP>::remove(long index, bool canBeTentative) {
    if (index >= _size || embeddings[index].expired())
        return;
    if (index < _size - 1)
        freeIndices.emplace_back(index);
    else
        matrixEndIndex -= --_size;
    
    if (canBeTentative)
        tentativeIndices.erase(embeddings[index].id());
    
    embeddings[index].releaseVector();
    --_embeddingCount;
}

template<LinkagePolicy LP>
void EmbeddingDistanceMatrix<LP>::stream(std::vector<EmbeddingSegment>& newFinalized, std::vector<EmbeddingSegment>& newTentative) {
    std::unordered_map<UUIDWrapper, long> keptTentativeIndices = {};
    std::vector<SpeakerEmbeddingWrapper> unseenFinalized = {};
    std::vector<SpeakerEmbeddingWrapper> unseenTentative = {};
    unseenFinalized.reserve(newFinalized.size());
    unseenTentative.reserve(newTentative.size());
    keptTentativeIndices.reserve(newTentative.size());

    // Filter for unmatched embeddings in both the tentative indices and the incoming embeddings
    for (auto& segment : newTentative) {
        const auto segmentId = segment.id();
        auto pIndex = tentativeIndices.find(segmentId);
        
        if (pIndex == tentativeIndices.end()) {
            unseenTentative.emplace_back(segment.centroid());
            continue;
        }
        
        auto index = pIndex->second;
        if (tentativeIndices.erase(segmentId))
            keptTentativeIndices[segmentId] = index;
    }

    for (auto& segment : newFinalized) {
        const auto segmentId = segment.id();
        auto pIndex = tentativeIndices.find(segmentId);

        if (pIndex == tentativeIndices.end()) {
            unseenFinalized.emplace_back(segment.centroid());
            continue;
        }

        auto index = pIndex->second;
        tentativeIndices.erase(segmentId);
    }

    // Remove unmatched indices
    for (auto [_, index]: tentativeIndices)
        remove(index, false);
    tentativeIndices = std::move(keptTentativeIndices);

    // Append new embeddings
    for (auto& embedding: unseenFinalized)
        insert(embedding, false);
    
    for (auto& embedding: unseenTentative)
        insert(embedding, true);
}

template<LinkagePolicy LP>
EmbeddingDistanceMatrix<LP>& EmbeddingDistanceMatrix<LP>::operator=(EmbeddingDistanceMatrix&& other) noexcept {
    delete[] matrix;
    delete[] embeddings;
    matrix = other.matrix;
    embeddings = other.embeddings;
    freeIndices = std::move(other.freeIndices);
    tentativeIndices = std::move(other.tentativeIndices);
    _embeddingCount = other._embeddingCount;
    _capacity = other._capacity;
    _size = other._size;
    matrixEndIndex = other.matrixEndIndex;
    other.matrix = nullptr;
    other.embeddings = nullptr;
    other._embeddingCount = 0;
    other._capacity = 0;
    other._size = 0;
    other.matrixEndIndex = 0;
    return *this;
}

template<LinkagePolicy LP>
EmbeddingDistanceMatrix<LP> EmbeddingDistanceMatrix<LP>::gatherAndPop(const std::vector<long> &indices) {
    const auto k = static_cast<long>(indices.size());
    if (k == 0) return {};
    
    EmbeddingDistanceMatrix<LP> result;
    result.reserve(k);
    
    result._size = k;
    result._embeddingCount = k;
    result.matrixEndIndex = k * (k - 1) / 2;
    this->_embeddingCount -= k;

    // Copy pairwise distances directly from the source packed lower-triangular matrix.
    for (long newRow = 1; newRow < k; ++newRow) {
        const auto oldRow = indices[newRow];
        const auto dstBase = newRow * (newRow - 1) / 2;
        for (auto newCol = 0; newCol < newRow; ++newCol) {
            auto r = oldRow, c = indices[newCol];
            if (r < c) std::swap(r, c);
            result.matrix[dstBase + newCol] = this->matrix[r * (r - 1) / 2 + c];
        }
    }
    
    // Transfer embeddings
    long newIndex = 0;
    for (auto oldIndex: indices) {
        // Put the embedding in the new matrix
        auto& dstEmbedding = result.embeddings[newIndex];
        dstEmbedding = std::move(this->embeddings[oldIndex]);
        
        const auto embId = dstEmbedding.id();
        if (this->tentativeIndices.erase(embId))
            result.tentativeIndices[embId] = newIndex;
        ++newIndex;
        
        // Free the spot in this matrix
        if (oldIndex < this->_size - 1) 
            this->freeIndices.emplace_back(oldIndex);
        else
            this->matrixEndIndex -= --this->_size;
    }
    
    return result;
}

template<LinkagePolicy LP>
void EmbeddingDistanceMatrix<LP>::absorb(const EmbeddingDistanceMatrix& other) {
    if (other._embeddingCount == 0) return;

    // Ensure capacity for all incoming embeddings
    this->_embeddingCount += other._embeddingCount;
    if (this->_embeddingCount > this->_capacity)
        reserve(std::max(this->_capacity * 2, this->_embeddingCount));

    // Map other's indices to new indices in this matrix.
    auto oldToNew = std::make_unique<long[]>(other._size); // index in other -> index in this
    auto isNew = std::make_unique<bool[]>(this->_size);
    std::fill(oldToNew.get(), oldToNew.get() + other._size, -1);
    const auto oldSize = this->_size;
    
    // Transfer embeddings
    for (long i = 0; i < other._size; ++i) {
        if (other.embeddings[i].expired())
            continue;
        
        // Get next index
        long nextIndex;
        if (!this->freeIndices.empty()) {
            nextIndex = this->freeIndices.back();
            this->freeIndices.pop_back();
            isNew[nextIndex] = true;
        } else {
            nextIndex = this->_size;
            this->matrixEndIndex += this->_size++;
        }
        oldToNew[i] = nextIndex;
        
        // Add the embedding
        auto& dstEmbedding = this->embeddings[nextIndex];
        dstEmbedding = other.embeddings[i];
        
        auto embId = dstEmbedding.id();
        if (other.tentativeIndices.contains(embId))
            tentativeIndices[embId] = nextIndex;
        
        // Compute intra-matrix distances
        auto dstMatrixIndex = nextIndex * (nextIndex - 1) / 2;

        if (nextIndex < oldSize) {
            for (auto j = 0; j < nextIndex; ++j) {
                if (isNew[j]) continue;
                matrix[dstMatrixIndex++] = LP::distance(dstEmbedding, embeddings[j]);
            }
            // Update each (r, index) squaredDistanceTo for r > index
            // Start at (r, c) = (index + 1, 0)
            dstMatrixIndex = nextIndex * (nextIndex + 1) / 2;
            for (auto j = nextIndex + 1; j < oldSize; ++j) {
                // This makes column = index in first iteration and increments the index in subsequent ones
                dstMatrixIndex += j - 1;
                if (isNew[j]) continue;
                matrix[dstMatrixIndex] = LP::distance(dstEmbedding, embeddings[j]);
            }
        } else {
            for (auto j = 0; j < oldSize; ++j) {
                if (isNew[j]) continue;
                matrix[dstMatrixIndex++] = LP::distance(dstEmbedding, embeddings[j]);
            }
        }
    }
    
    // Copy pairwise distances
    for (long rOld = 1; rOld < other._size; ++rOld) {
        const auto rNew = oldToNew[rOld];
        const auto srcBase = rNew * (rNew - 1) / 2;
        for (auto cOld = 0; cOld < rOld; ++cOld) {
            auto r = rNew, c = oldToNew[cOld];
            if (r < c) std::swap(r, c);
            this->matrix[r * (r - 1) / 2 + c] = other.matrix[srcBase + cOld];
        }
    }
}

template class EmbeddingDistanceMatrix<WardLinkage>;
template class EmbeddingDistanceMatrix<CosineAverageLinkage>;
template class EmbeddingDistanceMatrix<WeightedAverageLinkage>;
template class EmbeddingDistanceMatrix<WeightedCosineAverageLinkage>;
