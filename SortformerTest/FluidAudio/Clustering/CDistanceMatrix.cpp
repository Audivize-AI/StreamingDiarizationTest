//
// Created by Benjamin Lee on 1/29/26.
//
#include "CDistanceMatrix.hpp"
#include "Dendrogram.hpp"


CDistanceMatrix::CDistanceMatrix(const LinkagePolicy* linkagePolicy): 
        matrix(nullptr),
        _embeddings(nullptr),
        linkagePolicy(linkagePolicy) {}

CDistanceMatrix::CDistanceMatrix(CDistanceMatrix&& other) noexcept :
        matrix(other.matrix),
        _embeddings(other._embeddings),
        _tentativeIndices(std::move(other._tentativeIndices)),
        _freeIndices(std::move(other._freeIndices)),
        _size(other._size),
        _capacity(other._capacity),
        _embeddingCount(other._embeddingCount),
        _matrixEndIndex(other._matrixEndIndex),
        linkagePolicy(other.linkagePolicy)
{
    other.matrix = nullptr;
    other._embeddings = nullptr;
    other._size = 0;
    other._capacity = 0;
    other._embeddingCount = 0;
    other._matrixEndIndex = 0;
}

CDistanceMatrix::~CDistanceMatrix() {
    delete[] matrix;
    delete[] _embeddings;
}

void CDistanceMatrix::reserve(long newCapacity) {
    if (newCapacity <= _capacity)
        return;
    
    // Copy embeddings
    auto newEmbeddings = new SpeakerEmbeddingWrapper[newCapacity]{};
    if (_embeddings != nullptr) {
        for (auto i = 0; i < _size; ++i) {
            newEmbeddings[i] = std::move(_embeddings[i]);
        }
        delete[] _embeddings;
    }
    _embeddings = newEmbeddings;

    // Copy matrix
    auto newMatrixCapacity = newCapacity * (newCapacity - 1) / 2;
    auto newMatrix = new float[newMatrixCapacity];
    if (matrix != nullptr) {
        std::memcpy(newMatrix, matrix, _matrixEndIndex * sizeof(float));
        delete[] matrix;
    }
    matrix = newMatrix;
    
    _capacity = newCapacity;
}


float CDistanceMatrix::distance(long row, long col) const {
    // Diagonal elements are always 0 because cosineDistance(E, E) = 0
    if (row == col)
        return 0.f;

    // Data is stored as a lower triangle matrix without the diagonal, so we need col ≤ row.
    if (row < col)
        std::swap(row, col);

    // n(n+1) / 2 - n = n(n-1) / 2
    return matrix[row * (row - 1) / 2 + col];
}


void CDistanceMatrix::insert(SpeakerEmbeddingWrapper const& embedding, bool isTentative) {
    long index, matrixIndex;
    
    if (_freeIndices.empty()) {
        index = _size;
        // Ensure there is enough capacity if appending to the end
        if (_size + 1 > _capacity)
            reserve(std::max(_capacity * 2, _size + 1));
        _matrixEndIndex += _size++;
    } else {
        index = _freeIndices.back();
        _freeIndices.pop_back();

        // Update each (row, index) squaredDistanceTo for row > index
        // Start at (r, c) = (index + 1, index)
        matrixIndex = index * (index + 3) / 2;
        for (auto row = index + 1; row < _size; ++row) {
            matrix[matrixIndex] = linkagePolicy->distance(embedding, _embeddings[row]);
            matrixIndex += row;
        }
    }

    // Update each (index, col)
    matrixIndex = index * (index - 1) / 2;
    for (auto col = 0; col < index; ++col) {
        matrix[matrixIndex++] = linkagePolicy->distance(embedding, _embeddings[col]);
    }

    _embeddings[index] = embedding;
    ++_embeddingCount;

    if (isTentative)
        _tentativeIndices[embedding.id()] = index;
}


void CDistanceMatrix::replace(long index, SpeakerEmbeddingWrapper& embedding) {
    if (_embeddings[index] == embedding) return;
    
    // Update each (index, col)
    auto matrixIndex = index * (index - 1) / 2;

    for (auto i = 0; i < index; ++i) {
        matrix[matrixIndex++] = linkagePolicy->distance(embedding, _embeddings[i]);
    }

    if (index < _size) {
        // Update each (r, index) squaredDistanceTo for r > index
        // Start at (r, c) = (index + 1, 0)
        matrixIndex = index * (index + 1) / 2;
        for (auto i = index + 1; i < _size; ++i) {
            // This makes column = index in first iteration and increments the index in subsequent ones
            matrixIndex += i - 1;
            matrix[matrixIndex] = linkagePolicy->distance(embedding, _embeddings[i]);
        }
    }

    _embeddings[index] = embedding;
}


void CDistanceMatrix::remove(long index, bool canBeTentative) {
    if (index >= _size || _embeddings[index].expired())
        return;
    if (index < _size - 1)
        _freeIndices.emplace_back(index);
    else
        _matrixEndIndex -= --_size;
    
    if (canBeTentative)
        _tentativeIndices.erase(_embeddings[index].id());
    
    _embeddings[index].releaseVector();
    --_embeddingCount;
}


void CDistanceMatrix::stream(std::span<EmbeddingSegmentWrapper> newFinalized, std::span<EmbeddingSegmentWrapper> newTentative) {
    std::unordered_map<UUIDWrapper, long> keptTentativeIndices = {};
    std::vector<SpeakerEmbeddingWrapper> unseenFinalized = {};
    std::vector<SpeakerEmbeddingWrapper> unseenTentative = {};
    unseenFinalized.reserve(newFinalized.size());
    unseenTentative.reserve(newTentative.size());
    keptTentativeIndices.reserve(newTentative.size());

    // Filter for unmatched embeddings in both the tentative indices and the incoming embeddings
    for (auto& segment : newTentative) {
#if (!MUST_LINK)
        for (auto& segment: segment.embeddings()) {
#endif
            const auto id = segment.id();
            auto pIndex = _tentativeIndices.find(id);
            
            if (pIndex == _tentativeIndices.end()) {
#if (MUST_LINK)
                unseenTentative.emplace_back(segment.centroid(linkagePolicy));
#else
                unseenTentative.emplace_back(segment);
#endif
                continue;
            }
            
            auto index = pIndex->second;
            if (_tentativeIndices.erase(id))
                keptTentativeIndices[id] = index;
#if (!MUST_LINK)
        }
#endif
    }

    for (auto& segment : newFinalized) {
#if (!MUST_LINK)
        for (auto& segment : segment.embeddings()) {
#endif
            const auto id = segment.id();
            auto pIndex = _tentativeIndices.find(id);
            
            if (pIndex == _tentativeIndices.end()) {
#if (MUST_LINK)
                unseenFinalized.emplace_back(segment.centroid(linkagePolicy));
#else
                unseenFinalized.emplace_back(segment);
#endif
                continue;
            }
            
            _tentativeIndices.erase(id);
#if (!MUST_LINK)
        }
#endif
    }

    // Remove unmatched indices
    for (auto [_, index]: _tentativeIndices)
        remove(index, false);
    
    _tentativeIndices = std::move(keptTentativeIndices);

    // Append new embeddings
    for (auto& embedding: unseenFinalized)
        insert(embedding, false);
    
    for (auto& embedding: unseenTentative)
        insert(embedding, true);
}

CDistanceMatrix& CDistanceMatrix::operator=(CDistanceMatrix&& other) noexcept {
    delete[] matrix;
    delete[] _embeddings;
    matrix = other.matrix;
    _embeddings = other._embeddings;
    _size = other._size;
    _capacity = other._capacity;
    _embeddingCount = other._embeddingCount;
    _matrixEndIndex = other._matrixEndIndex;
    
    other.matrix = nullptr;
    other._embeddings = nullptr;
    other._size = 0;
    other._capacity = 0;
    other._embeddingCount = 0;
    other._matrixEndIndex = 0;
    return *this;
}

CDistanceMatrix CDistanceMatrix::gatherAndPop(std::span<const long> indices) {
    const auto k = static_cast<long>(indices.size());
    if (k == 0) return CDistanceMatrix(linkagePolicy);
    
    CDistanceMatrix result(linkagePolicy);
    result.reserve(k);
    
    result._size = k;
    result._embeddingCount = k;
    result._matrixEndIndex = k * (k - 1) / 2;
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
    
    result._tentativeIndices.reserve(_tentativeIndices.size());
    
    for (auto oldIndex: indices) {
        // Put the embedding in the new matrix
        auto& dstEmbedding = result._embeddings[newIndex];
        dstEmbedding = std::move(this->_embeddings[oldIndex]);
        
        const auto embId = dstEmbedding.id();
        if (_tentativeIndices.erase(embId))
            result._tentativeIndices[embId] = newIndex;
        ++newIndex;
        
        // Free the spot in this matrix
        if (oldIndex < _size - 1)
            _freeIndices.emplace_back(oldIndex);
        else
            _matrixEndIndex -= --_size;
    }
    
    return result;
}

void CDistanceMatrix::insertTentativeFrom(const CDistanceMatrix& other) {
    const auto tentativeCount = static_cast<long>(other._tentativeIndices.size());
    if (tentativeCount == 0) return;

    const auto targetCount = _embeddingCount + tentativeCount;
    if (targetCount > _capacity) {
        reserve(std::max(_capacity * 2, targetCount));
    }

    for (const auto& [_, index] : other._tentativeIndices) {
        if (index < 0 || index >= other._size) continue;
        const auto& embedding = other._embeddings[index];
        if (embedding.expired()) continue;
        insert(embedding, true);
    }
}


void CDistanceMatrix::absorb(const CDistanceMatrix& other) {
    if (other._embeddingCount == 0) return;
    
    // Ensure capacity for all incoming embeddings
    _embeddingCount += other._embeddingCount;
    if (_embeddingCount > _capacity)
        reserve(std::max(_capacity * 2, _embeddingCount));

    // Map other's indices to new indices in this matrix.
    auto oldToNew = std::make_unique<long[]>(other._size); // index in other -> index in this
    auto isNew = std::make_unique<bool[]>(_size);
    std::fill(oldToNew.get(), oldToNew.get() + other._size, -1);
    const auto oldSize = _size;
    
    // Transfer embeddings
    for (long i = 0; i < other._size; ++i) {
        if (other._embeddings[i].expired())
            continue;
        
        // Get next index
        long nextIndex;
        if (!_freeIndices.empty()) {
            nextIndex = _freeIndices.back();
            _freeIndices.pop_back();
            isNew[nextIndex] = true;
        } else {
            nextIndex = _size;
            _matrixEndIndex += _size++;
        }
        oldToNew[i] = nextIndex;
        
        // Add the embedding
        auto& dstEmbedding = this->_embeddings[nextIndex];
        dstEmbedding = other._embeddings[i];
        
        auto embId = dstEmbedding.id();
        if (other._tentativeIndices.contains(embId))
            _tentativeIndices[embId] = nextIndex;
        
        // Compute intra-matrix distances
        auto dstMatrixIndex = nextIndex * (nextIndex - 1) / 2;

        if (nextIndex < oldSize) {
            for (auto j = 0; j < nextIndex; ++j) {
                if (isNew[j]) continue;
                matrix[dstMatrixIndex++] = linkagePolicy->distance(dstEmbedding, _embeddings[j]);
            }
            // Update each (r, index) squaredDistanceTo for r > index
            // Start at (r, c) = (index + 1, 0)
            dstMatrixIndex = nextIndex * (nextIndex + 1) / 2;
            for (auto j = nextIndex + 1; j < oldSize; ++j) {
                // This makes column = index in first iteration and increments the index in subsequent ones
                dstMatrixIndex += j - 1;
                if (isNew[j]) continue;
                matrix[dstMatrixIndex] = linkagePolicy->distance(dstEmbedding, _embeddings[j]);
            }
        } else {
            for (auto j = 0; j < oldSize; ++j) {
                if (isNew[j]) continue;
                matrix[dstMatrixIndex++] = linkagePolicy->distance(dstEmbedding, _embeddings[j]);
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

Dendrogram CDistanceMatrix::dendrogram(bool ignoreTentative) const {
    return Dendrogram(*this, ignoreTentative);
}
