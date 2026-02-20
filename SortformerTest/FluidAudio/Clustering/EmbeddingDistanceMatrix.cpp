//
// Created by Benjamin Lee on 1/29/26.
//
#include "EmbeddingDistanceMatrix.hpp"
#include "Dendrogram.hpp"


EmbeddingDistanceMatrix::EmbeddingDistanceMatrix(const LinkagePolicy* linkagePolicy): 
        matrix(nullptr),
        _embeddings(nullptr),
        linkagePolicy(linkagePolicy),
        data(new ControlBlock()) {}

EmbeddingDistanceMatrix::EmbeddingDistanceMatrix(EmbeddingDistanceMatrix const& other) :
        matrix(other.matrix),
        _embeddings(other._embeddings),
        data(other.data),
        linkagePolicy(other.linkagePolicy)
{
    inc();
}

EmbeddingDistanceMatrix::EmbeddingDistanceMatrix(EmbeddingDistanceMatrix&& other) noexcept :
        matrix(other.matrix),
        _embeddings(other._embeddings),
        data(other.data),
        linkagePolicy(other.linkagePolicy)
{
    other.matrix = nullptr;
    other._embeddings = nullptr;
    other.data = nullptr;
}

EmbeddingDistanceMatrix::~EmbeddingDistanceMatrix() {
    dec();
}

void EmbeddingDistanceMatrix::reserve(long newCapacity) {
    if (newCapacity <= data->capacity)
        return;
    
    // Copy embeddings
    auto newEmbeddings = new SpeakerEmbeddingWrapper[newCapacity]{};
    auto _size = data->size;
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
        std::memcpy(newMatrix, matrix, data->matrixEndIndex * sizeof(float));
        delete[] matrix;
    }
    matrix = newMatrix;
    
    data->capacity = newCapacity;
}


float EmbeddingDistanceMatrix::distance(long row, long col) const {
    // Diagonal elements are always 0 because cosineDistance(E, E) = 0
    if (row == col)
        return 0.f;

    // Data is stored as a lower triangle matrix without the diagonal, so we need col ≤ row.
    if (row < col)
        std::swap(row, col);

    // n(n+1) / 2 - n = n(n-1) / 2
    return matrix[row * (row - 1) / 2 + col];
}


void EmbeddingDistanceMatrix::insert(SpeakerEmbeddingWrapper const& embedding, bool isTentative) {
    long index, matrixIndex;
    auto _size = data->size;
    
    if (data->freeIndices.empty()) {
        auto _capacity = data->capacity;
        index = _size++;
        // Ensure there is enough capacity if appending to the end
        if (_size > _capacity)
            reserve(std::max(_capacity * 2, _size));
        data->matrixEndIndex += data->size++;
    } else {
        index = data->freeIndices.back();
        data->freeIndices.pop_back();

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
    ++data->embeddingCount;

    if (isTentative)
        data->tentativeIndices[embedding.id()] = index;
}


void EmbeddingDistanceMatrix::replace(long index, SpeakerEmbeddingWrapper& embedding) {
    if (_embeddings[index] == embedding) return;
    
    // Update each (index, col)
    auto matrixIndex = index * (index - 1) / 2;

    for (auto i = 0; i < index; ++i) {
        matrix[matrixIndex++] = linkagePolicy->distance(embedding, _embeddings[i]);
    }

    auto _size = data->size;
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


void EmbeddingDistanceMatrix::remove(long index, bool canBeTentative) {
    auto _size = data->size;
    if (index >= _size || _embeddings[index].expired())
        return;
    if (index < _size - 1)
        data->freeIndices.emplace_back(index);
    else
        data->matrixEndIndex -= --data->size;
    
    if (canBeTentative)
        data->tentativeIndices.erase(_embeddings[index].id());
    
    _embeddings[index].releaseVector();
    --data->embeddingCount;
}


void EmbeddingDistanceMatrix::stream(std::span<EmbeddingSegmentWrapper> newFinalized, std::span<EmbeddingSegmentWrapper> newTentative) {
    std::unordered_map<UUIDWrapper, long> keptTentativeIndices = {};
    std::vector<SpeakerEmbeddingWrapper> unseenFinalized = {};
    std::vector<SpeakerEmbeddingWrapper> unseenTentative = {};
    unseenFinalized.reserve(newFinalized.size());
    unseenTentative.reserve(newTentative.size());
    keptTentativeIndices.reserve(newTentative.size());

    // Filter for unmatched embeddings in both the tentative indices and the incoming embeddings
    auto tentativeIndices = std::move(data->tentativeIndices);
    for (auto& segment : newTentative) {
#if (!MUST_LINK)
        for (auto& segment: segment.embeddings()) {
#endif
            const auto id = segment.id();
            auto pIndex = tentativeIndices.find(id);
            
            if (pIndex == tentativeIndices.end()) {
#if (MUST_LINK)
                unseenTentative.emplace_back(segment.centroid(linkagePolicy));
#else
                unseenTentative.emplace_back(segment);
#endif
                continue;
            }
            
            auto index = pIndex->second;
            if (tentativeIndices.erase(id))
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
            auto pIndex = tentativeIndices.find(id);
            
            if (pIndex == tentativeIndices.end()) {
#if (MUST_LINK)
                unseenFinalized.emplace_back(segment.centroid(linkagePolicy));
#else
                unseenFinalized.emplace_back(segment);
#endif
                continue;
            }
            
            tentativeIndices.erase(id);
#if (!MUST_LINK)
        }
#endif
    }

    // Remove unmatched indices
    for (auto [_, index]: tentativeIndices)
        remove(index, false);
    
    data->tentativeIndices = std::move(keptTentativeIndices);

    // Append new embeddings
    for (auto& embedding: unseenFinalized)
        insert(embedding, false);
    
    for (auto& embedding: unseenTentative)
        insert(embedding, true);
}

EmbeddingDistanceMatrix& EmbeddingDistanceMatrix::operator=(const EmbeddingDistanceMatrix& other) {
    if (this == &other) return *this;
    if (this->data == other.data) return *this;
    dec();
    matrix = other.matrix;
    _embeddings = other._embeddings;
    data = other.data;
    inc();
    return *this;
}

EmbeddingDistanceMatrix& EmbeddingDistanceMatrix::operator=(EmbeddingDistanceMatrix&& other) noexcept {
    if (this->data == other.data) return *this;
    dec();
    matrix = other.matrix;
    _embeddings = other._embeddings;
    data = other.data;
    other.matrix = nullptr;
    other._embeddings = nullptr;
    other.data = nullptr;
    return *this;
}

EmbeddingDistanceMatrix EmbeddingDistanceMatrix::gatherAndPop(std::span<const long> indices) {
    const auto k = static_cast<long>(indices.size());
    if (k == 0) return EmbeddingDistanceMatrix(linkagePolicy);
    
    EmbeddingDistanceMatrix result(linkagePolicy);
    result.reserve(k);
    
    result.data->size = k;
    result.data->embeddingCount = k;
    result.data->matrixEndIndex = k * (k - 1) / 2;
    this->data->embeddingCount -= k;

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
    
    auto myTentativeIndices = std::move(data->tentativeIndices);
    auto myFreeIndices = std::move(data->freeIndices);
    auto mySize = data->size;
    auto myMatrixEndIndex = data->matrixEndIndex;
    std::unordered_map<UUIDWrapper, long> splitTentativeIndices{};
    splitTentativeIndices.reserve(myTentativeIndices.size());
    
    for (auto oldIndex: indices) {
        // Put the embedding in the new matrix
        auto& dstEmbedding = result._embeddings[newIndex];
        dstEmbedding = std::move(this->_embeddings[oldIndex]);
        
        const auto embId = dstEmbedding.id();
        if (myTentativeIndices.erase(embId))
            splitTentativeIndices[embId] = newIndex;
        ++newIndex;
        
        // Free the spot in this matrix
        if (oldIndex < mySize - 1)
            myFreeIndices.emplace_back(oldIndex);
        else
            myMatrixEndIndex -= --mySize;
    }
    
    result.data->tentativeIndices = std::move(splitTentativeIndices);
    
    data->size = mySize;
    data->matrixEndIndex = myMatrixEndIndex;
    data->freeIndices = std::move(myFreeIndices);
    data->tentativeIndices = std::move(myTentativeIndices);
    
    return result;
}


void EmbeddingDistanceMatrix::absorb(const EmbeddingDistanceMatrix& other) {
    auto otherEmbeddingCount = other.data->embeddingCount;
    if (otherEmbeddingCount == 0) return;
    auto otherSize = other.data->size;
    auto otherTentativeIndices = std::move(other.data->tentativeIndices);
    
    auto myCapacity = data->capacity;
    auto myEmbeddingCount = data->embeddingCount;
    auto myMatrixEndIndex = data->matrixEndIndex;
    auto mySize = data->size;
    auto myFreeIndices = std::move(data->freeIndices);
    auto myTentativeIndices = std::move(data->tentativeIndices);
    
    // Ensure capacity for all incoming embeddings
    myEmbeddingCount += otherEmbeddingCount;
    if (myEmbeddingCount > myCapacity)
        reserve(std::max(myCapacity * 2, myEmbeddingCount));

    // Map other's indices to new indices in this matrix.
    auto oldToNew = std::make_unique<long[]>(otherSize); // index in other -> index in this
    auto isNew = std::make_unique<bool[]>(mySize);
    std::fill(oldToNew.get(), oldToNew.get() + otherSize, -1);
    const auto oldSize = mySize;
    
    // Transfer embeddings
    for (long i = 0; i < otherSize; ++i) {
        if (other._embeddings[i].expired())
            continue;
        
        // Get next index
        long nextIndex;
        if (!myFreeIndices.empty()) {
            nextIndex = myFreeIndices.back();
            myFreeIndices.pop_back();
            isNew[nextIndex] = true;
        } else {
            nextIndex = mySize;
            myMatrixEndIndex += mySize++;
        }
        oldToNew[i] = nextIndex;
        
        // Add the embedding
        auto& dstEmbedding = this->_embeddings[nextIndex];
        dstEmbedding = other._embeddings[i];
        
        auto embId = dstEmbedding.id();
        if (otherTentativeIndices.contains(embId))
            myTentativeIndices[embId] = nextIndex;
        
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
    for (long rOld = 1; rOld < otherSize; ++rOld) {
        const auto rNew = oldToNew[rOld];
        const auto srcBase = rNew * (rNew - 1) / 2;
        for (auto cOld = 0; cOld < rOld; ++cOld) {
            auto r = rNew, c = oldToNew[cOld];
            if (r < c) std::swap(r, c);
            this->matrix[r * (r - 1) / 2 + c] = other.matrix[srcBase + cOld];
        }
    }
    
    other.data->tentativeIndices = std::move(otherTentativeIndices);
    
    data->size = mySize;
    data->embeddingCount = myEmbeddingCount;
    data->matrixEndIndex = myMatrixEndIndex;
    data->freeIndices = std::move(myFreeIndices);
    data->tentativeIndices = std::move(myTentativeIndices);
}

Dendrogram EmbeddingDistanceMatrix::dendrogram(bool ignoreTentative) const {
    return Dendrogram(*this, ignoreTentative);
}
