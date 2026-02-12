//
// Created by Benjamin Lee on 1/29/26.
//
#include "EmbeddingDistanceMatrix.hpp"
#include <iostream>


template<LinkagePolicy LP>
EmbeddingDistanceMatrix<LP>::EmbeddingDistanceMatrix(): matrix(nullptr), embeddings(nullptr) {}

template<LinkagePolicy LP>
EmbeddingDistanceMatrix<LP>::EmbeddingDistanceMatrix(EmbeddingDistanceMatrix&& other) noexcept :
    matrix(other.matrix),
    embeddings(other.embeddings),
    freeIndices(std::move(other.freeIndices)),
    tentativeIndices(std::move(other.tentativeIndices)),
    idToIndex(std::move(other.idToIndex)),
    mustLinkIndices(std::move(other.mustLinkIndices)),
    tentativeMustLinkIndices(std::move(other.tentativeMustLinkIndices)),
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
    auto newEmbeddings = new Embedding[newCapacity]{};
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
    
    idToIndex.reserve(newCapacity);

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
void EmbeddingDistanceMatrix<LP>::insert(Embedding& embedding, bool isTentative) {
    long index, matrixIndex;
    if (freeIndices.empty()) {
        index = _size;
        // Ensure there is enough _capacity if appending to the end
        if (_size + 1 > _capacity) {
            reserve(std::max(_capacity * 2, _size + 1));
        }
        _size++;
        matrixEndIndex = _size * (_size - 1) / 2;
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

    embedding.takeOwnership();
    embedding.setMatrixIndex(index);
    embeddings[index] = embedding;
    idToIndex[embedding.id()] = index;
    ++_embeddingCount;

    if (isTentative)
        tentativeIndices.emplace_back(index);
}

template<LinkagePolicy LP>
void EmbeddingDistanceMatrix<LP>::replace(long index, Embedding& embedding) {
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

    embedding.takeOwnership();
    embedding.setMatrixIndex(index);
    idToIndex.erase(embeddings[index].id());
    idToIndex[embedding.id()] = index;
    embeddings[index] = embedding;
}

template<LinkagePolicy LP>
void EmbeddingDistanceMatrix<LP>::remove(long index) {
    if (index >= _size || embeddings[index].expired())
        return;
    if (index < _size - 1)
        freeIndices.emplace_back(index);
    else
        --_size;
    
    embeddings[index].releaseVector();
    idToIndex.erase(embeddings[index].id());
    --_embeddingCount;
}

template<LinkagePolicy LP>
void EmbeddingDistanceMatrix<LP>::stream(std::vector<EmbeddingSegment>& newFinalized, std::vector<EmbeddingSegment>& newTentative) {
    std::vector<Embedding*> unseenFinalized = {};
    std::vector<Embedding*> unseenTentative = {};
    std::vector<std::ptrdiff_t> keptTentativeIndices = {};
    unseenFinalized.reserve(newFinalized.size());
    unseenTentative.reserve(newTentative.size());
    keptTentativeIndices.reserve(tentativeIndices.size());

    // Filter for unmatched embeddings in both the tentative indices and the incoming embeddings
    for (auto& segment : newTentative) {
        for (auto& embedding : segment.embeddings) {
            bool isNew = true;
            for (std::size_t i = 0; i < tentativeIndices.size();) {
                if (embedding == embeddings[tentativeIndices[i]]) {
                    isNew = false;
                    keptTentativeIndices.push_back(tentativeIndices[i]);
                    tentativeIndices.erase(tentativeIndices.begin() + i);
                    break;
                }
                ++i;
            }
            if (isNew)
                unseenTentative.push_back(&embedding);
        }
    }

    for (auto& segment : newFinalized) {
        for (auto& embedding: segment.embeddings) {
            bool isNew = true;
            for (std::size_t i = 0; i < tentativeIndices.size();) {
                if (embedding == embeddings[tentativeIndices[i]]) {
                    isNew = false;
                    tentativeIndices.erase(tentativeIndices.begin() + i);
                    break;
                }
                ++i;
            }
            if (isNew)
                unseenFinalized.push_back(&embedding);
        }
    }

    // Remove unmatched indices
    for (auto index: tentativeIndices)
        remove(index);
    tentativeIndices = std::move(keptTentativeIndices);

    // Append new embeddings
    for (auto pEmbedding: unseenFinalized)
        insert(*pEmbedding, false);

    for (auto pEmbedding: unseenTentative)
        insert(*pEmbedding, true);

    updateMustLinkConstraints(newFinalized, newTentative);
}

template<LinkagePolicy LP>
void EmbeddingDistanceMatrix<LP>::updateMustLinkConstraints(std::vector<EmbeddingSegment> const& finalized, std::vector<EmbeddingSegment> const& tentative) {
    this->mustLinkIndices.clear();
    this->tentativeMustLinkIndices.clear();

    auto appendConstraints = [this](std::vector<MustLinkConstraint>& output, std::vector<EmbeddingSegment> const& segments) -> long {
        long dropped = 0;
        for (auto const& segment: segments) {
            if (segment.embeddings.size() < 2) continue;

            auto constraint = MustLinkConstraint(segment.embeddings.size());
            bool allPresent = true;

            for (std::size_t i = 0; i < segment.embeddings.size(); ++i) {
                auto iter = this->idToIndex.find(segment.embeddings[i].id());
                if (iter == this->idToIndex.end()) {
                    allPresent = false;
                    break;
                }
                constraint.indices[i] = iter->second;
            }

            if (allPresent) {
                output.emplace_back(std::move(constraint));
            } else {
                ++dropped;
            }
        }
        return dropped;
    };

    const auto droppedTentative = appendConstraints(this->tentativeMustLinkIndices, tentative);
    const auto droppedFinalized = appendConstraints(this->mustLinkIndices, finalized);
    if (droppedTentative > 0 || droppedFinalized > 0) {
        std::cerr << "[AHC] Dropped must-link constraints with missing ids"
                  << " (finalized=" << droppedFinalized
                  << ", tentative=" << droppedTentative << ")" << std::endl;
    }
}

template<LinkagePolicy LP>
EmbeddingDistanceMatrix<LP>& EmbeddingDistanceMatrix<LP>::operator=(EmbeddingDistanceMatrix&& other) noexcept {
    delete[] matrix;
    delete[] embeddings;
    matrix = other.matrix;
    embeddings = other.embeddings;
    freeIndices = std::move(other.freeIndices);
    tentativeIndices = std::move(other.tentativeIndices);
    idToIndex = std::move(other.idToIndex);
    mustLinkIndices = std::move(other.mustLinkIndices);
    tentativeMustLinkIndices = std::move(other.tentativeMustLinkIndices);
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

template class EmbeddingDistanceMatrix<WardLinkage>;
template class EmbeddingDistanceMatrix<CosineAverageLinkage>;
