//
// Created by Benjamin Lee on 1/29/26.
//

#pragma once

#include "EmbeddingDistanceMatrix.hpp"
#include "Cluster.hpp"
#include "ClusterLinkage.hpp"
#include <cmath>
#include <limits>
#include <unordered_set>

enum class ClusteringFilter: char {
    none      = '\x0',
    all       = '\x3',
    finalized = '\x1',
    tentative = '\x2',
};

template<LinkagePolicy LP>
class Dendrogram {
public:
    static constexpr DendrogramNode nullNode = {-1, -1, -1, -1, -1.f, 0};
    
    Dendrogram() = default;
    Dendrogram(Dendrogram const& clustering) = default;
    Dendrogram(Dendrogram&& clustering) noexcept ;
    explicit Dendrogram(EmbeddingDistanceMatrix<LP>& embeddingMatrix, ClusteringFilter filter);
    explicit Dendrogram(EmbeddingDistanceMatrix<LP>& embeddingMatrix, bool ignoreTentative = false);
    
    [[nodiscard]] inline long nodeCount() const { return count; }
    [[nodiscard]] inline std::shared_ptr<DendrogramNode[]> nodes() const { return dendrogram; }
    [[nodiscard]] inline DendrogramNode root() const { return dendrogram ? dendrogram[rootId] : nullNode; }
    [[nodiscard]] inline long rootIndex() const { return rootId; }
    [[nodiscard]] std::vector<Cluster> extractClusters(float linkageThreshold = -1.f, long maxClusters = -1) const;
    
    Dendrogram& operator=(const Dendrogram&) = default;
    Dendrogram& operator=(Dendrogram&&) = default;
private:
    struct Aux {
        float* matrix{nullptr};
        bool* activeFlags{nullptr};
        long* matrixToNode{nullptr};
        std::vector<std::ptrdiff_t> freeIndices{};
        long size{0};
        long matrixEndIndex{0};
        long numClustersRemaining{0};
        long firstActiveMatrixIndex{0};
        long numMerged{0};
        float maxGap{0};
        
        ~Aux() {
            delete[] matrix;
            delete[] activeFlags;
            delete[] matrixToNode;
        }
    };
    
    long count{0};
    std::shared_ptr<DendrogramNode[]> dendrogram{nullptr};
    long rootId{-1};
    float elbowLinkage{0};
    
    void buildDendrogram(Aux& aux);
    void applyConstraints(std::vector<MustLinkConstraint> const& constraints, Aux& aux);
    
    // Merge two rows in the matrix to make a new cluster and return the row of the merged buildDendrogram
    std::ptrdiff_t merge(std::ptrdiff_t leftIndex, std::ptrdiff_t rightIndex, Aux& aux, bool mustLink = false);
    
    // Get the index of the nearest neighbor to a matrix row
    [[nodiscard]] static std::ptrdiff_t nearestNeighbor(std::ptrdiff_t index, Aux& aux) ;

    // Get the _spread between the cluster at index row and the buildDendrogram at index col where row != col
    [[nodiscard]] static inline float distance(std::ptrdiff_t row, std::ptrdiff_t col, Aux& aux) {
        if (row < 0 || col < 0 || row >= aux.size || col >= aux.size || aux.matrix == nullptr) {
            return std::numeric_limits<float>::infinity();
        }
        if (col > row) std::swap(row, col);
        const auto index = row * (row - 1) / 2 + col;
        if (index < 0 || index >= aux.matrixEndIndex) {
            return std::numeric_limits<float>::infinity();
        }
        const auto value = aux.matrix[index];
        if (!std::isfinite(value)) {
            return std::numeric_limits<float>::infinity();
        }
        return value;
    
    }
    [[nodiscard]] Cluster collectCluster(const DendrogramNode &root) const;
};

// Overload the bitwise AND operator
constexpr ClusteringFilter operator&(ClusteringFilter lhs, ClusteringFilter rhs) {
    return static_cast<ClusteringFilter>(
            static_cast<std::underlying_type_t<ClusteringFilter>>(lhs) &
            static_cast<std::underlying_type_t<ClusteringFilter>>(rhs)
    );
}

// Overload the bitwise OR operator
constexpr ClusteringFilter operator|(ClusteringFilter lhs, ClusteringFilter rhs) {
    return static_cast<ClusteringFilter>(
            static_cast<std::underlying_type_t<ClusteringFilter>>(lhs) |
            static_cast<std::underlying_type_t<ClusteringFilter>>(rhs)
    );
}

// Overload the bitwise NOT operator
constexpr ClusteringFilter operator~(ClusteringFilter rhs) {
    return static_cast<ClusteringFilter>(~static_cast<std::underlying_type_t<ClusteringFilter>>(rhs));
}
