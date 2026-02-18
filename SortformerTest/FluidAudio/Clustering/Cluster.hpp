#pragma once

#include "SpeakerEmbeddingWrapper.hpp"
#include <memory>
#include <unordered_set>

struct DendrogramNode {
    long matrixIndex{0};
    long leftChild{-1};
    long rightChild{-1};
    long count{1};
    long segmentCount{0};
    float weight{1};
    float mergeDistance{0};
};

class Cluster {
private:
    std::shared_ptr<long[]> indices{nullptr};
    long _count;
    long _segmentCount;
    float _weight;
    float _spread;

public:
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using value_type        = long;
        using difference_type   = std::ptrdiff_t;
        using pointer           = long*;
        using reference         = long&;
        
        pointer ptr;
        inline reference operator*() const { return *ptr; }
        inline pointer operator->() const { return ptr; }
        inline Iterator& operator++() { ++ptr; return *this; }
        inline Iterator operator++(int) { auto tmp = *this; ++ptr; return tmp; }
        friend bool operator== (const Iterator& a, const Iterator& b) = default;
        friend bool operator!= (const Iterator& a, const Iterator& b) = default;
    };

    inline Cluster(Cluster&& other) noexcept: 
            indices(std::move(other.indices)),
            _count(other._count), 
            _segmentCount(other._segmentCount),
            _weight(other._weight), 
            _spread(other._spread) {}
            
    inline Cluster(const Cluster& other) = default;
    explicit inline Cluster(const DendrogramNode& node):
            indices(std::make_shared<long[]>(node.count)),
            _count(node.count),
            _segmentCount(node.segmentCount),
            _weight(node.weight),
            _spread(node.mergeDistance) {}

    inline long& operator[](long i) { return indices[i]; }
    inline long operator[](long i) const { return indices[i]; }

    [[nodiscard]] inline long count() const { return _count; }
    [[nodiscard]] inline long segmentCount() const { return _count; }
    [[nodiscard]] inline float weight() const { return _weight; }
    [[nodiscard]] inline float spread() const { return _spread; }

    [[nodiscard]] inline long front() const { return indices[0]; }
    [[nodiscard]] inline long back() const { return indices[_count - 1]; }
    
    [[nodiscard]] inline Iterator begin() const { return Iterator(indices.get()); }
    [[nodiscard]] inline Iterator end() const { return Iterator(indices.get() + _count); }

    inline Cluster& operator=(const Cluster& cluster) = default;
    inline Cluster& operator=(Cluster&& cluster) = default;
};