#pragma once

#include "Embedding.hpp"
#include <memory>
#include <unordered_set>

struct MustLinkConstraint {
    std::unique_ptr<long[]> indices;
    std::size_t count;

    explicit MustLinkConstraint(std::size_t count): indices(std::make_unique<long[]>(count)), count(count) {}
    MustLinkConstraint(MustLinkConstraint&& other) noexcept: indices(std::move(other.indices)), count(other.count) {}
};

struct DendrogramNode {
    long matrixIndex{0};
    long leftChild{-1};
    long rightChild{-1};
    long count{1};
    float weight{1};
    float mergeDistance{0};
    bool mustLink=false;
};

class Cluster {
private:
    std::shared_ptr<long[]> indices{nullptr};
    long _count;
    float _weight;
    float _spread;

public:
    struct Iterator {
        long* ptr;
        inline long& operator*() const { return *ptr; }
        inline long* operator->() const { return ptr; }
        inline Iterator& operator++() { ++ptr; return *this; }
        inline Iterator operator++(int) { auto tmp = *this; ++ptr; return tmp; }
        friend bool operator== (const Iterator& a, const Iterator& b) = default;
        friend bool operator!= (const Iterator& a, const Iterator& b) = default;
    };

    inline Cluster(Cluster&& other) noexcept = default;
    inline Cluster(const Cluster& other) = default;
    explicit inline Cluster(const DendrogramNode& node):
            indices(std::make_shared<long[]>(node.count)),
            _count(node.count),
            _weight(node.weight),
            _spread(node.mergeDistance) {}

    inline long& operator[](long i) const { return indices[i]; };

    [[nodiscard]] inline long count() const { return _count; }
    [[nodiscard]] inline float weight() const { return _weight; }
    [[nodiscard]] inline float spread() const { return _spread; }

    [[nodiscard]] inline long front() const { return indices[0]; }
    [[nodiscard]] inline long back() const { return indices[_count - 1]; }
    
    [[nodiscard]] inline Iterator begin() const { return Iterator(indices.get()); }
    [[nodiscard]] inline Iterator end() const { return Iterator(indices.get() + _count); }

    [[nodiscard]] inline Cluster& operator=(const Cluster& cluster) = default;
    [[nodiscard]] inline Cluster& operator=(Cluster&& cluster) = default;
};