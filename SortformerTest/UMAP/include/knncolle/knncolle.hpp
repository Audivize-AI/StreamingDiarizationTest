#ifndef KNNCOLLE_KNNCOLLE_HPP
#define KNNCOLLE_KNNCOLLE_HPP
#include <vector>
#include <utility>

namespace knncolle {

template<typename Index, typename Float>
using NeighborList = std::vector<std::vector<std::pair<Index, Float>>>;

// Basic Builder/Prebuilt mocks needed by initialize.hpp logic if it expands templates
template<typename Matrix, typename Index, typename Float>
class Base {
public:
    virtual ~Base() = default;
    virtual NeighborList<Index, Float> find_nearest_neighbors(Index, int) const = 0;
};

// find_nearest_neighbors free function mock
template<typename Index = int, typename Float = double, typename Matrix>
NeighborList<Index, Float> find_nearest_neighbors(const Matrix&, int, int = 1) {
    return {}; 
}

template<typename A, typename B, typename C, typename D> class Builder {};
template<typename A, typename B> class SimpleMatrix { public: SimpleMatrix(size_t, size_t, const B*) {} };
template<typename A, typename B> class Matrix {};
template<typename A, typename B, typename C> class Prebuilt {};

}
#endif
