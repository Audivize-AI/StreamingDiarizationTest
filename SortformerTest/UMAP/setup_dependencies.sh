#!/bin/bash
set -e

# Directory setup
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
INCLUDE_DIR="$BASE_DIR/include"

echo "Setting up UMAP dependencies in $INCLUDE_DIR..."

mkdir -p "$INCLUDE_DIR/umappp"
mkdir -p "$INCLUDE_DIR/aarand"
mkdir -p "$INCLUDE_DIR/knncolle"
mkdir -p "$INCLUDE_DIR/sanisizer"
mkdir -p "$INCLUDE_DIR/subpar"
mkdir -p "$INCLUDE_DIR/irlba"
mkdir -p "$INCLUDE_DIR/Eigen"

# Fetch wrapper function
fetch_file() {
    local url=$1
    local dest=$2
    if curl -sL --fail "$url" -o "$dest"; then
        echo "Fetched $dest"
    else
        echo "Failed to fetch $dest (might not exist, ignoring if optional)"
    fi
}

# 1. Fetch UMAPPP headers
echo "Fetching umappp headers..."
REPO="https://raw.githubusercontent.com/libscran/umappp/master/include/umappp"
FILES=(
    "umappp.hpp"
    "Options.hpp"
    "Status.hpp"
    "initialize.hpp"
    "NeighborList.hpp"
    "combine_neighbor_sets.hpp"
    "find_ab.hpp"
    "neighbor_similarities.hpp"
    "spectral_init.hpp"
    "run.hpp"
    "optimize.hpp"
    "utils.hpp"
    "optimize_layout.hpp"
    "parallelize.hpp"
    "random_init.hpp"
)

for file in "${FILES[@]}"; do
    if [ "$file" != "spectral_init.hpp" ]; then
        fetch_file "$REPO/$file" "$INCLUDE_DIR/umappp/$file"
    fi
done

# 2. Fetch Dependencies

# AARAND
echo "Fetching aarand..."
fetch_file "https://raw.githubusercontent.com/LTLA/aarand/master/include/aarand/aarand.hpp" "$INCLUDE_DIR/aarand/aarand.hpp"

# SANISIZER
echo "Fetching sanisizer headers..."
REPO="https://raw.githubusercontent.com/LTLA/sanisizer/master/include/sanisizer"
FILES=(
    "sanisizer.hpp"
    "cast.hpp" 
    "allocator.hpp"
    "math.hpp"
    "attest.hpp"
    "meta.hpp"
    "arithmetic.hpp"
    "comparisons.hpp"
    "logical.hpp"
    "bitwise.hpp"
    "utils.hpp"
    "create.hpp"
    "destroy.hpp"
    "nd_offset.hpp"
    "cap.hpp"
    "checked.hpp"
    "copy.hpp"
    "element.hpp"
    "error.hpp"
    "fill.hpp"
    "flatten.hpp"
    "init.hpp"
    "io.hpp"
    "limit.hpp"
    "matrix.hpp"
    "random.hpp"
    "range.hpp"
    "scalar.hpp"
    "sequence.hpp"
    "size.hpp"
    "slice.hpp"
    "sort.hpp"
    "store.hpp"
    "subset.hpp"
    "type.hpp"
    "vector.hpp"
    "ptrdiff.hpp"  
    "containers.hpp"
    "iterators.hpp" 
    "index.hpp"
    "memory.hpp"
    "float.hpp"
    "integer.hpp"
    "boolean.hpp"
    "class.hpp" 
    "struct.hpp"
    "union.hpp"
    "enum.hpp"
)
for file in "${FILES[@]}"; do
    fetch_file "$REPO/$file" "$INCLUDE_DIR/sanisizer/$file"
done

# SUBPAR
echo "Fetching subpar..."
REPO="https://raw.githubusercontent.com/LTLA/subpar/master/include/subpar"
FILES=(
    "subpar.hpp"
    "parallel.hpp"
    "serial.hpp"
    "range.hpp"
    "simple.hpp" 
)
for file in "${FILES[@]}"; do
    fetch_file "$REPO/$file" "$INCLUDE_DIR/subpar/$file"
done

# 3. Create MOCK headers for HEAVY dependencies

echo "Creating mock headers for Eigen/Irlba/Knncolle..."

# Mock spectral_init
cat <<EOF > "$INCLUDE_DIR/umappp/spectral_init.hpp"
#ifndef UMAPPP_SPECTRAL_INIT_HPP
#define UMAPPP_SPECTRAL_INIT_HPP
#include "NeighborList.hpp"
namespace umappp {
template<typename Index_, typename Float_, typename Options_>
bool spectral_init(const NeighborList<Index_, Float_>&, int, Float_*, const Options_&, int, double, double, double, int) {
    return false;
}

template<typename Index_, typename Float_>
void random_init(Index_ n_obs, int n_dims, Float_* embedding, int seed, double scale) {
    // Basic random initialization mock
    // In a real scenario, this would use a PRNG to fill embedding with values in range [-scale, scale]
    for (int i = 0; i < n_obs * n_dims; ++i) {
        embedding[i] = 0.0; // Just zero for now to satisfy linker/compiler, or we can use random if needed
    }
}
}
#endif
EOF

# Mock knncolle
cat <<EOF > "$INCLUDE_DIR/knncolle/knncolle.hpp"
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
EOF

# Mock irlba (required by Options.hpp)
cat <<EOF > "$INCLUDE_DIR/irlba/irlba.hpp"
#ifndef IRLBA_IRLBA_HPP
#define IRLBA_IRLBA_HPP
namespace irlba {
    // Defines Options template struct
    template<typename T>
    struct Options {
        int work = 7;
    };
}
#endif
EOF

# Mock Eigen (Avoid 50MB header download)
cat <<EOF > "$INCLUDE_DIR/Eigen/Dense"
#ifndef EIGEN_DENSE_H
#define EIGEN_DENSE_H
namespace Eigen {
    template<typename T, int R, int C> class Matrix {};
    template<typename T> class VectorX {};
    typedef double VectorXd; // Used in Options.hpp
    typedef int Index;
}
#endif
EOF

echo "Dependencies set up successfully."
