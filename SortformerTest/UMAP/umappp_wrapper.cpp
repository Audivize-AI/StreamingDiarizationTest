#include "umappp_wrapper.h"
#include <vector>
#include <memory>
#include <iostream>

// Include the UMAPPP library headers
// Using relative path to avoid needing header search path config immediately
#include "include/umappp/umappp.hpp" 

struct UmapStatusWrapper {
    std::unique_ptr<umappp::Status<int, float>> status;
};

extern "C" {

UmapStatusPtr umap_initialize(
    int n,
    int d,
    int k,
    float min_dist,
    const int* knn_indices,
    const float* knn_distances,
    float* output,
    int seed
) {
    if (n <= 0 || k <= 0) return nullptr;

    // Convert flat arrays to NeighborList
    umappp::NeighborList<int, float> neighbors(n);
    for (int i = 0; i < n; ++i) {
        neighbors[i].reserve(k);
        for (int j = 0; j < k; ++j) {
            int idx = knn_indices[i * k + j];
            float dist = knn_distances[i * k + j];
            if (idx >= 0 && idx < n) {
                neighbors[i].push_back({idx, dist});
            }
        }
    }

    umappp::Options opt;
    opt.min_dist = min_dist;
    // We strictly control neighbors via input graph, but setting this is good practice
    opt.num_neighbors = k; 
    
    // Use RANDOM initialization to avoid heavy dependencies (Eigen/ARPACK/Id) required for spectral
    opt.initialize_method = umappp::InitializeMethod::RANDOM; 
    opt.initialize_seed = seed;
    
    // Disable multi-threading for simplicity on mobile/embedded
    opt.num_threads = 1; 

    try {
        // Initialize status and initial embedding
        auto status = umappp::initialize(std::move(neighbors), d, output, opt);
        auto wrapper = new UmapStatusWrapper();
        wrapper->status = std::make_unique<umappp::Status<int, float>>(std::move(status));
        return wrapper;
    } catch (const std::exception& e) {
        std::cerr << "UMAP Init Error: " << e.what() << std::endl;
        return nullptr;
    } catch (...) {
        return nullptr;
    }
}

int umap_run(UmapStatusPtr status_ptr, float* output, int epoch_limit) {
    auto wrapper = static_cast<UmapStatusWrapper*>(status_ptr);
    if (!wrapper || !wrapper->status) return -1;
    
    // If epoch_limit <= 0, run to completion.
    // If epoch_limit > 0, execute until that epoch is reached (or completion)
    if (epoch_limit <= 0) {
        wrapper->status->run(output);
    } else {
        wrapper->status->run(output, epoch_limit);
    }
    
    return wrapper->status->epoch();
}

int umap_num_epochs(UmapStatusPtr status_ptr) {
    auto wrapper = static_cast<UmapStatusWrapper*>(status_ptr);
    if (!wrapper || !wrapper->status) return 0;
    return wrapper->status->num_epochs();
}

void umap_free(UmapStatusPtr status_ptr) {
    delete static_cast<UmapStatusWrapper*>(status_ptr);
}

}
