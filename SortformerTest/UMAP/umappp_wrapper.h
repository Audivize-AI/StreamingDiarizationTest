#ifndef UMAPPP_WRAPPER_H
#define UMAPPP_WRAPPER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void* UmapStatusPtr;

// Initialize UMAP with pre-computed KNN graph
// n: number of embeddings
// d: dimension of output embedding (usually 2)
// k: number of neighbors
// min_dist: min_dist parameter
// knn_indices: Flattened array of indices (n * k)
// knn_distances: Flattened array of distances (n * k)
// output: Buffer for embedding (n * d). Will be initialized with random start positions.
// seed: Random seed
// Returns: Opaque pointer to UMAP status/state
UmapStatusPtr umap_initialize(
    int n,
    int d,
    int k,
    float min_dist,
    const int* knn_indices,
    const float* knn_distances,
    float* output,
    int seed
);

// Run UMAP optimization for a specific number of epochs
// status: Handle returned by initialize
// output: Buffer for embedding (n * d). Must differ from initialize if you want to keep initial state, but typically same buffer is used.
// epoch_limit: 0 for all epochs (default ~200-500), or a specific number to run partial update.
// Returns: Current epoch number reached
int umap_run(UmapStatusPtr status, float* output, int epoch_limit);

// Get total number of epochs planned
int umap_num_epochs(UmapStatusPtr status);

// Free resources
void umap_free(UmapStatusPtr status);

#ifdef __cplusplus
}
#endif

#endif
