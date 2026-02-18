#pragma once

struct ClusteringConfig {
    float minSeparation = 0.1;
    float mergeThreshold = 0.3;
    long maxEmbeddings = 20;
    long minEmbeddings = 8;
    long maxRepresentatives = 4;
};