#import "AHCSpeakerForestBridge.h"

#include <span>
#include <utility>
#include <vector>

@interface AHCDistanceMatrix () {
@private
    CDistanceMatrix _matrix;
}

- (instancetype)initWithRawMatrix:(CDistanceMatrix &&)matrix;

@end

@implementation AHCDistanceMatrix

- (instancetype)initWithLinkagePolicy:(const LinkagePolicy *)linkagePolicy {
    self = [super init];
    if (self != nil) {
        _matrix = CDistanceMatrix(linkagePolicy);
    }
    return self;
}

- (instancetype)initWithRawMatrix:(CDistanceMatrix &&)matrix {
    self = [super init];
    if (self != nil) {
        _matrix = std::move(matrix);
    }
    return self;
}

- (NSInteger)embeddingCount {
    return static_cast<NSInteger>(_matrix.embeddingCount());
}

- (NSInteger)finalizedCount {
    return static_cast<NSInteger>(_matrix.finalizedCount());
}

- (NSInteger)tentativeCount {
    return static_cast<NSInteger>(_matrix.tentativeCount());
}

- (NSInteger)size {
    return static_cast<NSInteger>(_matrix.size());
}

- (void)reserve:(NSInteger)newCapacity {
    _matrix.reserve(static_cast<long>(newCapacity));
}

- (void)insertEmbedding:(const SpeakerEmbeddingWrapper &)embedding tentative:(BOOL)isTentative {
    _matrix.insert(embedding, isTentative);
}

- (SpeakerEmbeddingWrapper)embedding:(NSInteger)index {
    return _matrix.embedding(static_cast<long>(index));
}

- (void)streamWithFinalized:(EmbeddingSegmentWrapper *_Nullable)finalized
             finalizedCount:(NSInteger)finalizedCount
                  tentative:(EmbeddingSegmentWrapper *_Nullable)tentative
             tentativeCount:(NSInteger)tentativeCount {
    std::span<EmbeddingSegmentWrapper> finalizedSpan{};
    std::span<EmbeddingSegmentWrapper> tentativeSpan{};

    if (finalized != nullptr && finalizedCount > 0) {
        finalizedSpan = std::span<EmbeddingSegmentWrapper>(
            finalized,
            static_cast<std::size_t>(finalizedCount)
        );
    }
    if (tentative != nullptr && tentativeCount > 0) {
        tentativeSpan = std::span<EmbeddingSegmentWrapper>(
            tentative,
            static_cast<std::size_t>(tentativeCount)
        );
    }

    _matrix.stream(finalizedSpan, tentativeSpan);
}

- (Dendrogram)dendrogram:(BOOL)ignoreTentative {
    return _matrix.dendrogram(ignoreTentative);
}

- (AHCDistanceMatrix *)gatherAndPopIndices:(const long *_Nullable)indices count:(NSInteger)count {
    std::span<const long> indexSpan{};
    if (indices != nullptr && count > 0) {
        indexSpan = std::span<const long>(indices, static_cast<std::size_t>(count));
    }

    auto matrix = _matrix.gatherAndPop(indexSpan);
    return [[AHCDistanceMatrix alloc] initWithRawMatrix:std::move(matrix)];
}

- (void)absorb:(AHCDistanceMatrix *)other {
    if (other == nil) return;
    _matrix.absorb(other->_matrix);
}

- (void)insertTentativeFrom:(AHCDistanceMatrix *)other {
    if (other == nil) return;
    _matrix.insertTentativeFrom(other->_matrix);
}

- (SpeakerEmbeddingWrapper)computeCentroidWithPolicy:(const LinkagePolicy *)policy
                                             cluster:(const Cluster &)cluster {
    return policy->computeCentroid(_matrix.embeddings(), cluster);
}

@end
