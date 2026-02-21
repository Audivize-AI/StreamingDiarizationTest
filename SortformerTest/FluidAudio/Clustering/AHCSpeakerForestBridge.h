#pragma once

#import <Foundation/Foundation.h>

#ifdef __cplusplus
#include "CDistanceMatrix.hpp"
#include "Cluster.hpp"
#include "Dendrogram.hpp"
#endif

NS_ASSUME_NONNULL_BEGIN

@interface AHCDistanceMatrix : NSObject

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithLinkagePolicy:(const LinkagePolicy *)linkagePolicy;

- (NSInteger)embeddingCount;
- (NSInteger)finalizedCount;
- (NSInteger)tentativeCount;
- (NSInteger)size;

- (void)reserve:(NSInteger)newCapacity;
- (void)insertEmbedding:(const SpeakerEmbeddingWrapper &)embedding tentative:(BOOL)isTentative;
- (SpeakerEmbeddingWrapper)embedding:(NSInteger)index;

- (void)streamWithFinalized:(EmbeddingSegmentWrapper *_Nullable)finalized
             finalizedCount:(NSInteger)finalizedCount
                  tentative:(EmbeddingSegmentWrapper *_Nullable)tentative
             tentativeCount:(NSInteger)tentativeCount;

- (Dendrogram)dendrogram:(BOOL)ignoreTentative;
- (AHCDistanceMatrix *)gatherAndPopIndices:(const long *_Nullable)indices count:(NSInteger)count;
- (void)absorb:(AHCDistanceMatrix *)other;
- (void)insertTentativeFrom:(AHCDistanceMatrix *)other;

- (SpeakerEmbeddingWrapper)computeCentroidWithPolicy:(const LinkagePolicy *)policy
                                             cluster:(const Cluster &)cluster;

@end

NS_ASSUME_NONNULL_END
