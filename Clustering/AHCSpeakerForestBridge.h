#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface AHCEmbeddingSample : NSObject
@property(nonatomic, readonly) NSUUID *identifier;
@property(nonatomic, readonly) NSInteger speakerIndex;
@property(nonatomic, readonly) float weight;
@property(nonatomic, readonly) NSData *vectorData;

- (instancetype)initWithIdentifier:(NSUUID *)identifier
                      speakerIndex:(NSInteger)speakerIndex
                            weight:(float)weight
                        vectorData:(NSData *)vectorData NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
@end

@interface AHCEmbeddingSegmentInput : NSObject
@property(nonatomic, readonly) NSInteger speakerIndex;
@property(nonatomic, readonly) NSArray<AHCEmbeddingSample *> *embeddings;

- (instancetype)initWithSpeakerIndex:(NSInteger)speakerIndex
                          embeddings:(NSArray<AHCEmbeddingSample *> *)embeddings NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
@end

@interface AHCDendrogramNodeSnapshot : NSObject
@property(nonatomic, readonly) NSInteger index;
@property(nonatomic, readonly) NSInteger matrixIndex;
@property(nonatomic, readonly) NSInteger leftChild;
@property(nonatomic, readonly) NSInteger rightChild;
@property(nonatomic, readonly) NSInteger speakerIndex;
@property(nonatomic, readonly) NSInteger count;
@property(nonatomic, readonly) float weight;
@property(nonatomic, readonly) float mergeDistance;
@property(nonatomic, readonly) BOOL mustLink;

- (instancetype)initWithIndex:(NSInteger)index
                  matrixIndex:(NSInteger)matrixIndex
                    leftChild:(NSInteger)leftChild
                   rightChild:(NSInteger)rightChild
                  speakerIndex:(NSInteger)speakerIndex
                        count:(NSInteger)count
                       weight:(float)weight
                mergeDistance:(float)mergeDistance
                     mustLink:(BOOL)mustLink NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
@end

@interface AHCDendrogramSnapshot : NSObject
@property(nonatomic, readonly) NSInteger rootIndex;
@property(nonatomic, readonly) NSInteger nodeCount;
@property(nonatomic, readonly) NSInteger activeLeafCount;
@property(nonatomic, readonly) NSArray<AHCDendrogramNodeSnapshot *> *nodes;

- (instancetype)initWithRootIndex:(NSInteger)rootIndex
                        nodeCount:(NSInteger)nodeCount
                  activeLeafCount:(NSInteger)activeLeafCount
                            nodes:(NSArray<AHCDendrogramNodeSnapshot *> *)nodes NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
@end

@interface AHCSpeakerForestBridge : NSObject

- (instancetype)initWithNumRepresentatives:(NSInteger)numRepresentatives
                             minEmbeddings:(NSInteger)minEmbeddings
                             maxEmbeddings:(NSInteger)maxEmbeddings NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

- (void)reset;

- (BOOL)streamWithFinalizedSegments:(NSArray<AHCEmbeddingSegmentInput *> *)finalized
                  tentativeSegments:(NSArray<AHCEmbeddingSegmentInput *> *)tentative;

- (AHCDendrogramSnapshot *)dendrogramSnapshot;

@end

NS_ASSUME_NONNULL_END
