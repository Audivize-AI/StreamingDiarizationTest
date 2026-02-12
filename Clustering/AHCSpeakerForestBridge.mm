#import "AHCSpeakerForestBridge.h"

#include <algorithm>
#include <memory>
#include <vector>

#include "SpeakerForest.hpp"

namespace {
constexpr NSInteger kEmbeddingDimensions = static_cast<NSInteger>(Embedding::dims);

UUID uuidFromNSUUID(NSUUID *nsuuid) {
    uuid_t bytes = {0};
    [nsuuid getUUIDBytes:bytes];
    return UUID(bytes);
}

Embedding embeddingFromSample(AHCEmbeddingSample *sample) {
    const auto *values = static_cast<const float *>(sample.vectorData.bytes);
    auto uuid = uuidFromNSUUID(sample.identifier);
    return Embedding(uuid, values, static_cast<long>(sample.speakerIndex), true, sample.weight);
}

std::vector<EmbeddingSegment> convertSegments(NSArray<AHCEmbeddingSegmentInput *> *segments) {
    std::vector<EmbeddingSegment> output;
    output.reserve(segments.count);

    for (AHCEmbeddingSegmentInput *segment in segments) {
        EmbeddingSegment cppSegment(static_cast<long>(segment.speakerIndex), static_cast<long>(segment.embeddings.count));
        for (AHCEmbeddingSample *sample in segment.embeddings) {
            if (sample.vectorData.length != kEmbeddingDimensions * static_cast<NSInteger>(sizeof(float))) {
                continue;
            }

            auto embedding = embeddingFromSample(sample);
            cppSegment.add(std::move(embedding));
        }

        if (!cppSegment.embeddings.empty()) {
            output.emplace_back(std::move(cppSegment));
        }
    }

    return output;
}
} // namespace

@interface AHCEmbeddingSample ()
@property(nonatomic, readwrite) NSUUID *identifier;
@property(nonatomic, readwrite) NSInteger speakerIndex;
@property(nonatomic, readwrite) float weight;
@property(nonatomic, readwrite) NSData *vectorData;
@end

@implementation AHCEmbeddingSample

- (instancetype)initWithIdentifier:(NSUUID *)identifier
                      speakerIndex:(NSInteger)speakerIndex
                            weight:(float)weight
                        vectorData:(NSData *)vectorData {
    self = [super init];
    if (self == nil) {
        return nil;
    }

    _identifier = identifier;
    _speakerIndex = speakerIndex;
    _weight = weight;
    _vectorData = [vectorData copy];
    return self;
}

@end

@interface AHCEmbeddingSegmentInput ()
@property(nonatomic, readwrite) NSInteger speakerIndex;
@property(nonatomic, readwrite) NSArray<AHCEmbeddingSample *> *embeddings;
@end

@implementation AHCEmbeddingSegmentInput

- (instancetype)initWithSpeakerIndex:(NSInteger)speakerIndex
                          embeddings:(NSArray<AHCEmbeddingSample *> *)embeddings {
    self = [super init];
    if (self == nil) {
        return nil;
    }

    _speakerIndex = speakerIndex;
    _embeddings = [embeddings copy];
    return self;
}

@end

@interface AHCDendrogramNodeSnapshot ()
@property(nonatomic, readwrite) NSInteger index;
@property(nonatomic, readwrite) NSInteger matrixIndex;
@property(nonatomic, readwrite) NSInteger leftChild;
@property(nonatomic, readwrite) NSInteger rightChild;
@property(nonatomic, readwrite) NSInteger speakerIndex;
@property(nonatomic, readwrite) NSInteger count;
@property(nonatomic, readwrite) float weight;
@property(nonatomic, readwrite) float mergeDistance;
@property(nonatomic, readwrite) BOOL mustLink;
@end

@implementation AHCDendrogramNodeSnapshot

- (instancetype)initWithIndex:(NSInteger)index
                  matrixIndex:(NSInteger)matrixIndex
                    leftChild:(NSInteger)leftChild
                   rightChild:(NSInteger)rightChild
                  speakerIndex:(NSInteger)speakerIndex
                        count:(NSInteger)count
                       weight:(float)weight
                mergeDistance:(float)mergeDistance
                     mustLink:(BOOL)mustLink {
    self = [super init];
    if (self == nil) {
        return nil;
    }

    _index = index;
    _matrixIndex = matrixIndex;
    _leftChild = leftChild;
    _rightChild = rightChild;
    _speakerIndex = speakerIndex;
    _count = count;
    _weight = weight;
    _mergeDistance = mergeDistance;
    _mustLink = mustLink;
    return self;
}

@end

@interface AHCDendrogramSnapshot ()
@property(nonatomic, readwrite) NSInteger rootIndex;
@property(nonatomic, readwrite) NSInteger nodeCount;
@property(nonatomic, readwrite) NSInteger activeLeafCount;
@property(nonatomic, readwrite) NSArray<AHCDendrogramNodeSnapshot *> *nodes;
@end

@implementation AHCDendrogramSnapshot

- (instancetype)initWithRootIndex:(NSInteger)rootIndex
                        nodeCount:(NSInteger)nodeCount
                  activeLeafCount:(NSInteger)activeLeafCount
                            nodes:(NSArray<AHCDendrogramNodeSnapshot *> *)nodes {
    self = [super init];
    if (self == nil) {
        return nil;
    }

    _rootIndex = rootIndex;
    _nodeCount = nodeCount;
    _activeLeafCount = activeLeafCount;
    _nodes = [nodes copy];
    return self;
}

@end

@interface AHCSpeakerForestBridge () {
    std::unique_ptr<SpeakerForest> _forest;
    long _numRepresentatives;
    long _minEmbeddings;
    long _maxEmbeddings;
}
@end

@implementation AHCSpeakerForestBridge

- (instancetype)initWithNumRepresentatives:(NSInteger)numRepresentatives
                             minEmbeddings:(NSInteger)minEmbeddings
                             maxEmbeddings:(NSInteger)maxEmbeddings {
    self = [super init];
    if (self == nil) {
        return nil;
    }

    _numRepresentatives = static_cast<long>(std::max<NSInteger>(1, numRepresentatives));
    _minEmbeddings = static_cast<long>(std::max<NSInteger>(1, minEmbeddings));
    _maxEmbeddings = static_cast<long>(std::max<NSInteger>(1, maxEmbeddings));
    _forest = std::make_unique<SpeakerForest>(_numRepresentatives, _minEmbeddings, _maxEmbeddings);
    return self;
}

- (void)reset {
    _forest = std::make_unique<SpeakerForest>(_numRepresentatives, _minEmbeddings, _maxEmbeddings);
}

- (BOOL)streamWithFinalizedSegments:(NSArray<AHCEmbeddingSegmentInput *> *)finalized
                  tentativeSegments:(NSArray<AHCEmbeddingSegmentInput *> *)tentative {
    if (_forest == nullptr) {
        [self reset];
        return NO;
    }

    try {
        auto finalizedSegments = convertSegments(finalized);
        auto tentativeSegments = convertSegments(tentative);
        _forest->streamEmbeddingSegments(finalizedSegments, tentativeSegments);
        return YES;
    } catch (const std::bad_alloc &e) {
        NSLog(@"AHCSpeakerForestBridge bad_alloc during stream: %s", e.what());
        [self reset];
    } catch (const std::exception &e) {
        NSLog(@"AHCSpeakerForestBridge std::exception during stream: %s", e.what());
        [self reset];
    } catch (...) {
        NSLog(@"AHCSpeakerForestBridge unknown C++ exception during stream");
        [self reset];
    }
    return NO;
}

- (AHCDendrogramSnapshot *)dendrogramSnapshot {
    if (_forest == nullptr) {
        return [[AHCDendrogramSnapshot alloc] initWithRootIndex:-1 nodeCount:0 activeLeafCount:0 nodes:@[]];
    }

    try {
        auto snapshot = _forest->dendrogramSnapshot();
        NSMutableArray<AHCDendrogramNodeSnapshot *> *nodes = [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(std::max<long>(0, snapshot.nodeCount))];

        for (long i = 0; i < snapshot.nodeCount && i < static_cast<long>(snapshot.nodes.size()); ++i) {
            const auto &node = snapshot.nodes[static_cast<size_t>(i)];
            [nodes addObject:[[AHCDendrogramNodeSnapshot alloc]
                initWithIndex:static_cast<NSInteger>(i)
                  matrixIndex:static_cast<NSInteger>(node.matrixIndex)
                    leftChild:static_cast<NSInteger>(node.leftChild)
                   rightChild:static_cast<NSInteger>(node.rightChild)
                  speakerIndex:static_cast<NSInteger>(node.speakerIndex)
                        count:static_cast<NSInteger>(node.count)
                       weight:node.weight
                mergeDistance:node.mergeDistance
                     mustLink:node.mustLink]];
        }

        return [[AHCDendrogramSnapshot alloc]
            initWithRootIndex:static_cast<NSInteger>(snapshot.rootIndex)
                    nodeCount:static_cast<NSInteger>(snapshot.nodeCount)
              activeLeafCount:static_cast<NSInteger>(snapshot.activeLeafCount)
                        nodes:nodes];
    } catch (const std::exception &e) {
        NSLog(@"AHCSpeakerForestBridge snapshot failed: %s", e.what());
        [self reset];
        return [[AHCDendrogramSnapshot alloc] initWithRootIndex:-1 nodeCount:0 activeLeafCount:0 nodes:@[]];
    } catch (...) {
        NSLog(@"AHCSpeakerForestBridge snapshot failed with unknown C++ exception");
        [self reset];
        return [[AHCDendrogramSnapshot alloc] initWithRootIndex:-1 nodeCount:0 activeLeafCount:0 nodes:@[]];
    }
}

@end
