#import "AHCSpeakerForestBridge.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ClusteringConfig.hpp"
#include "SpeakerForest.hpp"


namespace {
    using LinkageType = WeightedAverageLinkage;
//    using LinkageType = CosineAverageLinkage;
//    using LinkageType = WeightedCosineAverageLinkage;
//    using LinkageType = WardLinkage;
    using ForestType = SpeakerForest<LinkageType>;
    using SegmentType = EmbeddingSegmentWrapper<LinkageType>;
    using MatrixType = EmbeddingDistanceMatrix<LinkageType>;
    using DendrogramType = Dendrogram<LinkageType>;

    constexpr bool kUsesWardLogDistance = std::is_same_v<LinkageType, WardLinkage>;
    constexpr float kWardLogFloor = 1e-12f;

    float transformedLinkageDistance(float mergeDistance) {
        if (!std::isfinite(mergeDistance)) {
            return 0.f;
        }

        if (kUsesWardLogDistance) {
            const float flooredDistance = std::max(mergeDistance, kWardLogFloor);
            const float loggedDistance = std::log(flooredDistance);
            return std::isfinite(loggedDistance) ? loggedDistance : 0.f;
        }

        return mergeDistance < 0.f ? 0.f : mergeDistance;
    }

    

    std::vector<SegmentType> convertSegments(
                                             NSArray<AHCEmbeddingSegmentInput *> *segments,
                                             std::unordered_map<UUIDWrapper, long> &speakerIndexBySegmentId,
                                             NSUInteger maxSegments
                                             ) {
        std::vector<SegmentType> output;
        if (segments.count == 0) {
            return output;
        }
        
        const NSUInteger startIndex = segments.count > maxSegments ? segments.count - maxSegments : 0;
        output.reserve(segments.count - startIndex);
        
        for (NSUInteger idx = startIndex; idx < segments.count; ++idx) {
            AHCEmbeddingSegmentInput *segment = [segments objectAtIndex:idx];
            if (segment.segmentIdentifier == nil || segment.embeddings.count == 0) {
                continue;
            }
            
            std::vector<const void *> swiftEmbeddingPointers;
            swiftEmbeddingPointers.reserve(segment.embeddings.count);
            
            for (AHCEmbeddingSample *sample in segment.embeddings) {
                if (sample.swiftEmbeddingPointer != nullptr) {
                    swiftEmbeddingPointers.push_back(sample.swiftEmbeddingPointer);
                }
            }
            
            if (swiftEmbeddingPointers.empty()) {
                continue;
            }
            
            uuid_t segmentId = {0};
            [segment.segmentIdentifier getUUIDBytes:segmentId];
            const auto segmentIdWrapper = UUIDWrapper(segmentId);
            speakerIndexBySegmentId[segmentIdWrapper] = static_cast<long>(segment.speakerIndex);
            
            output.emplace_back(
                                segmentId,
                                static_cast<long>(segment.speakerIndex),
                                const_cast<const void **>(swiftEmbeddingPointers.data()),
                                static_cast<long>(swiftEmbeddingPointers.size()),
                                nullptr,
                                0
                                );
        }
        
        return output;
    }

    void pruneSpeakerLookup(
                            const MatrixType &matrix,
                            std::unordered_map<UUIDWrapper, long> &speakerIndexBySegmentId
                            ) {
        std::unordered_set<UUIDWrapper> activeSegmentIds;
        activeSegmentIds.reserve(static_cast<size_t>(std::max<long>(0, matrix.embeddingCount())));
        
        for (long i = 0; i < matrix.size(); ++i) {
            const auto &embedding = matrix.embedding(i);
            if (embedding.hasVector()) {
                activeSegmentIds.insert(embedding.id());
            }
        }
        
        for (auto it = speakerIndexBySegmentId.begin(); it != speakerIndexBySegmentId.end();) {
            if (!activeSegmentIds.contains(it->first)) {
                it = speakerIndexBySegmentId.erase(it);
            } else {
                ++it;
            }
        }
    }

    std::vector<long> reachableNodeIds(const DendrogramType &dendrogram, long rootIndex, long maxNodeIndex) {
        if (rootIndex < 0 || maxNodeIndex <= 0) {
            return {};
        }
        
        std::unordered_set<long> visited;
        visited.reserve(static_cast<size_t>(std::max<long>(1, dendrogram.nodeCount())));
        
        std::vector<long> stack;
        stack.reserve(static_cast<size_t>(std::max<long>(1, dendrogram.nodeCount())));
        stack.push_back(rootIndex);
        
        auto nodes = dendrogram.nodes();
        if (!nodes) {
            return {};
        }
        
        while (!stack.empty()) {
            const long nodeId = stack.back();
            stack.pop_back();
            
            if (nodeId < 0 || nodeId >= maxNodeIndex || visited.contains(nodeId)) {
                continue;
            }
            
            visited.insert(nodeId);
            const auto &node = nodes[nodeId];
            
            if (node.leftChild >= 0) {
                stack.push_back(node.leftChild);
            }
            if (node.rightChild >= 0) {
                stack.push_back(node.rightChild);
            }
        }
        
        std::vector<long> orderedIds(visited.begin(), visited.end());
        std::sort(orderedIds.begin(), orderedIds.end());
        return orderedIds;
    }

    AHCDendrogramSnapshot *emptySnapshot(NSInteger activeLeafCount = 0) {
        return [[AHCDendrogramSnapshot alloc] initWithRootIndex:-1 nodeCount:0 activeLeafCount:activeLeafCount nodes:@[]];
    }
} // namespace

@interface AHCEmbeddingSample ()
@property(nonatomic, readwrite) const void *swiftEmbeddingPointer;
@end

@implementation AHCEmbeddingSample

- (instancetype)initWithSwiftEmbeddingPointer:(const void *)swiftEmbeddingPointer {
    self = [super init];
    if (self == nil) {
        return nil;
    }

    _swiftEmbeddingPointer = swiftEmbeddingPointer;
    return self;
}

@end

@interface AHCEmbeddingSegmentInput ()
@property(nonatomic, readwrite) NSUUID *segmentIdentifier;
@property(nonatomic, readwrite) NSInteger speakerIndex;
@property(nonatomic, readwrite) NSArray<AHCEmbeddingSample *> *embeddings;
@end

@implementation AHCEmbeddingSegmentInput

- (instancetype)initWithSegmentIdentifier:(NSUUID *)segmentIdentifier
                             speakerIndex:(NSInteger)speakerIndex
                               embeddings:(NSArray<AHCEmbeddingSample *> *)embeddings {
    self = [super init];
    if (self == nil) {
        return nil;
    }

    _segmentIdentifier = [segmentIdentifier copy];
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
@property(nonatomic, readwrite) BOOL exceedsLinkageThreshold;
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
                     mustLink:(BOOL)mustLink
      exceedsLinkageThreshold:(BOOL)exceedsLinkageThreshold {
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
    _exceedsLinkageThreshold = exceedsLinkageThreshold;
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
    std::shared_ptr<ClusteringConfig> _config;
    std::unique_ptr<ForestType> _forest;
    std::unordered_map<UUIDWrapper, long> _speakerIndexBySegmentId;
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

    _config = std::make_shared<ClusteringConfig>();
    _config->maxRepresentatives = static_cast<long>(std::max<NSInteger>(1, numRepresentatives));
    _config->minEmbeddings = static_cast<long>(std::max<NSInteger>(1, minEmbeddings));
    _config->maxEmbeddings = static_cast<long>(std::max<NSInteger>(1, maxEmbeddings));

    _forest = std::make_unique<ForestType>(_config);
    _speakerIndexBySegmentId.reserve(static_cast<size_t>(_config->maxEmbeddings * 2));
    return self;
}

- (void)reset {
    _forest = std::make_unique<ForestType>(_config);
    _speakerIndexBySegmentId.clear();
}

- (BOOL)streamWithFinalizedSegments:(NSArray<AHCEmbeddingSegmentInput *> *)finalized
                  tentativeSegments:(NSArray<AHCEmbeddingSegmentInput *> *)tentative {
    if (_forest == nullptr) {
        [self reset];
        return NO;
    }

    try {
        const auto finalizedCap = static_cast<NSUInteger>(std::max<long>(256, _config->maxEmbeddings * 2));
        const auto tentativeCap = static_cast<NSUInteger>(std::max<long>(128, _config->maxEmbeddings));

        auto finalizedSegments = convertSegments(finalized, _speakerIndexBySegmentId, finalizedCap);
        auto tentativeSegments = convertSegments(tentative, _speakerIndexBySegmentId, tentativeCap);
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
        return emptySnapshot();
    }
    
    try {
        const auto &matrix = _forest->distanceMatrix();
        const auto activeLeafCount = static_cast<NSInteger>(std::max<long>(0, matrix.embeddingCount()));
        const auto &dendrogram = _forest->currentDendrogram();
        const auto rootIndex = dendrogram.rootId();
        
        if (rootIndex < 0) {
            return emptySnapshot(activeLeafCount);
        }
        
        auto nodes = dendrogram.nodes();
        if (!nodes) {
            return emptySnapshot(activeLeafCount);
        }
        
        pruneSpeakerLookup(matrix, _speakerIndexBySegmentId);
        
        const auto maxNodeIndex = std::max<long>(
                                                 0,
                                                 std::max(dendrogram.nodeCount(), matrix.size())
                                                 );
        auto nodeIds = reachableNodeIds(dendrogram, rootIndex, maxNodeIndex);
        if (nodeIds.empty()) {
            return emptySnapshot(activeLeafCount);
        }
        
        NSMutableArray<AHCDendrogramNodeSnapshot *> *snapshotNodes =
        [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(nodeIds.size())];

        bool hasLinkageCutoff = false;
        float linkageCutoff = 0.f;
        if constexpr (kUsesWardLogDistance) {
            std::vector<float> internalMergeDistances;
            internalMergeDistances.reserve(nodeIds.size());
            for (long nodeId : nodeIds) {
                const auto &node = nodes[nodeId];
                if (node.leftChild < 0 || node.rightChild < 0 ||
                    node.leftChild >= maxNodeIndex || node.rightChild >= maxNodeIndex) {
                    continue;
                }

                const float mergeDistance = transformedLinkageDistance(node.mergeDistance);
                internalMergeDistances.push_back(mergeDistance);
            }

            if (!internalMergeDistances.empty()) {
                double sum = 0.0;
                for (float distance : internalMergeDistances) {
                    sum += static_cast<double>(distance);
                }
                const double mean = sum / static_cast<double>(internalMergeDistances.size());

                double variance = 0.0;
                for (float distance : internalMergeDistances) {
                    const double delta = static_cast<double>(distance) - mean;
                    variance += delta * delta;
                }
                variance /= static_cast<double>(internalMergeDistances.size());

                const double stddev = std::sqrt(variance);
                linkageCutoff = static_cast<float>(mean + 2.0 * stddev);
                hasLinkageCutoff = std::isfinite(linkageCutoff);
            }
        } else {
            linkageCutoff = _config ? _config->mergeThreshold : 0.f;
            hasLinkageCutoff = std::isfinite(linkageCutoff);
        }
        
        for (long nodeId : nodeIds) {
            const auto &node = nodes[nodeId];
            
            const bool isLeaf = node.leftChild < 0 || node.rightChild < 0;
            long speakerIndex = -1;
            if (isLeaf && node.matrixIndex >= 0 && node.matrixIndex < matrix.size()) {
                const auto &embedding = matrix.embedding(node.matrixIndex);
                if (embedding.hasVector()) {
                    const auto it = _speakerIndexBySegmentId.find(embedding.id());
                    if (it != _speakerIndexBySegmentId.end()) {
                        speakerIndex = it->second;
                    }
                }
            }
            
            const float mergeDistance = transformedLinkageDistance(node.mergeDistance);

            BOOL exceedsLinkageThreshold = NO;
            if (node.leftChild >= 0 && node.rightChild >= 0 &&
                node.leftChild < maxNodeIndex && node.rightChild < maxNodeIndex) {
                exceedsLinkageThreshold = hasLinkageCutoff && std::isfinite(mergeDistance) && mergeDistance > linkageCutoff;
            }
            
            [snapshotNodes addObject:[[AHCDendrogramNodeSnapshot alloc]
                                      initWithIndex:static_cast<NSInteger>(nodeId)
                                      matrixIndex:static_cast<NSInteger>(node.matrixIndex)
                                      leftChild:static_cast<NSInteger>(node.leftChild)
                                      rightChild:static_cast<NSInteger>(node.rightChild)
                                      speakerIndex:static_cast<NSInteger>(speakerIndex)
                                      count:static_cast<NSInteger>(node.count)
                                      weight:node.weight
                                      mergeDistance:mergeDistance
                                      mustLink:NO
                                      exceedsLinkageThreshold:exceedsLinkageThreshold]];
        }
        
        return [[AHCDendrogramSnapshot alloc]
                initWithRootIndex:static_cast<NSInteger>(rootIndex)
                nodeCount:static_cast<NSInteger>(nodeIds.size())
                activeLeafCount:activeLeafCount
                nodes:snapshotNodes];
    } catch (const std::exception &e) {
        NSLog(@"AHCSpeakerForestBridge snapshot failed: %s", e.what());
        [self reset];
        return emptySnapshot();
    } catch (...) {
        NSLog(@"AHCSpeakerForestBridge snapshot failed with unknown C++ exception");
        [self reset];
        return emptySnapshot();
    }
}

@end
