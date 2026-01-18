import Foundation
import Accelerate

/// Complete diarization timeline managing streaming predictions and segments
public class SortformerTimeline {
    private struct StreamingState {
        var starts: [Int]
        var isSpeaking: [Bool]
        var lastSegments: [(start: Int, end: Int)]
        
        init(numSpeakers: Int) {
            self.starts = Array(repeating: 0, count: numSpeakers)
            self.isSpeaking = Array(repeating: false, count: numSpeakers)
            self.lastSegments = Array(repeating: (0, 0), count: numSpeakers)
        }
    }
    
    /// Post-processing configuration
    public let config: SortformerPostProcessingConfig
    
    /// Embedding extraction configuration
    public let embeddingConfig: EmbeddingConfig

    /// Finalized frame-wise speaker predictions
    /// Shape: [numFrames, numSpeakers]
    public private(set) var framePredictions: [Float] = []

    /// Tentative predictions
    /// Shape: [numTentative, numSpeakers]
    public private(set) var tentativePredictions: [Float] = []
    
    /// Total number of finalized median-filtered frames
    public private(set) var nextFrame: Int = 0
    
    /// Number of finalized frames stored
    public var numFinalized: Int {
        framePredictions.count / config.numSpeakers
    }

    /// Number of tentative frames (including right context frames from chunk)
    public var numTentative: Int {
        tentativePredictions.count / config.numSpeakers
    }
    
    /// Finalized segments
    public private(set) var segments: [[SortformerSegment]] = []
    
    /// Tentative segments (may change as more predictions arrive)
    public private(set) var tentativeSegments: [[SortformerSegment]] = []
    
    /// Finalized single-speaker segments with embedding tracking
    public internal(set) var embeddingSegments: [EmbeddingSegment] = []
    
    /// Tentative single-speaker segments (no embeddings yet)
    public internal(set) var tentativeEmbeddingSegments: [EmbeddingSegment] = []
    
    /// Pending embedding extraction requests
    public internal(set) var pendingEmbeddingRequests: [EmbeddingExtractionRequest] = []
    
    /// Orphaned embeddings that couldn't be assigned to any segment (temporary buffer)
    /// These will be tried again on the next replaceEmbeddingSegments call
    private var orphanedEmbeddings: [(embedding: TitaNetEmbedding, speakerIndex: Int, age: Int)] = []
    
    /// Maximum age (in update calls) before orphaned embeddings are discarded
    private let maxOrphanAge: Int = 5

    /// Get total duration of finalized predictions in seconds
    public var duration: Float {
        Float(nextFrame) * config.frameDurationSeconds
    }

    /// Get total duration including tentative predictions in seconds
    public var tentativeDuration: Float {
        Float(nextFrame + numTentative) * config.frameDurationSeconds
    }
    
    /// Number of finalized frames in which segments can still be updated
    private var tentativePadding: Int {
        config.onsetPadFrames + config.offsetPadFrames + config.minFramesOff
    }

    /// Active segments being built (one per speaker, nil if speaker not active)
    private var state: StreamingState
    
    /// Filter
    private let filter: SortformerFilter

    /// Logger for warnings
    static let logger = AppLogger(category: "SortformerTimeline")

    /// Initialize with configuration for streaming usage
    /// - Parameters:
    ///   - config: Sortformer post-processing configuration
    ///   - embeddingConfig: Embedding extraction configuration
    public init(
        config: SortformerPostProcessingConfig = .default(for: .default),
        embeddingConfig: EmbeddingConfig = EmbeddingConfig()
    ) {
        self.config = config
        self.embeddingConfig = embeddingConfig
        self.state = StreamingState(numSpeakers: config.numSpeakers)
        self.segments = Array(repeating: [], count: config.numSpeakers)
        self.tentativeSegments = Array(repeating: [], count: config.numSpeakers)
        let weights = [Float](repeating: 0.5, count: config.numFilteredFrames)
        self.filter = SortformerFilter(weights: weights, numSpeakers: config.numSpeakers)
    }

    /// Initialize with existing probabilities (e.g. from batch processing or restored state)
    /// - Parameters:
    ///   - allPredictions: Raw speaker probabilities (flattened)
    ///   - config: Configuration object
    ///   - isComplete: If true, treats the provided probabilities as the complete timeline and finalizes everything immediately.
    ///                 If false, treats them as initial raw predictions that may be extended.
    public convenience init(
        allPredictions: [Float],
        config: SortformerPostProcessingConfig = .default(for: .default),
        isComplete: Bool = true
    ) {
        self.init(config: config)
        let numFrames = allPredictions.count / config.numSpeakers
        var newSegments: [SortformerSegment] = []
        extractSegments(
            predictions: allPredictions,
            numFrames: numFrames,
            startFrame: 0,
            isFinalized: true,
            addTrailingTentative: true,
            state: &state,
            finalizedResults: &segments,
            tentativeResults: &tentativeSegments,
            flatResults: &newSegments
        )
        self.framePredictions = allPredictions
        self.nextFrame = numFrames
        trimPredictions()
        
        (self.embeddingSegments, self.tentativeEmbeddingSegments) = self.extractDisjointSegments(
            segments: newSegments, firstTentativeFrame: .max)

        if isComplete {
            // Finalize everything immediately
            finalize()
        }
    }

    /// Add a new chunk of predictions from the diarizer
    public func addChunk(_ chunk: SortformerChunkResult) -> SortformerTimelineDifference {
        // Apply EMA filter to existing predictions using FIFO as reference
        // This smooths the tail of framePredictions before appending new data
        let updatedFrameCount = min(numFinalized, chunk.fifoFrameCount, filter.windowSize)
        let updatedPredCount = updatedFrameCount * config.numSpeakers
        
        #if DEBUG
        print("📊 [addChunk] START: nextFrame=\(nextFrame), updatedFrameCount=\(updatedFrameCount), numFinalized=\(numFinalized)")
        print("   fifoFrameCount=\(chunk.fifoFrameCount), filterWindowSize=\(filter.windowSize)")
        print("   Existing segments before filter:")
        for speakerIndex in 0..<config.numSpeakers {
            if !segments[speakerIndex].isEmpty {
                print("     Speaker \(speakerIndex): \(segments[speakerIndex].map { "[\($0.startFrame)-\($0.endFrame)]" }.joined(separator: ", "))")
            }
        }
        #endif
        
        if !framePredictions.isEmpty && !chunk.fifoPredictions.isEmpty {
            framePredictions.withUnsafeMutableBufferPointer { predPtr in
                chunk.fifoPredictions.withUnsafeBufferPointer { fifoPtr in
                    let predBase = predPtr.baseAddress!
                        .advanced(by: predPtr.count - updatedPredCount)
                    let fifoBase = fifoPtr.baseAddress!
                        .advanced(by: fifoPtr.count - updatedPredCount)
                    do {
                        try filter.update(predBase, with: fifoBase, result: predBase, count: updatedPredCount)
                    } catch {
                        fatalError(error.localizedDescription)
                    }
                }
            }
        }

        // Get streaming state BEFORE clearing tentative segments
        // This allows us to correctly reconstruct state from tentative segments
        let firstUpdatedFrame = nextFrame - updatedFrameCount
        
        #if DEBUG
        print("   firstUpdatedFrame=\(firstUpdatedFrame) (update region: [\(firstUpdatedFrame)-\(nextFrame)])")
        #endif
        
        var oldState = getStreamingState(atFrame: firstUpdatedFrame)
        
        // Now safe to clear tentative segments
        for i in 0..<config.numSpeakers {
            tentativeSegments[i].removeAll(keepingCapacity: true)
        }

        // Append new predictions BEFORE extracting segments
        // This ensures segments that span the boundary are properly extracted
        framePredictions.append(contentsOf: chunk.speakerPredictions)
        tentativePredictions = chunk.tentativePredictions
        
        // Calculate the total finalized frame range to process
        let newNextFrame = nextFrame + chunk.frameCount
        let totalUpdatedFrameCount = newNextFrame - firstUpdatedFrame
        
        // Get the predictions for the full updated range [firstUpdatedFrame, newNextFrame)
        let fullUpdatedPreds = Array(framePredictions.suffix(totalUpdatedFrameCount * config.numSpeakers))
        
        // Extract segments from the full updated range
        var updatedSegments: [SortformerSegment] = []
        var updatedFinalized: [[SortformerSegment]] = Array(repeating: [], count: config.numSpeakers)
        var updatedTentative: [[SortformerSegment]] = Array(repeating: [], count: config.numSpeakers)
        
        extractSegments(
            predictions: fullUpdatedPreds,
            numFrames: totalUpdatedFrameCount,
            startFrame: firstUpdatedFrame,
            isFinalized: true,
            addTrailingTentative: false,
            state: &oldState,
            finalizedResults: &updatedFinalized,
            tentativeResults: &updatedTentative,
            flatResults: &updatedSegments
        )
        
        #if DEBUG
        print("   Segments extracted from fullUpdatedPreds [\(firstUpdatedFrame)-\(newNextFrame)]:")
        for speakerIndex in 0..<config.numSpeakers {
            if !updatedFinalized[speakerIndex].isEmpty {
                print("     Speaker \(speakerIndex): \(updatedFinalized[speakerIndex].map { "[\($0.startFrame)-\($0.endFrame)]" }.joined(separator: ", "))")
            }
        }
        #endif
        
        // Collect old segments that overlap with the updated region for diff computation
        // O(k) tail search: segments are sorted by startFrame, so we find the first overlapping one
        // and take the suffix. A segment overlaps [firstUpdatedFrame, newNextFrame) if:
        // seg.startFrame < newNextFrame AND seg.endFrame > firstUpdatedFrame
        var oldSegmentsInRegion: [[SortformerSegment]] = Array(repeating: [], count: config.numSpeakers)
        var firstOverlapIndices: [Int] = Array(repeating: 0, count: config.numSpeakers)
        
        for speakerIndex in 0..<config.numSpeakers {
            let speakerSegs = segments[speakerIndex]
            // Find the first segment that could overlap: one whose endFrame > firstUpdatedFrame
            // Since segments are sorted by startFrame, search backwards from the end
            if let lastNonOverlapping = speakerSegs.lastIndex(where: { $0.endFrame <= firstUpdatedFrame }) {
                // Segments from lastNonOverlapping + 1 onwards may overlap
                let startIdx = lastNonOverlapping + 1
                firstOverlapIndices[speakerIndex] = startIdx
                oldSegmentsInRegion[speakerIndex] = Array(speakerSegs.suffix(from: startIdx))
            } else if !speakerSegs.isEmpty {
                // All segments might overlap (or list is empty)
                firstOverlapIndices[speakerIndex] = 0
                oldSegmentsInRegion[speakerIndex] = speakerSegs
            }
        }
        
        #if DEBUG
        print("   Old segments in region [\(firstUpdatedFrame)-\(newNextFrame)]:")
        for speakerIndex in 0..<config.numSpeakers {
            if !oldSegmentsInRegion[speakerIndex].isEmpty {
                print("     Speaker \(speakerIndex): \(oldSegmentsInRegion[speakerIndex].map { "[\($0.startFrame)-\($0.endFrame)]" }.joined(separator: ", "))")
            }
        }
        #endif
        
        // Compute the difference between old and new segments in the updated region
        // The diff identifies which segments to delete (changed) and which to insert (new)
        // Overlapping segments are considered "matched" and neither deleted nor inserted
        let diff = SortformerTimelineDifference(
            oldSegments: oldSegmentsInRegion,
            newSegments: updatedFinalized
        )
        
        #if DEBUG
        if !diff.isEmpty {
            print("   Diff: \(diff.insertions.count) insertions, \(diff.deletions.count) deletions, \(diff.updates.count) updates")
            for ins in diff.insertions {
                print("     INSERT: speaker \(ins.speakerIndex) [\(ins.startFrame)-\(ins.endFrame)]")
            }
            for del in diff.deletions {
                print("     DELETE: speaker \(del.speakerIndex) [\(del.startFrame)-\(del.endFrame)]")
            }
            for upd in diff.updates {
                print("     UPDATE: speaker \(upd.speakerIndex) [\(upd.oldStartFrame)-\(upd.oldEndFrame)] -> [\(upd.newStartFrame)-\(upd.newEndFrame)]")
            }
        }
        #endif

        // O(k) tail replacement: instead of checking each segment, we:
        // 1. Remove all segments from firstOverlapIndex onwards (these are in the update region)
        // 2. Append all new segments from updatedFinalized (already sorted)
        // This avoids O(n) searches for existence checks and insertion points
        for speakerIndex in 0..<config.numSpeakers {
            let firstOverlapIdx = firstOverlapIndices[speakerIndex]
            // Remove tail (segments in update region)
            if firstOverlapIdx < segments[speakerIndex].count {
                segments[speakerIndex].removeSubrange(firstOverlapIdx...)
            }
            // Append new segments (they're already sorted by startFrame from extractSegments)
            segments[speakerIndex].append(contentsOf: updatedFinalized[speakerIndex])
        }
        
        // Update nextFrame
        nextFrame = newNextFrame
        
        // Extract tentative segments from tentative predictions
        extractSegments(
            predictions: chunk.tentativePredictions,
            numFrames: chunk.tentativeFrameCount,
            startFrame: nextFrame,
            isFinalized: false,
            addTrailingTentative: true,
            state: &oldState,
            finalizedResults: &segments,
            tentativeResults: &tentativeSegments,
            flatResults: &updatedSegments
        )
        
        // Trim predictions
        trimPredictions()
        
        // Update disjoint segments - get ALL segments for complete recomputation
        // We pass 0 to get all segments, then fully replace embeddingSegments
        let allSegments = getSegmentsAfter(frame: 0)
        let (newDisjoint, newTentativeDisjoint) = extractDisjointSegments(
            segments: allSegments, firstTentativeFrame: nextFrame)
        
        // Fully replace embedding segments with fresh computation
        // Transfer embeddings from old segments to new ones by matching frame ranges
        replaceEmbeddingSegments(with: newDisjoint)
        tentativeEmbeddingSegments = newTentativeDisjoint
        
        // Generate periodic embedding requests for tentative segments
        // First extraction when segment reaches minEmbeddingFrames, then every maxEmbeddingFrames
        for tentativeSeg in tentativeEmbeddingSegments {
            let segLength = tentativeSeg.length
            let minLen = embeddingConfig.minEmbeddingFrames
            let maxLen = embeddingConfig.maxEmbeddingFrames
            
            // Skip if too short
            guard segLength >= minLen else { continue }
            
            // Calculate where the next embedding should be extracted
            let existingCoverage = tentativeSeg.embeddings.reduce(0) { $0 + $1.length }
            let embeddingCount = tentativeSeg.embeddings.count
            
            // First embedding: when segment reaches minLen
            // Subsequent: every maxLen frames of new content
            let shouldExtract: Bool
            if embeddingCount == 0 {
                // No embeddings yet - extract if we have enough length
                shouldExtract = true
            } else {
                // Already has embeddings - extract if we've grown by maxLen since last
                let uncoveredLength = segLength - existingCoverage
                shouldExtract = uncoveredLength >= maxLen
            }
            
            if shouldExtract {
                // Only request for the new portion (from end of existing coverage to current end)
                let lastCoveredFrame = tentativeSeg.embeddings.map { $0.endFrame }.max() ?? tentativeSeg.startFrame
                let requestStart = max(lastCoveredFrame, tentativeSeg.endFrame - maxLen)
                let requestEnd = tentativeSeg.endFrame
                
                if requestEnd - requestStart >= minLen {
                    let request = EmbeddingExtractionRequest(
                        segment: tentativeSeg,
                        startFrame: requestStart,
                        endFrame: requestEnd
                    )
                    pendingEmbeddingRequests.append(request)
                    Self.logger.debug("Generated tentative embedding request [\(requestStart)-\(requestEnd)] for speaker \(tentativeSeg.speakerIndex)")
                }
            }
        }
        
        // DEBUG: Validate SortformerSegments every frame
        #if DEBUG
        validateSortformerSegments(context: "addChunk, nextFrame=\(nextFrame)")
        #endif
        
        return diff
    }
    
    /// Validate SortformerSegments by recomputing from scratch and comparing
    /// This is O(n) - only use for debugging, not in production
    private func validateSortformerSegments(context: String) {
        // Recompute segments from scratch using all frame predictions
        var freshState = StreamingState(numSpeakers: config.numSpeakers)
        var freshSegments: [[SortformerSegment]] = Array(repeating: [], count: config.numSpeakers)
        var freshTentative: [[SortformerSegment]] = Array(repeating: [], count: config.numSpeakers)
        var flatResults: [SortformerSegment] = []
        
        // Extract from finalized predictions
        extractSegments(
            predictions: framePredictions,
            numFrames: numFinalized,
            startFrame: 0,
            isFinalized: true,
            addTrailingTentative: false,
            state: &freshState,
            finalizedResults: &freshSegments,
            tentativeResults: &freshTentative,
            flatResults: &flatResults
        )
        
        // Compare against current segments
        for speakerIndex in 0..<config.numSpeakers {
            let current = segments[speakerIndex]
            let fresh = freshSegments[speakerIndex]
            
            if current.count != fresh.count {
                print("🔴 [\(context)] SEGMENT COUNT MISMATCH for speaker \(speakerIndex)")
                print("   Current count: \(current.count), Fresh count: \(fresh.count)")
                print("   Current segments:")
                for (i, seg) in current.enumerated() {
                    print("     [\(i)] start=\(seg.startFrame), end=\(seg.endFrame), finalized=\(seg.isFinalized)")
                }
                print("   Fresh segments:")
                for (i, seg) in fresh.enumerated() {
                    print("     [\(i)] start=\(seg.startFrame), end=\(seg.endFrame), finalized=\(seg.isFinalized)")
                }
                Self.logger.error("[\(context)] Segment count mismatch for speaker \(speakerIndex): current=\(current.count) fresh=\(fresh.count)")
                assertionFailure("Segment count mismatch for speaker \(speakerIndex)")
                continue
            }
            
            for i in 0..<current.count {
                let c = current[i]
                let f = fresh[i]
                if c.startFrame != f.startFrame || c.endFrame != f.endFrame {
                    print("🔴 [\(context)] SEGMENT BOUNDS MISMATCH for speaker \(speakerIndex) segment[\(i)]")
                    print("   Current: start=\(c.startFrame), end=\(c.endFrame), finalized=\(c.isFinalized)")
                    print("   Fresh:   start=\(f.startFrame), end=\(f.endFrame), finalized=\(f.isFinalized)")
                    
                    // Print prediction values around the boundary to understand threshold crossings
                    let minStart = max(0, min(c.startFrame, f.startFrame) - 3)
                    let maxStart = min(numFinalized, max(c.startFrame, f.startFrame) + 3)
                    print("   Predictions around start boundary (onset threshold=\(config.onsetThreshold)):")
                    for frame in minStart..<maxStart {
                        let predIndex = frame * config.numSpeakers + speakerIndex
                        if predIndex < framePredictions.count {
                            let pred = framePredictions[predIndex]
                            let currentMarker = frame == c.startFrame + config.onsetPadFrames ? " ← current onset" : ""
                            let freshMarker = frame == f.startFrame + config.onsetPadFrames ? " ← fresh onset" : ""
                            let thresholdMarker = pred > config.onsetThreshold ? " ✓" : ""
                            print("     frame \(frame): \(String(format: "%.4f", pred))\(thresholdMarker)\(currentMarker)\(freshMarker)")
                        }
                    }
                    
                    print("   All current segments for speaker \(speakerIndex):")
                    for (j, seg) in current.enumerated() {
                        let marker = j == i ? " ⬅️" : ""
                        print("     [\(j)] start=\(seg.startFrame), end=\(seg.endFrame)\(marker)")
                    }
                    print("   All fresh segments for speaker \(speakerIndex):")
                    for (j, seg) in fresh.enumerated() {
                        let marker = j == i ? " ⬅️" : ""
                        print("     [\(j)] start=\(seg.startFrame), end=\(seg.endFrame)\(marker)")
                    }
                    Self.logger.error("[\(context)] Segment bounds mismatch for speaker \(speakerIndex) seg[\(i)]: current=[\(c.startFrame)-\(c.endFrame)] fresh=[\(f.startFrame)-\(f.endFrame)]")
                    // Note: This can happen when onset occurs just before the update window.
                    // The incremental update misses it, but full re-extraction catches it.
                    // This is a known limitation of FIFO-based incremental updates.
                }
            }
        }
        
        // Also recompute embedding segments
        let allSegments = freshSegments.flatMap { $0 }
        let (freshEmbedding, _) = extractDisjointSegments(segments: allSegments, firstTentativeFrame: nextFrame)
        
        if embeddingSegments.count != freshEmbedding.count {
            // This count difference is expected - the fresh count is computed from all segments
            // but incremental updates may have different segment counts due to merge/split operations
            // The validation below will catch actual problems (overlaps, duplicates)
        }
        
        // Check for overlapping embedding segments (this should never happen)
        for i in 0..<embeddingSegments.count {
            for j in (i+1)..<embeddingSegments.count {
                let seg1 = embeddingSegments[i]
                let seg2 = embeddingSegments[j]
                if seg1.speakerIndex == seg2.speakerIndex && seg1.frames.overlaps(seg2.frames) {
                    print("🔴 [\(context)] OVERLAPPING EMBEDDING SEGMENTS DETECTED")
                    print("   Speaker: \(seg1.speakerIndex)")
                    print("   Segment 1 [\(i)]: start=\(seg1.startFrame), end=\(seg1.endFrame)")
                    print("   Segment 2 [\(j)]: start=\(seg2.startFrame), end=\(seg2.endFrame)")
                    print("   Overlap range: \(max(seg1.startFrame, seg2.startFrame)) - \(min(seg1.endFrame, seg2.endFrame))")
                    Self.logger.error("[\(context)] OVERLAPPING embedding segments: speaker=\(seg1.speakerIndex) seg1=[\(seg1.startFrame)-\(seg1.endFrame)] seg2=[\(seg2.startFrame)-\(seg2.endFrame)]")
                    assertionFailure("Overlapping embedding segments")
                }
            }
        }
    }
    
    /// Get segments that overlap with [fromFrame, ∞)
    /// O(k) where k = number of segments in the region (typically constant in streaming)
    private func getSegmentsAfter(frame: Int) -> [SortformerSegment] {
        var result: [SortformerSegment] = []
        
        // We need segments that overlap with [frame, ∞), i.e., endFrame > frame
        for speakerSegs in segments {
            // Find the first segment that doesn't overlap: endFrame <= frame
            // Everything after that overlaps
            if let lastNonOverlapping = speakerSegs.lastIndex(where: { $0.endFrame <= frame }) {
                // Segments from lastNonOverlapping + 1 onwards overlap
                result.append(contentsOf: speakerSegs.suffix(from: lastNonOverlapping + 1))
            } else {
                // All segments overlap (or array is empty)
                result.append(contentsOf: speakerSegs)
            }
        }
        
        // Add all tentative segments (they're always at the end)
        for speakerSegs in tentativeSegments {
            result.append(contentsOf: speakerSegs)
        }
        
        return result
    }
    
    // TODO: Determine what is causing it to sometimes dupe embedding segments
    /// Update embedding segments from newly extracted disjoint segments
    /// O(k) where k = number of segments in the update region
    /// Preserves segments before updatedFromFrame unchanged
    private func updateEmbeddingSegmentsFromNewDisjoint(_ newSegments: [EmbeddingSegment], updatedFromFrame: Int) {
        // Find the split point: segments that END before the update boundary stay unchanged
        // Segments that extend into the update region (endFrame > updatedFromFrame) need processing
        let unchangedEndIndex = embeddingSegments.lastIndex { $0.endFrame <= updatedFromFrame }.map { $0 + 1 } ?? 0
        
        // Keep segments that are completely before the update region
        let unchangedSegments = Array(embeddingSegments.prefix(unchangedEndIndex))
        
        // Get old segments that overlap with the update region (endFrame > updatedFromFrame)
        let oldSegmentsInRegion = Array(embeddingSegments.suffix(from: unchangedEndIndex))
        
        // Build new segments for the update region
        var newResult: [EmbeddingSegment] = []
        
        for newSeg in newSegments {
            // Find overlapping old segments in the region (same speaker)
            let overlappingOld = oldSegmentsInRegion.filter { old in
                old.speakerIndex == newSeg.speakerIndex &&
                old.frames.overlaps(newSeg.frames)
            }
            
            // Check for exact match
            if let exactMatch = overlappingOld.first(where: { old in
                old.startFrame == newSeg.startFrame && old.endFrame == newSeg.endFrame
            }) {
                newResult.append(exactMatch)
                continue
            }
            
            // TODO: Optimize this. Embeddings should be added all at once
            // Transfer embeddings from overlapping old segments
            
            // OUTLINE:
            // 1. Collect all segments that overlap with this one.
            // 2. Feed it all the segments
            // 3. Discard segments that don't overlap
            for oldSeg in overlappingOld {
                for embedding in oldSeg.embeddings {
                    let overlapWithNew = embedding.overlapLength(with: newSeg)
                    
                    // Check if this embedding belongs to this new segment more than others
                    let otherNewSegments = newSegments.filter { $0.id != newSeg.id && $0.speakerIndex == newSeg.speakerIndex }
                    let bestOverlapWithOthers = otherNewSegments.map { embedding.overlapLength(with: $0) }.max() ?? 0
                    
                    if overlapWithNew > bestOverlapWithOthers && overlapWithNew > 0 {
                        let outsideFrames = embedding.length - overlapWithNew
                        if outsideFrames <= embeddingConfig.maxOutsideFrames {
                            newSeg.addEmbeddings([embedding])
                        }
                    }
                }
            }
            
            newResult.append(newSeg)
        }
        
        // Combine: unchanged + new (already sorted since both are sorted and non-overlapping)
        embeddingSegments = unchangedSegments + newResult
        
        // DEBUG: Check for overlapping or problematic segments every frame
        #if DEBUG
        validateEmbeddingSegments(context: "updateEmbeddingSegmentsFromNewDisjoint, updatedFromFrame=\(updatedFromFrame)")
        #endif
        
        // Generate embedding requests only for new finalized segments without embeddings
        for segment in newResult where segment.embeddings.isEmpty && segment.isFinalized {
            let requests = segment.getEmbeddingRequests(
                maxGapSize: embeddingConfig.maxEmbeddingGap,
                maxEmbeddingLength: embeddingConfig.maxEmbeddingFrames
            )
            pendingEmbeddingRequests.append(contentsOf: requests)
        }
    }
    
    /// Fully replace embedding segments with new ones, transferring embeddings from old to new
    /// This avoids the complexity of incremental updates and prevents sync issues
    private func replaceEmbeddingSegments(with newSegments: [EmbeddingSegment]) {
        // Track which embeddings from old segments were successfully transferred
        var transferredEmbeddingIds = Set<UUID>()
        
        // Source embeddings from both finalized and tentative segments
        // This ensures that when tentative segments finalize (and merge), their embeddings are preserved
        let allOldSegments = embeddingSegments + tentativeEmbeddingSegments
        
        // Transfer embeddings from old segments to new segments
        for newSeg in newSegments {
            // Find overlapping old segments with same speaker
            let overlappingOld = allOldSegments.filter { old in
                old.speakerIndex == newSeg.speakerIndex &&
                old.frames.overlaps(newSeg.frames)
            }
            
            // Check for exact match first (common case - segment unchanged)
            if let exactMatch = overlappingOld.first(where: { old in
                old.startFrame == newSeg.startFrame && old.endFrame == newSeg.endFrame
            }) {
                // Transfer all embeddings directly
                newSeg.addEmbeddings(exactMatch.embeddings)
                for emb in exactMatch.embeddings {
                    transferredEmbeddingIds.insert(emb.id)
                }
                continue
            }
            
            // Transfer embeddings from overlapping segments
            for oldSeg in overlappingOld {
                for embedding in oldSeg.embeddings {
                    let overlapWithNew = embedding.overlapLength(with: newSeg)
                    
                    // Check if this embedding belongs to this new segment more than others
                    let otherNewSegments = newSegments.filter { $0.id != newSeg.id && $0.speakerIndex == newSeg.speakerIndex }
                    let bestOverlapWithOthers = otherNewSegments.map { embedding.overlapLength(with: $0) }.max() ?? 0
                    
                    if overlapWithNew > bestOverlapWithOthers && overlapWithNew > 0 {
                        let outsideFrames = embedding.length - overlapWithNew
                        if outsideFrames <= embeddingConfig.maxOutsideFrames {
                            newSeg.addEmbeddings([embedding])
                            transferredEmbeddingIds.insert(embedding.id)
                        }
                    }
                }
            }
            
            // Try to reassign orphaned embeddings from previous updates
            for (idx, orphan) in orphanedEmbeddings.enumerated().reversed() {
                guard orphan.speakerIndex == newSeg.speakerIndex else { continue }
                
                let overlap = orphan.embedding.overlapLength(with: newSeg)
                if overlap > 0 {
                    let outsideFrames = orphan.embedding.length - overlap
                    if outsideFrames <= embeddingConfig.maxOutsideFrames {
                        newSeg.addEmbeddings([orphan.embedding])
                        orphanedEmbeddings.remove(at: idx)
                    }
                }
            }
        }
        
        // Collect orphaned embeddings (from old segments, not transferred)
        for oldSeg in allOldSegments {
            for embedding in oldSeg.embeddings {
                if !transferredEmbeddingIds.contains(embedding.id) {
                    // This embedding wasn't transferred - add to orphan buffer
                    orphanedEmbeddings.append((embedding: embedding, speakerIndex: oldSeg.speakerIndex, age: 0))
                }
            }
        }
        
        // Age and prune orphaned embeddings
        orphanedEmbeddings = orphanedEmbeddings.compactMap { orphan in
            let newAge = orphan.age + 1
            if newAge > maxOrphanAge {
                return nil // Too old, discard
            }
            return (orphan.embedding, orphan.speakerIndex, newAge)
        }
        
        // Replace with new segments
        embeddingSegments = newSegments
        
        // Clear stale pending requests (segments may have changed)
        pendingEmbeddingRequests.removeAll()
        
        // Generate embedding requests for segments without embeddings
        // Only for segments long enough to extract embeddings from
        var segmentsNeedingEmbeddings = 0
        for segment in newSegments where segment.embeddings.isEmpty && segment.isFinalized && segment.length >= embeddingConfig.minEmbeddingFrames {
            segmentsNeedingEmbeddings += 1
            let requests = segment.getEmbeddingRequests(
                maxGapSize: embeddingConfig.maxEmbeddingGap,
                maxEmbeddingLength: embeddingConfig.maxEmbeddingFrames,
                minEmbeddingLength: embeddingConfig.minEmbeddingFrames
            )
            pendingEmbeddingRequests.append(contentsOf: requests)
        }
        
        #if DEBUG
        validateEmbeddingSegments(context: "replaceEmbeddingSegments")
        #endif
    }

    /// Validate embedding segments for debugging - checks for overlaps and small segments
    private func validateEmbeddingSegments(context: String) {
        for i in 0..<embeddingSegments.count {
            let seg = embeddingSegments[i]
            
            // Check for very small segments
            if seg.length < 2 {
                Self.logger.warning("[\(context)] Very small segment: speaker=\(seg.speakerIndex) frames=[\(seg.startFrame)-\(seg.endFrame)] length=\(seg.length)")
            }
            
            // Check for overlaps with other segments of the same speaker
            for j in (i+1)..<embeddingSegments.count {
                let other = embeddingSegments[j]
                if seg.speakerIndex == other.speakerIndex && seg.frames.overlaps(other.frames) {
                    print("🔴 [\(context)] OVERLAPPING EMBEDDING SEGMENTS")
                    print("   Speaker: \(seg.speakerIndex)")
                    print("   Segment[\(i)]: start=\(seg.startFrame), end=\(seg.endFrame), embeddings=\(seg.embeddings.count)")
                    print("   Segment[\(j)]: start=\(other.startFrame), end=\(other.endFrame), embeddings=\(other.embeddings.count)")
                    print("   All embedding segments for speaker \(seg.speakerIndex):")
                    for (k, s) in embeddingSegments.enumerated() where s.speakerIndex == seg.speakerIndex {
                        let marker = (k == i || k == j) ? " ⬅️" : ""
                        print("     [\(k)] start=\(s.startFrame), end=\(s.endFrame)\(marker)")
                    }
                    Self.logger.error("[\(context)] OVERLAPPING SEGMENTS: speaker=\(seg.speakerIndex) seg1=[\(seg.startFrame)-\(seg.endFrame)] seg2=[\(other.startFrame)-\(other.endFrame)]")
                    assertionFailure("Overlapping embedding segments detected for speaker \(seg.speakerIndex): [\(seg.startFrame)-\(seg.endFrame)] overlaps [\(other.startFrame)-\(other.endFrame)]")
                }
            }
            
            // Check for duplicate IDs
            for j in (i+1)..<embeddingSegments.count {
                if seg.id == embeddingSegments[j].id {
                    print("🔴 [\(context)] DUPLICATE EMBEDDING SEGMENT ID")
                    print("   ID: \(seg.id)")
                    print("   Segment[\(i)]: speaker=\(seg.speakerIndex), start=\(seg.startFrame), end=\(seg.endFrame)")
                    print("   Segment[\(j)]: speaker=\(embeddingSegments[j].speakerIndex), start=\(embeddingSegments[j].startFrame), end=\(embeddingSegments[j].endFrame)")
                    Self.logger.error("[\(context)] DUPLICATE ID: \(seg.id) at indices \(i) and \(j)")
                    assertionFailure("Duplicate embedding segment ID: \(seg.id)")
                }
            }
        }
    }

    /// Helper to update segments from predictions
    /// - Parameters:
    ///   - predictions: Frame-level predictions shaped as [numFrames, numSpk] flattened
    ///   - numFrames: Number of frames
    ///   - isFinalized: Whether the predictions are finalized
    ///   - addTrailingTentative: Whether add tentative segments with no close.
    /// - Returns: New segments
    private func extractSegments(
        predictions: [Float],
        numFrames: Int,
        startFrame: Int,
        isFinalized: Bool,
        addTrailingTentative: Bool,
        state: inout StreamingState,
        finalizedResults: inout [[SortformerSegment]],
        tentativeResults: inout [[SortformerSegment]],
        flatResults: inout [SortformerSegment],
    ) {
        guard numFrames > 0 else {
            return
        }
        
        let numSpeakers = config.numSpeakers
        let onset = config.onsetThreshold
        let offset = config.offsetThreshold
        let padOnset = config.onsetPadFrames
        let padOffset = config.offsetPadFrames
        let minFramesOn = config.minFramesOn
        let minFramesOff = config.minFramesOff

        // Segments ending after this frame should be tentative because:
        // 1. They might be extended by future predictions
        // 2. The gap-closer (minFramesOff) could merge them with future segments
        // We need buffer for: onset padding + offset padding + gap closer threshold
        let tentativeStartFrame: Int
        if isFinalized {
            tentativeStartFrame = (startFrame + numFrames) - tentativePadding
        } else {
            tentativeStartFrame = 0
        }
        
        for speakerIndex in 0..<numSpeakers {
            var start = state.starts[speakerIndex]
            var speaking = state.isSpeaking[speakerIndex]
            var lastSegment = state.lastSegments[speakerIndex]
            var wasLastSegmentFinal = isFinalized

            for i in 0..<numFrames {
                let index = speakerIndex + i * numSpeakers

                if speaking {
                    if predictions[index] >= offset {
                        continue
                    }

                    // Speaking -> not speaking
                    speaking = false
                    let end = startFrame + i + padOffset

                    // Ensure segment is long enough
                    guard end - start > minFramesOn else {
                        continue
                    }

                    // Segment is only finalized if it ends BEFORE the tentative boundary
                    // This ensures gap-closer can still merge it with future segments
                    wasLastSegmentFinal = isFinalized && (end < tentativeStartFrame)

                    let newSegment = SortformerSegment(
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: wasLastSegmentFinal,
                        frameDurationSeconds: config.frameDurationSeconds
                    )

                    flatResults.append(newSegment)
                    if wasLastSegmentFinal {
                        finalizedResults[speakerIndex].append(newSegment)
                    } else {
                        tentativeResults[speakerIndex].append(newSegment)
                    }
                    lastSegment = (start, end)

                } else if predictions[index] > onset {
                    // Not speaking -> speaking
                    start = max(0, startFrame + i - padOnset)
                    speaking = true

                    if start - lastSegment.end <= minFramesOff {
                        // Merge with last segment to avoid overlap
                        start = lastSegment.start

                        if wasLastSegmentFinal {
                            _ = finalizedResults[speakerIndex].popLast()
                        } else {
                            _ = tentativeResults[speakerIndex].popLast()
                        }
                        _ = flatResults.popLast()
                    }
                }
            }

            if isFinalized {
                state.isSpeaking[speakerIndex] = speaking
                state.starts[speakerIndex] = start
                state.lastSegments[speakerIndex] = lastSegment
            }

            // Add still-speaking segment as tentative when requested
            // This is skipped during finalized processing in addChunk (tentative will be processed next)
            // But enabled for batch init and tentative processing
            if addTrailingTentative {
                let end = startFrame + numFrames + padOffset
                if speaking && (end > start) {
                    let newSegment = SortformerSegment(
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: false,
                        frameDurationSeconds: config.frameDurationSeconds
                    )
                    tentativeResults[speakerIndex].append(newSegment)
                    flatResults.append(newSegment)
                }
            }
        }
    }
    
    /// Note: call this AFTER `self.numFrames` has been updated
    private func extractDisjointSegments(
        segments: [SortformerSegment],
        firstTentativeFrame: Int? = nil
    ) -> (finalized: [EmbeddingSegment], tentative: [EmbeddingSegment]) {
        guard segments.count > 0 else {
            return ([], [])
        }
        
        // Get ordered boundary frames
        var boundaryFrames: [(frame: Int, speaker: Int, isStart: Bool)] = []
        boundaryFrames.reserveCapacity(segments.count * 2)
        
        for segment in segments {
            boundaryFrames.append((segment.startFrame, segment.speakerIndex, true))
            boundaryFrames.append((segment.endFrame, segment.speakerIndex, false))
        }
        
        // Sort by frame, with ends before starts at the same frame
        // This ensures adjacent segments (one ends where another starts) are handled correctly
        boundaryFrames.sort {
            if $0.frame != $1.frame {
                return $0.frame < $1.frame
            }
            // At same frame: ends (isStart=false) come before starts (isStart=true)
            return !$0.isStart && $1.isStart
        }
        
        // Build the disjoint intervals
        var startFrame = boundaryFrames[0].frame
        var activeIds: Set<Int> = []
        
        var newDisjoint: [EmbeddingSegment] = []
        var newTentativeDisjoint: [EmbeddingSegment] = []
        
        let firstTentativeFrame = firstTentativeFrame ?? nextFrame - tentativePadding
        
        for (endFrame, speakerIndex, isStart) in boundaryFrames {
            // If exactly one speaker was active, this interval is a single-speaker segment
            if activeIds.count == 1, let activeSpeaker = activeIds.first, endFrame > startFrame {
                let isFinalized = endFrame < firstTentativeFrame
                let newSegment = EmbeddingSegment(
                    speakerIndex: activeSpeaker,  // Use the active speaker, not the boundary speaker
                    startFrame: startFrame,
                    endFrame: endFrame,
                    finalized: isFinalized
                )
                
                if isFinalized {
                    newDisjoint.append(newSegment)
                    
                    // Generate embedding requests for this new segment
                    let requests = newSegment.getEmbeddingRequests(
                        maxGapSize: embeddingConfig.maxEmbeddingGap,
                        maxEmbeddingLength: embeddingConfig.maxEmbeddingFrames
                    )
                    pendingEmbeddingRequests.append(contentsOf: requests)
                } else {
                    newTentativeDisjoint.append(newSegment)
                }
            }
            
            // Update state for next interval
            startFrame = endFrame
            
            if isStart {
                activeIds.insert(speakerIndex)
            } else {
                activeIds.remove(speakerIndex)
            }
        }
        
        // Merge adjacent same-speaker segments with small gaps
        let mergedFinalized = mergeAdjacentSegments(newDisjoint)
        let mergedTentative = mergeAdjacentSegments(newTentativeDisjoint)
        
        return (mergedFinalized, mergedTentative)
    }
    
    /// Merge adjacent segments for the same speaker if the gap is smaller than minSegmentGap
    private func mergeAdjacentSegments(_ segments: [EmbeddingSegment]) -> [EmbeddingSegment] {
        guard segments.count > 1 else { return segments }
        
        // Group by speaker
        var bySpeaker: [Int: [EmbeddingSegment]] = [:]
        for seg in segments {
            bySpeaker[seg.speakerIndex, default: []].append(seg)
        }
        
        var result: [EmbeddingSegment] = []
        
        for (_, speakerSegs) in bySpeaker {
            // Sort by start frame
            let sorted = speakerSegs.sorted { $0.startFrame < $1.startFrame }
            
            var merged: [EmbeddingSegment] = []
            for seg in sorted {
                if let last = merged.last,
                   seg.startFrame - last.endFrame < embeddingConfig.minSegmentGap {
                    // Merge: extend the last segment
                    last.endFrame = seg.endFrame
                    // Transfer embeddings
                    last.addEmbeddings(seg.embeddings)
                } else {
                    merged.append(seg)
                }
            }
            result.append(contentsOf: merged)
        }
        
        // Filter out segments that are too short for embedding extraction
        let filtered = result.filter { $0.length >= embeddingConfig.minEmbeddingFrames }
        
        // Sort by start frame
        return filtered.sorted { $0.startFrame < $1.startFrame }
    }
    
    /// Reconstruct streaming state at a specific frame
    /// Used when re-processing predictions from a specific point (e.g., after filter update)
    /// The returned state represents what the state would be JUST BEFORE processing frame `frame`
    private func getStreamingState(atFrame frame: Int) -> StreamingState {
        guard frame <= nextFrame else {
            return state
        }
        
        var reconstructedState = StreamingState(numSpeakers: config.numSpeakers)
        
        for speakerIndex in 0..<config.numSpeakers {
            // Check if the current segment is tentative
            if let firstTentative = tentativeSegments[speakerIndex].first,
               firstTentative.startFrame < frame {
                reconstructedState.starts[speakerIndex] = firstTentative.startFrame
                
                if let lastConfirmed = segments[speakerIndex].last {
                    reconstructedState.lastSegments[speakerIndex] = (lastConfirmed.startFrame, lastConfirmed.endFrame)
                }
                reconstructedState.isSpeaking[speakerIndex] = firstTentative.contains(frame - 1)
                
                continue
            }

            // Find the oldest segment that starts before this one
            let speakerSegments = self.segments[speakerIndex]
            guard let indexOfCurrentSegment = speakerSegments.lastIndex(
                where: { $0.startFrame < frame })
            else {
                // By default, the state for this speaker will be the initial state
                continue
            }
            
            let currentSegment = speakerSegments[indexOfCurrentSegment]
            
            // If the segment has ended before or exactly at `frame`, we're not speaking anymore
            if currentSegment.endFrame <= frame {
                // Segment already closed - we're NOT speaking
                reconstructedState.isSpeaking[speakerIndex] = false
                reconstructedState.lastSegments[speakerIndex] = (currentSegment.startFrame, currentSegment.endFrame)
                // Don't set starts - no open segment
            } else {
                // Segment is still ongoing at `frame`
                reconstructedState.starts[speakerIndex] = currentSegment.startFrame
                reconstructedState.isSpeaking[speakerIndex] = currentSegment.contains(frame - 1)
                
                if indexOfCurrentSegment > 0 {
                    let prevSegment = speakerSegments[indexOfCurrentSegment - 1]
                    reconstructedState.lastSegments[speakerIndex] = (prevSegment.startFrame, prevSegment.endFrame)
                }
            }
        }
        
        return reconstructedState
    }
    
    /// Apply a difference to the timeline segments
    /// - Parameter diff: The difference containing insertions, deletions, and updates
    private func applyDifference(_ diff: SortformerTimelineDifference) {
        guard !diff.isEmpty else { return }
        
        // Apply updates - modify existing segment boundaries
        for update in diff.updates {
            let speakerIndex = update.speakerIndex
            guard speakerIndex < config.numSpeakers else { continue }
            
            if let segmentIndex = segments[speakerIndex].firstIndex(where: { $0.id == update.segmentID }) {
                segments[speakerIndex][segmentIndex].startFrame = update.newStartFrame
                segments[speakerIndex][segmentIndex].endFrame = update.newEndFrame
                
                #if DEBUG
                print("📝 [applyDifference] Updated speaker \(speakerIndex) segment: [\(update.oldStartFrame)-\(update.oldEndFrame)] -> [\(update.newStartFrame)-\(update.newEndFrame)]")
                #endif
            }
        }
        
        // Apply deletions - remove segments by ID
        if !diff.deletions.isEmpty {
            let deletionIDs = Set(diff.deletions.map { $0.segmentID })
            
            for speakerIndex in 0..<config.numSpeakers {
                segments[speakerIndex].removeAll { deletionIDs.contains($0.id) }
            }
        }
        
        // Apply insertions - add new segments
        for insertion in diff.insertions {
            let speakerIndex = insertion.speakerIndex
            guard speakerIndex < config.numSpeakers else { continue }
            
            let newSegment = SortformerSegment(
                speakerIndex: speakerIndex,
                startFrame: insertion.startFrame,
                endFrame: insertion.endFrame,
                finalized: true,
                frameDurationSeconds: config.frameDurationSeconds
            )
            
            // Non-overlapping sorted segments: insert after the last one that starts before this
            if let lastBeforeIndex = segments[speakerIndex].lastIndex(where: { $0.startFrame < newSegment.startFrame }) {
                segments[speakerIndex].insert(newSegment, at: lastBeforeIndex + 1)
            } else {
                segments[speakerIndex].insert(newSegment, at: 0)
            }
        }
    }
    
    /// Reset the timeline to initial state
    public func reset() {
        framePredictions.removeAll()
        tentativePredictions.removeAll()
        nextFrame = 0
        state = StreamingState(numSpeakers: config.numSpeakers)
        segments = Array(repeating: [], count: config.numSpeakers)
        tentativeSegments = Array(repeating: [], count: config.numSpeakers)
        embeddingSegments.removeAll()
        tentativeEmbeddingSegments.removeAll()
        pendingEmbeddingRequests.removeAll()
    }

    /// Finalize all tentative data at end of recording
    /// Call this when no more chunks will be added to convert all tentative predictions and segments to finalized
    public func finalize() {
        framePredictions.append(contentsOf: self.tentativePredictions)
        nextFrame += numTentative
        tentativePredictions.removeAll()
        for i in 0..<config.numSpeakers {
            segments[i].append(contentsOf: tentativeSegments[i])
            tentativeSegments[i].removeAll()

            if let lastSegment = segments[i].last, lastSegment.length < config.minFramesOn {
                segments[i].removeLast()
            }
        }
        
        // Finalize tentative embedding segments
        for tentativeSeg in tentativeEmbeddingSegments {
            tentativeSeg.isFinalized = true
            embeddingSegments.append(tentativeSeg)
            
            // Only generate requests for segments long enough
            guard tentativeSeg.length >= embeddingConfig.minEmbeddingFrames else { continue }
            
            // Generate embedding requests for newly finalized segments
            let requests = tentativeSeg.getEmbeddingRequests(
                maxGapSize: embeddingConfig.maxEmbeddingGap,
                maxEmbeddingLength: embeddingConfig.maxEmbeddingFrames,
                minEmbeddingLength: embeddingConfig.minEmbeddingFrames
            )
            pendingEmbeddingRequests.append(contentsOf: requests)
        }
        tentativeEmbeddingSegments.removeAll()
        
        trimPredictions()
    }

    /// Get probability for a specific speaker at a specific finalized frame
    public func probability(speaker: Int, frame: Int) -> Float {
        guard frame < nextFrame, speaker < config.numSpeakers else { return 0.0 }
        return framePredictions[frame * config.numSpeakers + speaker]
    }

    /// Get tentative probability for a specific speaker at a specific tentative frame
    public func tentativeProbability(speaker: Int, frame: Int) -> Float {
        guard frame < numTentative, speaker < config.numSpeakers else { return 0.0 }
        return tentativePredictions[frame * config.numSpeakers + speaker]
    }

    /// Trim predictions to not take up so much space
    private func trimPredictions() {
        guard let maxStoredFrames = config.maxStoredFrames else {
            return
        }

        let numToRemove = framePredictions.count - maxStoredFrames * config.numSpeakers

        if numToRemove > 0 {
            framePredictions.removeFirst(numToRemove)
        }
    }
}

/// A single speaker segment from Sortformer
/// Can be mutated during streaming processing
public struct SortformerSegment: Sendable, Identifiable, SpeakerFrameRange {
    /// Segment ID
    public let id: UUID

    /// Speaker index in Sortformer output
    public var speakerIndex: Int

    /// Index of segment start frame
    public var startFrame: Int

    /// Index of segment end frame
    public var endFrame: Int

    /// Range of frames that this segment covers
    public var frames: Range<Int> { startFrame..<endFrame }

    /// Length of the segment in frames
    public var length: Int { endFrame - startFrame }

    /// Whether this segment is finalized
    public var isFinalized: Bool

    /// Start time in seconds
    public var startTime: Float { Float(startFrame) * frameDurationSeconds }

    /// End time in seconds
    public var endTime: Float { Float(endFrame) * frameDurationSeconds }

    /// Duration in seconds
    public var duration: Float { Float(endFrame - startFrame) * frameDurationSeconds }

    /// Duration of one frame in seconds
    public let frameDurationSeconds: Float

    /// Speaker label (e.g., "Speaker 0")
    public var speakerLabel: String {
        "Speaker \(speakerIndex)"
    }

    public init(
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        finalized: Bool = true,
        frameDurationSeconds: Float = 0.08
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
    }

    public init(
        speakerIndex: Int,
        startTime: Float,
        endTime: Float,
        finalized: Bool = true,
        frameDurationSeconds: Float = 0.08
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = Int(round(startTime / frameDurationSeconds))
        self.endFrame = Int(round(endTime / frameDurationSeconds))
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
    }
    
    /// Check if the segment contains a frame
    public func contains(_ frame: Int) -> Bool {
        return frames.contains(frame)
    }

    /// Check if this segment overlaps or touches another segment
    public func isContiguous<T>(with other: T) -> Bool
    where T: SortformerFrameRange {
        return (self.startFrame <= other.endFrame) && (other.startFrame <= self.endFrame)
    }
    
    /// Check if this overlaps with another segment
    public func overlaps<T>(with other: T) -> Bool
    where T: SortformerFrameRange {
        return frames.overlaps(other.frames)
    }

    /// Merge another segment into this one
    public mutating func absorb<T>(_ other: T)
    where T: SortformerFrameRange {
        self.startFrame = min(self.startFrame, other.startFrame)
        self.endFrame = max(self.endFrame, other.endFrame)
    }

    /// Extend the end of this segment
    public mutating func extendEnd(toFrame endFrame: Int) {
        self.endFrame = max(self.endFrame, endFrame)
    }

    /// Extend the start of this segment
    public mutating func extendStart(toFrame startFrame: Int) {
        self.startFrame = min(self.startFrame, startFrame)
    }
}


public enum SortformerTimelineError: Error, LocalizedError {
    case mismatchedPredSizes
    
    public var errorDescription: String? {
        switch self {
        case .mismatchedPredSizes:
            return "Preds and filtered preds should be the same size"
            
        }
    }
}
