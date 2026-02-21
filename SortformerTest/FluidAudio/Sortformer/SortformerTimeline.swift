import Foundation
import Accelerate
import OrderedCollections


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
        
        mutating func resetSlot(at index: Int) {
            starts[index] = 0
            isSpeaking[index] = false
            lastSegments[index] = (0, 0)
        }
        
        mutating func shiftBackSlot(at index: Int) {
            starts[index-1] = starts[index]
            isSpeaking[index-1] = isSpeaking[index]
            lastSegments[index-1] = lastSegments[index]
        }
    }
    
    /// Post-processing configuration
    public let config: SortformerPostProcessingConfig
    
    /// Embedding extraction configuration
    public let embeddingManager: EmbeddingManager
    
    public var embeddingConfig: EmbeddingConfig { embeddingManager.config }

    /// Finalized frame-wise speaker predictions
    /// Shape: [numFrames, numSpeakers]
    public private(set) var framePredictions: [Float] = []

    /// Tentative predictions
    /// Shape: [numTentative, numSpeakers]
    public private(set) var tentativePredictions: [Float] = []
    
    /// Total number of finalized median-filtered frames
    public private(set) var cursorFrame: Int = 0
    
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
    
    /// Get total duration of finalized predictions in seconds
    public var duration: Float {
        Float(cursorFrame) * config.frameDurationSeconds
    }

    /// Get total duration including tentative predictions in seconds
    public var tentativeDuration: Float {
        Float(cursorFrame + numTentative) * config.frameDurationSeconds
    }

    /// Active segments being built (one per speaker, nil if speaker not active)
    private var state: StreamingState
    
    /// Filter
    private let filter: SortformerFilter
    
    /// For preserving the UUID of tentative segments
    private var segmentIDs: [SegmentKey: UUID] = [:]

    /// Logger for warnings
    static let logger = AppLogger(category: "SortformerTimeline")
    
    private let queue = DispatchQueue(label: "ai.swift.sortformer.vectorclustering")

    /// Initialize with configuration for streaming usage
    /// - Parameters:
    ///   - config: Sortformer post-processing configuration
    ///   - embeddingConfig: Embedding extraction configuration
    public init(
        config: SortformerPostProcessingConfig = .default(for: .default),
        embeddingManager: EmbeddingManager
    ) {
        self.config = config
        self.embeddingManager = embeddingManager
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
        embeddingManager: EmbeddingManager,
        config: SortformerPostProcessingConfig = .default(for: .default),
        isComplete: Bool = true
    ) throws {
        self.init(config: config, embeddingManager: embeddingManager)
        let updateResult = SortformerStateUpdateResult(
            firstNewFrame: 0,
            newPredictions: allPredictions,
            newFrameCount: allPredictions.count / config.numSpeakers,
            oldPredictions: [],
            oldFrameCount: 0,
            finalizedFrameCount: 0
        )
        
        _ = try self.addChunk(updateResult, dropOldEmbeddingFrames: true)

        if isComplete {
            // Finalize everything immediately
            try finalize()
        }
    }

    /// Add a new chunk of predictions from the diarizer
    public func addChunk(_ chunk: SortformerStateUpdateResult, dropOldEmbeddingFrames: Bool = true) throws -> SortformerTimelineDifference {
        try queue.sync(flags: .barrier) {
            // Apply EMA filter to existing predictions using FIFO as reference
            // This smooths the tail of framePredictions before appending new data
            let updatedFrameCount = min(numTentative, chunk.oldFrameCount, filter.windowSize)
            let updatedPredCount = updatedFrameCount * config.numSpeakers
            
            if !(tentativePredictions.isEmpty || chunk.oldPredictions.isEmpty) {
                try tentativePredictions.withUnsafeMutableBufferPointer { currentPtr in
                    try chunk.oldPredictions.withUnsafeBufferPointer { incomingPtr in
                        let currentBase = currentPtr.baseAddress!
                            .advanced(by: currentPtr.count - updatedPredCount)
                        let incomingBase = incomingPtr.baseAddress!
                            .advanced(by: incomingPtr.count - updatedPredCount)
                        try filter.update(currentBase, with: incomingBase, result: currentBase, count: updatedPredCount)
                    }
                }
            }
            
            // Now safe to clear tentative segments
            for i in 0..<config.numSpeakers {
                for segment in tentativeSegments[i] {
                    segmentIDs[segment.key] = segment.id
                }
                tentativeSegments[i].removeAll(keepingCapacity: true)
            }
            
            // Append new predictions BEFORE extracting segments
            // This ensures segments that span the boundary are properly extracted
            let predsToFinalize = chunk.finalizedFrameCount * config.numSpeakers
            var newSegments: [SortformerSegment] = []
            let oldTentative = tentativeSegments
            
            let oldFinalizedSegmentCounts = segments.map(\.count)
            
            // Add finalized preds if there are new ones
            tentativePredictions.append(contentsOf: chunk.newPredictions)
            
            if predsToFinalize > 0 {
                let finalizedPreds = tentativePredictions.prefix(predsToFinalize)
                framePredictions.append(contentsOf: finalizedPreds)
                tentativePredictions.removeFirst(predsToFinalize)
                
                updateSegments(
                    predictions: finalizedPreds,
                    numFrames: chunk.finalizedFrameCount,
                    isFinalized: true,
                    addTrailingTentative: false,
                    accumulator: &newSegments
                )
                
                cursorFrame += chunk.finalizedFrameCount
            }
            print("Predictions: \(chunk.finalizedFrameCount) / \(chunk.oldFrameCount+chunk.newFrameCount) finalized")
            updateSegments(
                predictions: tentativePredictions,
                numFrames: numTentative,
                isFinalized: false,
                addTrailingTentative: true,
                accumulator: &newSegments
            )

            // Clear garbage segment IDs
            segmentIDs.removeAll(keepingCapacity: true)
            
            try updateEmbeddingSegments(from: newSegments, finalizedCutoffs: oldFinalizedSegmentCounts, dropEmbeddingFrames: dropOldEmbeddingFrames)
            
            // Trim predictions
            trimPredictions()
            
            // Compute difference
            let diff = SortformerTimelineDifference(
                old: oldTentative.flatMap(\.self),
                new: newSegments
            )
            return diff
        }
    }
    
    private func leastActiveSlot() -> Int {
        var lowestEnergy: Float = .infinity
        var lowestEnergyIndex: Int = 0
        
        let stride = vDSP_Stride(config.numSpeakers)
        let length = vDSP_Length(numTentative)
        for i in (0..<config.numSpeakers) {
            var totalEnergy: Float = 0
            vDSP_sve(tentativePredictions, stride, &totalEnergy, length)
            if totalEnergy < lowestEnergy {
                lowestEnergy = totalEnergy
                lowestEnergyIndex = i
            }
        }
        
        return lowestEnergyIndex
    }
    
    /// Remove the speaker at a given index
    public func removeSpeaker(_ speakerIndex: Int) {
        queue.sync(flags: .barrier) {
            let slotsToMove = config.numSpeakers - speakerIndex - 1
            guard slotsToMove > 0 else {
                func clearPreds(preds: inout [Float], totalFrames: Int, speakerIndex: Int) {
                    preds.withUnsafeMutableBufferPointer { predsBuf in
                        guard let p = predsBuf.baseAddress else { return }
                        vDSP_vclr(p + speakerIndex, vDSP_Stride(config.numSpeakers), vDSP_Length(totalFrames))
                    }
                }
                
                clearPreds(preds: &framePredictions, totalFrames: numFinalized, speakerIndex: speakerIndex)
                clearPreds(preds: &tentativePredictions, totalFrames: numTentative, speakerIndex: speakerIndex)
                segments[speakerIndex].removeAll()
                tentativeSegments[speakerIndex].removeAll()
                embeddingSegments.removeAll(where: { $0.slot == speakerIndex })
                tentativeEmbeddingSegments.removeAll(where: { $0.slot == speakerIndex })
                state.resetSlot(at: speakerIndex)
                return
            }
            
            func shiftPreds(preds: inout [Float], totalFrames: Int, speakerIndex: Int) {
                let S = config.numSpeakers
                preds.withUnsafeMutableBufferPointer { predsBuf in
                    guard let p = predsBuf.baseAddress else { return }
                    let n = Int32(totalFrames)
                    let stride = Int32(S)
                    
                    // Shift each column left by one (process in ascending order so
                    // column k-1 is already saved before column k overwrites it).
                    for k in speakerIndex + 1..<S {
                        cblas_scopy(n, p + k, stride, p + k - 1, stride)
                    }
                    
                    // Zero the vacated last column
                    vDSP_vclr(p + (S - 1), vDSP_Stride(S), vDSP_Length(totalFrames))
                }
            }
            
            func shiftSegments(segments: inout [[SortformerSegment]], after index: Int) {
                for speakerIndex in index+1..<config.numSpeakers {
                    for i in 0..<segments[speakerIndex].count {
                        segments[speakerIndex][i].slot -= 1
                    }
                    segments[speakerIndex - 1] = segments[speakerIndex]
                }
                segments[config.numSpeakers - 1].removeAll()
            }
            
            func shiftEmbeddingSegments(segments: inout [EmbeddingSegment], after index: Int) {
                segments.removeAll(where: { $0.slot == index })
                for i in 0..<segments.count where segments[i].slot > index {
                    segments[i].slot -= 1
                }
            }
            
            shiftPreds(preds: &framePredictions, totalFrames: numFinalized, speakerIndex: speakerIndex)
            shiftPreds(preds: &tentativePredictions, totalFrames: numTentative, speakerIndex: speakerIndex)
            shiftSegments(segments: &segments, after: speakerIndex)
            shiftSegments(segments: &tentativeSegments, after: speakerIndex)
            shiftEmbeddingSegments(segments: &embeddingSegments, after: speakerIndex)
            shiftEmbeddingSegments(segments: &tentativeEmbeddingSegments, after: speakerIndex)
            
            for i in speakerIndex + 1..<config.numSpeakers {
                state.shiftBackSlot(at: i)
            }
            state.resetSlot(at: config.numSpeakers - 1)
        }
    }

    /// Helper to update segments from predictions
    /// Appends the new segments to `accumulator`
    ///
    /// - Parameters:
    ///   - predictions: Frame-level predictions shaped as [numFrames, numSpk] flattened
    ///   - numFrames: Number of frames
    ///   - isFinalized: Whether the predictions are finalized
    ///   - addTrailingTentative: Whether add tentative segments with no close.
    ///   - accumulator: Where to continue building the flattened results
    private func updateSegments<T>(
        predictions: T,
        numFrames: Int,
        isFinalized: Bool,
        addTrailingTentative: Bool,
        accumulator: inout [SortformerSegment],
    ) where T: Sequence & Collection, T.Element == Float, T.Index == Int {
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
            tentativeStartFrame = (cursorFrame + numFrames) - config.minUnpaddedGap
        } else {
            tentativeStartFrame = 0
        }
        
        for speakerIndex in 0..<numSpeakers {
            var start = state.starts[speakerIndex]
            var speaking = state.isSpeaking[speakerIndex]
            var lastSegment = state.lastSegments[speakerIndex]
            var wasLastSegmentFinal = isFinalized

            for i in 0..<numFrames {
                let index = i * numSpeakers + speakerIndex

                if speaking {
                    guard predictions[index] < offset else {
                        continue
                    }

                    // Speaking -> not speaking
                    speaking = false
                    let end = cursorFrame + i + padOffset

                    // Ensure segment is long enough
                    guard end - start >= minFramesOn else {
                        continue
                    }

                    // Segment is only finalized if it ends BEFORE the tentative boundary
                    // This ensures gap-closer can still merge it with future segments
                    wasLastSegmentFinal = isFinalized && (end <= tentativeStartFrame)

                    let newSegment = SortformerSegment(
                        id: segmentIDs.removeValue(forKey: SegmentKey(speakerIndex: speakerIndex, start: start, end: end)),
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: wasLastSegmentFinal,
                        frameDurationSeconds: config.frameDurationSeconds
                    )

                    accumulator.append(newSegment)
                    if wasLastSegmentFinal {
                        segments[speakerIndex].append(newSegment)
                    } else {
                        tentativeSegments[speakerIndex].append(newSegment)
                    }
                    lastSegment = (start, end)

                } else if predictions[index] > onset {
                    // Not speaking -> speaking
                    start = max(0, cursorFrame + i - padOnset)
                    speaking = true

                    if start - lastSegment.end <= minFramesOff {
                        // Extend last segment to avoid overlap, effectively merging it with the new one
                        start = lastSegment.start
                        
                        let removedItem = wasLastSegmentFinal
                            ? segments[speakerIndex].popLast()
                            : tentativeSegments[speakerIndex].popLast()

                        if removedItem != nil {
                            accumulator.removeLast()
                        }
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
                let end = cursorFrame + numFrames + padOffset
                if speaking && (end > start) {
                    let newSegment = SortformerSegment(
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: false,
                        frameDurationSeconds: config.frameDurationSeconds
                    )
                    tentativeSegments[speakerIndex].append(newSegment)
                    accumulator.append(newSegment)
                } else {
                    print ("Speaking: \(speaking), start: \(start), end: \(end)")
                }
            }
            
        }
    }
    
    /// Note: call this AFTER `self.numFrames` has been updated
    private func updateEmbeddingSegments(
        from segments: [SortformerSegment],
        finalizedCutoffs oldFinalizedSegmentCounts: [Int],
        dropEmbeddingFrames: Bool
    ) throws {
        guard !segments.isEmpty else {
            return
        }
        
        // Recycle old embeddings
        embeddingManager.returnEmbeddings(from: tentativeEmbeddingSegments)
        
        // Compute tentative boundaries
        let minSegmentGap = embeddingConfig.minSegmentGap
        
        let endGap = max(config.minUnpaddedGap, minSegmentGap)
        let firstTentativeFrame = cursorFrame - endGap
        let streamingHorizonFrame = cursorFrame + numTentative - endGap
        
        var currentSegment: EmbeddingSegment = .none
        
        // Add additional finalized segments
        var segments = segments
        if let firstTentativeSegment = tentativeEmbeddingSegments.first,
           firstTentativeSegment.isValid,
           firstTentativeSegment.startFrame < firstTentativeFrame {
            
            let finalizedEnd = oldFinalizedSegmentCounts[firstTentativeSegment.slot]
            if let finalizedStart = self.segments[firstTentativeSegment.slot].prefix(finalizedEnd)
                    .lastIndex(where: { firstTentativeSegment.startFrame > $0.endFrame })?
                    .advanced(by: 1),
               finalizedStart < finalizedEnd
            {
                segments.append(contentsOf: self.segments[firstTentativeSegment.slot][finalizedStart..<finalizedEnd])
            }
        }
        tentativeEmbeddingSegments.removeAll(keepingCapacity: true)
            
        // Get ordered boundary frames
        var boundaryFrames: [(frame: Int, speakerIndex: Int, isStart: Bool, id: UUID, isFinalized: Bool)] = []
        boundaryFrames.reserveCapacity(segments.count * 2)
        
        for segment in segments {
            boundaryFrames.append((segment.startFrame, segment.slot, true, segment.id, segment.isFinalized))
            boundaryFrames.append((segment.endFrame, segment.slot, false, segment.id, segment.isFinalized))
        }
        
        // Sort by frame, with ends before starts at the same frame
        // This ensures adjacent segments (one ends where another starts) are handled correctly
        boundaryFrames.sort {
            ($0.frame < $1.frame) ||
            ($0.frame == $1.frame && !$0.isStart)
        }
        
        func appendSegment(_ segment: EmbeddingSegment) throws {
            guard segment.isValid else {
                return
            }
            
            // Add embeddings to the segment. It can't be updated anymore.
            try currentSegment.initializeEmbeddings(
                with: embeddingManager,
                streamingHorizonFrame: streamingHorizonFrame
            )
            
            guard !segment.embeddings.isEmpty else {
                return
            }
            
            // Append the embedding segment
            if segment.isFinalized {
                if embeddingSegments.last?.successfullyAbsorbed(segment) != true {
                    embeddingSegments.append(segment)
                }
            } else if tentativeEmbeddingSegments.last?.successfullyAbsorbed(segment) != true {
                tentativeEmbeddingSegments.append(segment)
            }
        }
        
        // Extract non-overlapping segments
        let firstBoundaryFrame = boundaryFrames.removeFirst()
        
        var activeSpeakerIds: [Int : (id: UUID, finalized: Bool)] = [firstBoundaryFrame.speakerIndex : (firstBoundaryFrame.id, firstBoundaryFrame.isFinalized) ]
        var startFrame: Int = firstBoundaryFrame.frame
        
        for (endFrame, speakerIndex, isStart, id, finalized) in boundaryFrames {
            // If exactly one speaker was active, this interval is a single-speaker segment
            if activeSpeakerIds.count == 1,
               endFrame > startFrame,
               let (activeSpeaker, (activeSegment, isActiveFinalized)) = activeSpeakerIds.first
            {
                let isFinalized = isActiveFinalized && endFrame <= firstTentativeFrame
                
                if currentSegment.isValid,
                   activeSpeaker == currentSegment.slot,
                   startFrame - currentSegment.endFrame < minSegmentGap
                {
                    // Merge with the previous segment
                    currentSegment.endFrame = endFrame
                    currentSegment.isFinalized = currentSegment.isFinalized && isFinalized
                    if currentSegment.segmentIds.last != activeSegment {
                        currentSegment.segmentIds.append(activeSegment)
                    }
                } else {
                    try appendSegment(currentSegment)
                    
                    // Make a new segment
                    currentSegment = EmbeddingSegment(
                        speakerIndex: activeSpeaker,
                        startFrame: startFrame,
                        endFrame: endFrame,
                        finalized: isFinalized,
                        segmentId: activeSegment
                    )
                }
            }
            
            // Update state for next interval
            startFrame = endFrame
            
            if isStart {
                activeSpeakerIds[speakerIndex] = (id, finalized)
            } else {
                activeSpeakerIds.removeValue(forKey: speakerIndex)
            }
        }
        
        // Initialize embeddings for the remaining active segment
        try appendSegment(currentSegment)
        
        // Clean up spare embeddings
        if dropEmbeddingFrames {
            embeddingManager.dropFrames(
                before: firstTentativeFrame - embeddingConfig.maxEmbeddingFrames)
        }
    }
    
    /// Reset the timeline to initial state
    public func reset() {
        framePredictions.removeAll()
        tentativePredictions.removeAll()
        cursorFrame = 0
        state = StreamingState(numSpeakers: config.numSpeakers)
        segments = Array(repeating: [], count: config.numSpeakers)
        tentativeSegments = Array(repeating: [], count: config.numSpeakers)
        embeddingSegments.removeAll()
        tentativeEmbeddingSegments.removeAll()
        embeddingManager.reset()
    }

    /// Finalize all tentative data at end of recording
    /// Call this when no more chunks will be added to convert all tentative predictions and segments to finalized
    public func finalize() throws {
        Self.logger.info("Finalizing timeline...")
        framePredictions.append(contentsOf: self.tentativePredictions)
        cursorFrame += numTentative
        tentativePredictions.removeAll()
        
        for i in 0..<config.numSpeakers {
            for j in 0..<tentativeSegments[i].count {
                tentativeSegments[i][j].isFinalized = true
            }
            segments[i].append(contentsOf: tentativeSegments[i])
            tentativeSegments[i].removeAll()

            if let lastSegment = segments[i].last, lastSegment.length < config.minFramesOn {
                segments[i].removeLast()
            }
        }

        // Finalize tentative embedding segments
        for i in 0..<tentativeEmbeddingSegments.count {
            tentativeEmbeddingSegments[i].isFinalized = true
            try tentativeEmbeddingSegments[i].initializeEmbeddings(with: embeddingManager, streamingHorizonFrame: cursorFrame)
        }

        embeddingSegments.append(contentsOf: tentativeEmbeddingSegments)
        tentativeEmbeddingSegments.removeAll()

        trimPredictions()
        Self.logger.info("Finished finalizing timeline")
    }

    /// Get probability for a specific speaker at a specific finalized frame
    public func probability(speaker: Int, frame: Int) -> Float {
        guard frame < cursorFrame, speaker < config.numSpeakers else { return 0.0 }
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
public final class SortformerSegment: @unchecked Sendable, Identifiable, Hashable, Comparable, SpeakerFrameRange {
    /// Segment ID
    public let id: UUID

    /// Speaker index in Sortformer output
    public var slot: Int

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
        "Speaker \(slot)"
    }
    
    public var key: SegmentKey { .init(from: self) }

    public init(
        id: UUID? = nil,
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        finalized: Bool = true,
        frameDurationSeconds: Float = 0.08
    ) {
        self.id = id ?? UUID()
        self.slot = speakerIndex
        self.startFrame = startFrame
        self.endFrame = endFrame
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
    }

    public init(
        id: UUID? = nil,
        speakerIndex: Int,
        startTime: Float,
        endTime: Float,
        finalized: Bool = true,
        frameDurationSeconds: Float = 0.08
    ) {
        self.id = id ?? UUID()
        self.slot = speakerIndex
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
        return SortformerFrameRangeHelpers.isContiguous(self, other)
    }
    
    /// Check if this segment part of the same segment as another one
    public func isContiguous<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool
    where T: SpeakerFrameRange {
        return SortformerFrameRangeHelpers.isContiguous(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    /// Check if this overlaps with another segment
    public func overlaps<T>(with other: T) -> Bool
    where T: SortformerFrameRange {
        return frames.overlaps(other.frames)
    }
    
    /// Check if this overlaps with another segment
    public func overlaps<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool
    where T: SpeakerFrameRange {
        SortformerFrameRangeHelpers.overlaps(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    /// Check if this overlaps with another segment
    public func overlapLength<T>(with other: T) -> Int
    where T: SortformerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, other)
    }
    
    /// Check if this overlaps with another segment
    public func overlapLength<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Int
    where T: SpeakerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }

    /// Merge another segment into this one
    public func absorb<T>(_ other: T)
    where T: SortformerFrameRange {
        self.startFrame = min(self.startFrame, other.startFrame)
        self.endFrame = max(self.endFrame, other.endFrame)
    }
    
    /// Merge another segment into this one
    public func absorbing<T>(_ other: T) -> SortformerSegment
    where T: SortformerFrameRange {
        let startFrame = min(self.startFrame, other.startFrame)
        let endFrame = max(self.endFrame, other.endFrame)
        
        return SortformerSegment(
            id: id,
            speakerIndex: slot,
            startFrame: startFrame,
            endFrame: endFrame,
            finalized: isFinalized,
            frameDurationSeconds: frameDurationSeconds
        )
    }

    /// Extend the end of this segment
    public func extendEnd(toFrame endFrame: Int) {
        self.endFrame = max(self.endFrame, endFrame)
    }

    /// Extend the start of this segment
    public func extendStart(toFrame startFrame: Int) {
        self.startFrame = min(self.startFrame, startFrame)
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(startFrame)
        hasher.combine(endFrame)
        hasher.combine(slot)
//        hasher.combine(id)
    }
    
    public static func == (lhs: SortformerSegment, rhs: SortformerSegment) -> Bool {
        return lhs.key == rhs.key
    }
    
    public static func < (lhs: SortformerSegment, rhs: SortformerSegment) -> Bool {
        (lhs.startFrame, lhs.endFrame) < (rhs.startFrame, rhs.endFrame)
    }
}

public struct SegmentKey: Hashable {
    let start: Int
    let end: Int
    let speakerIndex: Int
    
    public init(speakerIndex: Int, start: Int, end: Int) {
        self.start = start
        self.end = end
        self.speakerIndex = speakerIndex
    }
    
    public init<T>(from segment: T) where T: SpeakerFrameRange {
        self.start = segment.startFrame
        self.end = segment.endFrame
        self.speakerIndex = segment.slot
    }
    
    public func hash(into hasher: inout Hasher) {
        hasher.combine(start)
        hasher.combine(end)
        hasher.combine(speakerIndex)
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
