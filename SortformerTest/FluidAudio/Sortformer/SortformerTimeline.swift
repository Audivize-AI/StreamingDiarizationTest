import Foundation
import Accelerate
import OrderedCollections


/// Complete diarization timeline managing streaming predictions and segments
public class SortformerTimeline {
    typealias SegmentID = UInt64
    
    private struct StreamingState {
        var starts: [Int]
        var isSpeaking: [Bool]
        var lastSegments: [SpeakerSegment]
        
        init(numSpeakers: Int) {
            self.starts = Array(repeating: 0, count: numSpeakers)
            self.isSpeaking = Array(repeating: false, count: numSpeakers)
            self.lastSegments = Array(repeating: 0, count: numSpeakers)
        }
        
        mutating func resetSlot(at index: Int) {
            starts[index] = 0
            isSpeaking[index] = false
            lastSegments[index] = 0
        }
        
        mutating func shiftBackSlot(at index: Int) {
            starts[index-1] = starts[index]
            isSpeaking[index-1] = isSpeaking[index]
            lastSegments[index-1] = lastSegments[index]
        }
    }
    
    /// Post-processing configuration
    public let config: SortformerTimelineConfig
    
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
    public private(set) var speakers: SpeakerDatabase
    
    public var activeSpeakers: [Int : SpeakerProfile] { speakers.activeSpeakers }
    
    public var inactiveSpeakers: [Int : SpeakerProfile] { speakers.inactiveSpeakers }
    
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
    
    /// Logger for warnings
    static let logger = AppLogger(category: "SortformerTimeline")
    
    private let queue = DispatchQueue(label: "ai.swift.sortformer.vectorclustering")

    private var centroidCache: [UUID : SpeakerClusterCentroid] = [:]
    
    /// Initialize with configuration for streaming usage
    /// - Parameters:
    ///   - config: Sortformer post-processing configuration
    ///   - embeddingConfig: Embedding extraction configuration
    public init(
        config: SortformerTimelineConfig = .default(for: .default),
        embeddingManager: EmbeddingManager
    ) {
        self.config = config
        self.embeddingManager = embeddingManager
        self.state = StreamingState(numSpeakers: config.numSpeakers)
        self.speakers = .init(config: config)
        let weights = [Float](repeating: 0.25, count: config.numFilteredFrames)
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
        config: SortformerTimelineConfig = .default(for: .default),
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
            let updatedFrameCount = min(
                numTentative - config.filterLeftContext,
                chunk.oldFrameCount - config.filterLeftContext,
                filter.windowSize
            )
            
            let updatedPredCount = updatedFrameCount * config.numSpeakers
            
            if !(tentativePredictions.isEmpty || chunk.oldPredictions.isEmpty) {
                try tentativePredictions.withUnsafeMutableBufferPointer { currentPtr in
                    try chunk.oldPredictions.withUnsafeBufferPointer { incomingPtr in
                        let currentBase = currentPtr.baseAddress! + currentPtr.count - updatedPredCount
                        let incomingBase = incomingPtr.baseAddress! + incomingPtr.count - updatedPredCount
                        try filter.update(currentBase, with: incomingBase, result: currentBase, count: updatedPredCount)
                    }
                }
            }
            
            // Now safe to clear tentative segments
            for (slot, speaker) in activeSpeakers {
                speaker.clearTentativeSegments()
            }
            
            // Append new predictions BEFORE extracting segments
            // This ensures segments that span the boundary are properly extracted
            let predsToFinalize = chunk.finalizedFrameCount * config.numSpeakers
            var newSegments: [SpeakerSegment] = []
            let oldTentative = tentativeSegments
            
            let oldFinalizedSegmentCounts = speakers.activeSpeakers.map(\.count)
            
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
            
            updateSegments(
                predictions: tentativePredictions,
                numFrames: numTentative,
                isFinalized: false,
                addTrailingTentative: true,
                accumulator: &newSegments
            )
            
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
                embeddingSegments.removeAll(where: { $0.speakerId == speakerIndex })
                tentativeEmbeddingSegments.removeAll(where: { $0.speakerId == speakerIndex })
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
            
            func shiftSegments(segments: inout [[SpeakerSegment]], after index: Int) {
                for speakerIndex in index+1..<config.numSpeakers {
                    for i in 0..<segments[speakerIndex].count {
                        segments[speakerIndex][i].speakerId -= 1
                    }
                    segments[speakerIndex - 1] = segments[speakerIndex]
                }
                segments[config.numSpeakers - 1].removeAll()
            }
            
            func shiftEmbeddingSegments(segments: inout [EmbeddingSegment], after index: Int) {
                segments.removeAll(where: { $0.speakerId == index })
                for i in 0..<segments.count where segments[i].speakerId > index {
                    segments[i].speakerId -= 1
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
        accumulator: inout [SpeakerSegment],
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
            var segmentsAdded: Int = 0
            var start = state.starts[speakerIndex]
            var speaking = state.isSpeaking[speakerIndex]
            var lastSegment = state.lastSegments[speakerIndex]
            
            for i in 0..<numFrames {
                let index = i * numSpeakers + speakerIndex
                
                if speaking { // Await segment end
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
                    let isFinalized = isFinalized && (end <= tentativeStartFrame)
                    
                    let newSegment = SpeakerSegment(
                        speakerId: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        isFinalized: isFinalized
                    )
                    
                    accumulator.append(newSegment)
                    segmentsAdded += 1
                    if isFinalized {
                        speakers.activeSpeakers[speakerIndex, default: .init(config: config, speakerIndex: <#T##Int#>)].append(newSegment)
                    } else {
                        tentativeSegments[speakerIndex].append(newSegment)
                    }
                    lastSegment = (start, end, isFinalized)
                    
                } else if predictions[index] > onset { // Await segment start
                    // Not speaking -> speaking
                    start = max(0, cursorFrame + i - padOnset)
                    speaking = true
                    
                    if segmentsAdded > 0,
                       start - lastSegment.end <= minFramesOff {
                        // Extend last segment to avoid overlap, effectively merging it with the new one
                        start = lastSegment.start
                        
                        if lastSegment.isFinalized {
                            segments[speakerIndex].removeLast()
                        } else {
                            tentativeSegments[speakerIndex].removeLast()
                        }
                        
                        accumulator.removeLast()
                        segmentsAdded -= 1
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
                    let newSegment = SpeakerSegment(
                        speakerId: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        isFinalized: false
                    )
                    tentativeSegments[speakerIndex].append(newSegment)
                    accumulator.append(newSegment)
                }
            }
        }
    }
    
    /// Note: call this AFTER `self.numFrames` has been updated
    private func updateEmbeddingSegments(
        from segments: [SpeakerSegment],
        finalizedCutoffs oldFinalizedSegmentCounts: [Int],
        dropEmbeddingFrames: Bool
    ) throws {
        guard !segments.isEmpty else {
            return
        }
        
        // Recycle old embeddings
        embeddingManager.returnEmbeddings(from: tentativeEmbeddingSegments)
        
        for segment in tentativeEmbeddingSegments {
            centroidCache[segment.id] = segment.centroid
        }
        
        // Compute tentative boundaries
        let minSegmentGap = embeddingConfig.minFramesOff
        
        let endGap = max(config.minUnpaddedGap, minSegmentGap)
        let firstTentativeFrame = cursorFrame - endGap
        let streamingHorizonFrame = cursorFrame + numTentative - endGap
        
        var currentSegment: EmbeddingSegment = .none
        
        // Add additional finalized segments
        var segments = segments
        if let firstTentativeSegment = tentativeEmbeddingSegments.first,
           firstTentativeSegment.isValid,
           firstTentativeSegment.startFrame < firstTentativeFrame,
           let contextSegments = activeSpeakers[firstTentativeSegment.speakerId]?.finalizedSegments
        {
            let finalizedEnd = min(oldFinalizedSegmentCounts[firstTentativeSegment.speakerId], contextSegments.count)
            
            if let finalizedStart = contextSegments.prefix(finalizedEnd)
                .lastIndex(where: { firstTentativeSegment.startFrame > $0.endFrame })?
                .advanced(by: 1),
               finalizedStart < finalizedEnd
            {
                segments.append(contentsOf: contextSegments[finalizedStart..<finalizedEnd])
            }
        }
        
        tentativeEmbeddingSegments.removeAll(keepingCapacity: true)
        
        // Get ordered boundary frames
        var boundaryFrames: [(frame: Int, speakerIndex: Int, isStart: Bool, id: SegmentID, isFinalized: Bool)] = []
        boundaryFrames.reserveCapacity(segments.count * 2)
        
        for segment in segments {
            boundaryFrames.append((segment.startFrame, segment.speakerId, true, segment.id, segment.isFinalized))
            boundaryFrames.append((segment.endFrame, segment.speakerId, false, segment.id, segment.isFinalized))
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
                    embeddingSegments.last?.initializeCentroid(withCache: centroidCache)
                    embeddingSegments.append(segment)
                }
            } else if tentativeEmbeddingSegments.last?.successfullyAbsorbed(segment) != true {
                tentativeEmbeddingSegments.last?.initializeCentroid(withCache: centroidCache)
                tentativeEmbeddingSegments.append(segment)
            }
        }
        
        // Extract non-overlapping segments
        let firstBoundaryFrame = boundaryFrames.removeFirst()
        
        var activeSpeakerIds: [Int : (id: UInt64, finalized: Bool)] = [firstBoundaryFrame.speakerIndex : (firstBoundaryFrame.id, firstBoundaryFrame.isFinalized) ]
        var startFrame: Int = firstBoundaryFrame.frame
        
        for (endFrame, speakerIndex, isStart, id, finalized) in boundaryFrames {
            // If exactly one speaker was active, this interval is a single-speaker segment
            if activeSpeakerIds.count == 1,
               endFrame > startFrame,
               let (activeSpeaker, (activeSegment, isActiveFinalized)) = activeSpeakerIds.first
            {
                let isFinalized = isActiveFinalized && endFrame <= firstTentativeFrame
                
                if currentSegment.isValid,
                   activeSpeaker == currentSegment.speakerId,
                   startFrame - currentSegment.endFrame < minSegmentGap
                {
                    // Merge with the previous segment
                    currentSegment.endFrame = endFrame
                    currentSegment.isFinalized = currentSegment.isFinalized && isFinalized
                    if currentSegment.segments.last?.id != activeSegment {
                        currentSegment.segments.append(.init(integerLiteral: activeSegment))
                    }
                } else {
                    try appendSegment(currentSegment)
                    
                    // Make a new segment
                    currentSegment = EmbeddingSegment(
                        slot: activeSpeaker,
                        startFrame: startFrame,
                        endFrame: endFrame,
                        finalized: isFinalized,
                        segment: .init(integerLiteral: activeSegment)
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
        
        if !currentSegment.isFinalized {
            tentativeEmbeddingSegments.last?.initializeCentroid(withCache: centroidCache)
        } else {
            embeddingSegments.last?.initializeCentroid(withCache: centroidCache)
        }
        
        // Clean up spare embeddings
        if dropEmbeddingFrames {
            embeddingManager.dropFrames(
                before: firstTentativeFrame - embeddingConfig.maxEmbeddingFrames)
            centroidCache.removeAll(keepingCapacity: true)
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

extension SortformerTimeline {
    var speakerIdentitySegments: [SpeakerSegment] {
        segments.flatMap { $0 }
    }

    var tentativeSpeakerIdentitySegments: [SpeakerSegment] {
        tentativeSegments.flatMap { $0 }
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

// MARK: - Segment ID
public struct SpeakerSegment:
    SpeakerFrameRange,
    Identifiable,
    ExpressibleByIntegerLiteral,
    Hashable,
    Comparable,
    Equatable
{
    public typealias IntegerLiteralType = UInt64
    
    /*
     * MEMORY LAYOUT:
     *     | speaker ID | is finalized | segment length | start frame |
     * LSB |   0 - 11   |      12      |     13 - 33    |   34 - 63   | MSB
     */
    
    // Sizes
    private static let startBits:   Int = 31    // About 5.5 years
    private static let lengthBits:  Int = 20    // About 1 day
    private static let speakerBits: Int = 12    // 4096 speaker slots
    
    // Offsets
    private static let finalizedOffset: Int = speakerBits
    private static let lengthOffset: Int = finalizedOffset + 1
    private static let startOffset: Int = lengthBits + lengthOffset
    
    // Bit masks
    private static let startMaskInv: UInt64 = (1 << (UInt64.bitWidth - startBits)) - 1
    private static let lengthFieldMask: UInt64 = (1 << lengthBits) - 1
    private static let lengthMaskInv: UInt64 = ~(lengthFieldMask << lengthOffset)
    private static let finalizedMask: UInt64 = 1 << finalizedOffset
    private static let finalizedMaskInv: UInt64 = ~finalizedMask
    private static let speakerFieldMask: UInt64 = (1 << speakerBits) - 1
    private static let speakerMaskInv: UInt64 = ~speakerFieldMask
    
    /// The segment's actual contents
    public private(set) var bits: UInt64
    
    // - MARK: - Attribute retrieval
    
    /// Sortable segment identifier
    @inline(__always)
    public var id: UInt64 {
        bits & Self.finalizedMaskInv
    }
    
    /// Copy of the segment with the speaker identity removed
    @inline(__always)
    public var anonymized: SpeakerSegment { .init(integerLiteral: bits | Self.speakerFieldMask) }
    
    /// Copy of the segment with the finalized flag set to `true`
    @inline(__always)
    public var finalized: SpeakerSegment { .init(integerLiteral: bits | Self.finalizedMask) }
    
    /// Copy of the segment with the finalized flag set to `false`
    @inline(__always)
    public var unfinalized: SpeakerSegment { .init(integerLiteral: bits & Self.finalizedMaskInv) }
    
    /// Segment start frame
    @inline(__always)
    public var startFrame: Int {
        get {
            Int(truncatingIfNeeded: bits >> Self.startOffset)
        }
        set {
            let v = UInt64(truncatingIfNeeded: newValue)
            bits = (bits & Self.startMaskInv) | (v << Self.startOffset)
        }
    }
    
    /// Segment end frame
    @inline(__always)
    public var endFrame: Int {
        get { startFrame + length }
        set { length = newValue - startFrame }
    }
    
    /// Segment length
    @inline(__always)
    public var length: Int {
        get {
            Int(truncatingIfNeeded: (bits >> Self.lengthOffset) & Self.lengthFieldMask)
        }
        set {
            let v = UInt64(truncatingIfNeeded: newValue) & Self.lengthFieldMask
            bits = (bits & Self.lengthMaskInv) | (v << Self.lengthOffset)
        }
    }
    
    /// Speaker ID
    @inline(__always)
    public var speakerId: Int {
        get {
            Int(truncatingIfNeeded: bits & Self.speakerFieldMask)
        }
        set {
            let v = UInt64(truncatingIfNeeded: newValue) & Self.speakerFieldMask
            bits = (bits & Self.speakerMaskInv) | v
        }
    }
    
    /// Whether the segment is finalized
    @inline(__always)
    public var isFinalized: Bool {
        get {
            (bits & Self.finalizedMask) != 0
        }
        set {
            let v = newValue ? 1 : 0 as UInt64
            bits = (bits & Self.finalizedMaskInv) | (v << Self.finalizedOffset)
        }
    }
    
    /// Whether the segment is tentative
    @inline(__always)
    public var isTentative: Bool {
        get { !isFinalized }
        set { isFinalized = !newValue }
    }
    
    /// Frame range
    @inline(__always)
    public var frames: Range<Int> { startFrame..<endFrame }
    
    /// Start time in seconds
    public var startTime: Float { Float(startFrame) * SortformerTimelineConfig.frameDurationSeconds }

    /// End time in seconds
    public var endTime: Float { Float(endFrame) * SortformerTimelineConfig.frameDurationSeconds }
    
    // MARK: - Init
    @inline(__always)
    public init(speakerId: Int, startFrame: Int, length: Int, isFinalized: Bool = true) {
        let startU64 = UInt64(truncatingIfNeeded: startFrame)
        let lengthU64 = UInt64(truncatingIfNeeded: length) & Self.lengthFieldMask
        let slotU64 = UInt64(truncatingIfNeeded: speakerId) & Self.speakerFieldMask
        let isFinalU64 = isFinalized ? 1 : 0 as UInt64
        
        self.bits = (  startU64 << Self.startOffset     | lengthU64 << Self.lengthOffset |
                     isFinalU64 << Self.finalizedOffset | slotU64)
    }
    
    @inline(__always)
    public init(speakerId: Int, startFrame: Int, endFrame: Int, isFinalized: Bool = true) {
        self.init(speakerId: speakerId, startFrame: startFrame, length: endFrame - startFrame, isFinalized: isFinalized)
    }
    
    @inline(__always)
    public init<T>(from segment: T, isFinalized: Bool = true)
    where T: SpeakerFrameRange {
        self.init(speakerId: segment.speakerId, startFrame: segment.startFrame, length: segment.length, isFinalized: isFinalized)
    }
    
    @inline(__always)
    public init(integerLiteral value: UInt64) {
        self.bits = value
    }
    
    // MARK: - Operators
    
    @inline(__always)
    public func hash(into hasher: inout Hasher) {
        hasher.combine(bits)
    }
    
    @inline(__always)
    public static func < (lhs: Self, rhs: Self) -> Bool {
        return lhs.bits < rhs.bits
    }
    
    // MARK: - Speaker Frame Range
    
    /// Check if the segment contains a frame
    @inline(__always)
    public func contains(_ frame: Int) -> Bool {
        return frames.contains(frame)
    }

    /// Check if this segment overlaps or touches another segment
    @inline(__always)
    public func isContiguous<T>(with other: T) -> Bool
    where T: SortformerFrameRange {
        return SortformerFrameRangeHelpers.isContiguous(self, other)
    }
    
    /// Check if this segment part of the same segment as another one
    @inline(__always)
    public func isContiguous<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool
    where T: SpeakerFrameRange {
        return SortformerFrameRangeHelpers.isContiguous(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlaps<T>(with other: T) -> Bool
    where T: SortformerFrameRange {
        return frames.overlaps(other.frames)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlaps<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Bool
    where T: SpeakerFrameRange {
        SortformerFrameRangeHelpers.overlaps(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlapLength<T>(with other: T) -> Int
    where T: SortformerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, other)
    }
    
    /// Check if this overlaps with another segment
    @inline(__always)
    public func overlapLength<T>(with other: T, ensuringSameSpeaker: Bool = true) -> Int
    where T: SpeakerFrameRange {
        return SortformerFrameRangeHelpers.overlapLength(self, other, ensuringSameSpeaker: ensuringSameSpeaker)
    }
}
