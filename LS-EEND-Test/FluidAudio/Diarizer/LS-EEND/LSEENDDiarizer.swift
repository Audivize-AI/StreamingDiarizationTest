//
//  LSEENDDiarizer.swift
//  LS-EEND-Test
//
//  Streaming LS-EEND (Long-form Streaming End-to-End Neural Diarization)
//  implementation. Mirrors the Python CoreMLPipelineDiarizer's per-frame
//  semantics (STFT → log10-mel → CMN → subsample+context → T-block CoreML
//  call → finalize with silence flush). CPU-optimized: preallocated scratch,
//  vDSP for CMN, MLMultiArray reference swapping for state updates.
//

import AVFoundation
import Accelerate
import CoreML
import Foundation

public final class LSEENDDiarizer: Diarizer {

    // MARK: - Dependencies
    private var model: LSEENDModel? = nil
    private var preprocessor: LSEENDPreprocessor? = nil

    public var timeline: DiarizerTimeline

    // MARK: - Protocol properties

    public private(set) var isAvailable: Bool = false
    public private(set) var numFramesProcessed: Int = 0
    public let targetSampleRate: Int?
    public let modelFrameHz: Double?
    public let numSpeakers: Int?

    private var finalized: Bool = false

    /// Frames of CNN-warmup output remaining to drop before predictions map
    /// 1:1 to real audio time. Initialized to `metadata.convDelay` at every
    /// reset; decremented as warmup predictions are dropped from either
    /// streaming `process()` or bulk `drainAndUpdate`.
    private var warmupFramesRemaining: Int = 0

    // MARK: - Init

    public init(model: LSEENDModel) throws {
        self.model = model
        let metadata = model.metadata
        self.preprocessor = try LSEENDPreprocessor(from: metadata)

        self.timeline = DiarizerTimeline(
            config: .default(
                numSpeakers: metadata.maxSpeakers,
                frameDurationSeconds: metadata.frameDurationSeconds
            )
        )

        self.targetSampleRate = metadata.sampleRate
        self.modelFrameHz = Double(metadata.sampleRate) / Double(metadata.hopLength * metadata.subsampling)
        self.numSpeakers = metadata.maxSpeakers
        self.warmupFramesRemaining = metadata.convDelay
        self.isAvailable = true
    }

    public func loadFromHuggingFace(
        variant: LSEENDVariant = .dihard3,
        stepSize: LSEENDStepSize = .step100ms,
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuOnly,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws {
        let model = try await LSEENDModel.loadFromHuggingFace(
            variant: variant,
            stepSize: stepSize,
            cacheDirectory: cacheDirectory,
            computeUnits: computeUnits,
            progressHandler: progressHandler
        )
        self.model = model
        self.preprocessor = try LSEENDPreprocessor(from: model.metadata)
        // Re-seed warmup counter + clear any prior streaming state — the
        // new model may have a different `convDelay`, so leaving stale
        // state around would mis-trim the first chunk after hot-swap.
        resetStreamingState()
    }

    // MARK: - Debug helpers (parity tests)

    #if DEBUG
    /// Drive `samples` through preprocessor → STFT → log10-mel → CMN →
    /// subsample+context stack, and return the flat `[N × featDim]`
    /// stacked features that would be fed to CoreML. Used by
    /// `testFeat345Parity` to byte-compare against the Python fixture
    /// without running inference.
    internal func debugExtractFeatures<C: Collection>(
        _ samples: C, sourceSampleRate: Double?
    ) throws -> [Float] where C.Element == Float {
        guard let preprocessor, let model else { throw LSEENDError.notInitialized }
        preprocessor.reset()
        try preprocessor.enqueueAudio(
            (samples as? [Float]) ?? Array(samples),
            withSampleRate: sourceSampleRate
        )
        try preprocessor.finalize()

        let featDim = model.metadata.featDim
        var out: [Float] = []
        while let input = try preprocessor.emitNextChunk() {
            // `input.melFeatures` is preallocated + reused — copy out each
            // pass. Caller-allocated input arrays have tight strides, so a
            // flat read is safe (unlike model *output* arrays, which get
            // tile-padded strides; see CLAUDE.md gotcha #2).
            input.melFeatures.withUnsafeBufferPointer(ofType: Float.self) { buf in
                out.append(contentsOf: buf)
            }
        }
        _ = featDim  // exported so tests can sanity-check division
        return out
    }
    #endif

    // MARK: - Streaming API

    public func addAudio<C: Collection>(_ samples: C, sourceSampleRate: Double?) throws
    where C.Element == Float
    {
        guard !samples.isEmpty else { return }
        guard let preprocessor else {
            throw LSEENDError.notInitialized
        }
        try preprocessor.enqueueAudio(
            (samples as? [Float]) ?? Array(samples),
            withSampleRate: sourceSampleRate
        )
    }

    public func process() throws -> DiarizerTimelineUpdate? {
        guard let preprocessor, let model else {
            throw LSEENDError.notInitialized
        }

        let chunkSize = model.metadata.chunkSize
        let numSpeakers = model.metadata.maxSpeakers
        var newPreds: [Float] = []
        newPreds.reserveCapacity(preprocessor.readyChunks * numSpeakers * chunkSize)

        while let input = try preprocessor.emitNextChunk() {
            newPreds.append(contentsOf: try model.predict(from: input))
        }

        trimWarmup(&newPreds, numSpeakers: numSpeakers)

        guard !newPreds.isEmpty else { return nil }
        numFramesProcessed += newPreds.count / numSpeakers

        return try timeline.addPredictions(
            finalizedPredictions: newPreds,
            tentativePredictions: []
        )
    }

    public func process<C: Collection>(
        samples: C, sourceSampleRate: Double?
    ) throws -> DiarizerTimelineUpdate? where C.Element == Float {
        try addAudio(samples, sourceSampleRate: sourceSampleRate)
        return try process()
    }

    // MARK: - Offline API

    public func processComplete<C: Collection>(
        _ samples: C,
        sourceSampleRate: Double?,
        keepingEnrolledSpeakers keepSpeakers: Bool?,
        finalizeOnCompletion: Bool,
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws -> DiarizerTimeline where C.Element == Float {
        guard preprocessor != nil, model != nil else {
            throw LSEENDError.notInitialized
        }
        let keep = keepSpeakers ?? !timeline.hasSegments
        resetStreamingState()
        timeline.reset(keepingSpeakers: keep)

        try addAudio(samples, sourceSampleRate: sourceSampleRate)
        try drainAndUpdate(
            finalizeOnCompletion: finalizeOnCompletion,
            progressCallback: progressCallback
        )
        return timeline
    }

    public func processComplete(
        audioFileURL: URL,
        keepingEnrolledSpeakers keepSpeakers: Bool?,
        finalizeOnCompletion: Bool,
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws -> DiarizerTimeline {
        guard let preprocessor, model != nil else {
            throw LSEENDError.notInitialized
        }
        let keep = keepSpeakers ?? !timeline.hasSegments
        resetStreamingState()
        timeline.reset(keepingSpeakers: keep)

        try preprocessor.enqueueAudioFile(at: audioFileURL)
        try drainAndUpdate(
            finalizeOnCompletion: finalizeOnCompletion,
            progressCallback: progressCallback
        )
        return timeline
    }

    /// Shared drain path for both `processComplete` overloads. Runs
    /// preprocessor → model → timeline, optionally finalizing the stream.
    private func drainAndUpdate(
        finalizeOnCompletion: Bool,
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws {
        guard let preprocessor, let model else {
            throw LSEENDError.notInitialized
        }

        if finalizeOnCompletion {
            try preprocessor.finalize()
        }

        let chunkSize = model.metadata.chunkSize
        let numSpeakers = model.metadata.maxSpeakers
        let totalChunks = preprocessor.readyChunks
        var processed = 0
        var newPreds: [Float] = []
        newPreds.reserveCapacity(totalChunks * numSpeakers * chunkSize)

        while let input = try preprocessor.emitNextChunk() {
            newPreds.append(contentsOf: try model.predict(from: input))
            processed += 1
            progressCallback?(processed, totalChunks, 1)
        }

        trimWarmup(&newPreds, numSpeakers: numSpeakers)
        numFramesProcessed += newPreds.count / numSpeakers
        _ = try timeline.addPredictions(
            finalizedPredictions: newPreds,
            tentativePredictions: []
        )

        if finalizeOnCompletion {
            timeline.finalize()
            finalized = true
        }
    }

    // MARK: - Lifecycle

    public func reset() {
        resetStreamingState()
        timeline.reset(keepingSpeakers: false)
    }

    public func cleanup() {
        resetStreamingState()
        self.model = nil
        self.preprocessor = nil
        isAvailable = false
    }

    public func enrollSpeaker<C: Collection>(
        withAudio samples: C,
        sourceSampleRate: Double?,
        named name: String?,
        overwritingAssignedSpeakerName overwriteAssignedSpeakerName: Bool
    ) throws -> DiarizerSpeaker? where C.Element == Float {
        guard let metadata = model?.metadata else {
            return nil
        }
        // LS-EEND attractors are learned online — there's no separable
        // speaker-embedding head to populate from enrollment audio. Two
        // behaviors are supported:
        //   1) name-only: reserve the first free slot with this name, don't
        //      touch streaming state.
        //   2) seeded: run audio through the diarizer so KV state conditions
        //      on it; name whichever slot fires strongest.
        // We pick (2) when audio is provided, (1) otherwise.
        guard let name = name else { return nil }

        if samples.isEmpty {
            return timeline.upsertSpeaker(named: name, atIndex: nil)
        }

        let update = try process(samples: samples, sourceSampleRate: sourceSampleRate)
        let finalizedFrames = update?.chunkResult.finalizedPredictions ?? []
        let maxSpk = metadata.maxSpeakers

        var bestSlot = -1
        var bestActivity: Float = -1
        if !finalizedFrames.isEmpty {
            let frames = finalizedFrames.count / maxSpk
            for slot in 0..<maxSpk {
                var sum: Float = 0
                for f in 0..<frames {
                    sum += finalizedFrames[f * maxSpk + slot]
                }
                if sum > bestActivity {
                    bestActivity = sum
                    bestSlot = slot
                }
            }
        }

        let slot = bestSlot >= 0 ? bestSlot : nil
        if let slot, let existing = timeline.speakers[slot], !overwriteAssignedSpeakerName,
           existing.name != nil
        {
            return existing
        }
        return timeline.upsertSpeaker(named: name, atIndex: slot)
    }

    // MARK: - Private: state

    private func resetStreamingState() {
        preprocessor?.reset()
        numFramesProcessed = 0
        finalized = false
        warmupFramesRemaining = model?.metadata.convDelay ?? 0
    }

    /// Drop leading warmup predictions in place. CNN right-lookahead means
    /// the first `convDelay` model outputs have no real audio behind them
    /// and must not reach the timeline — otherwise segment timestamps are
    /// shifted by `convDelay · frameDurationSeconds`. Safe to call on an
    /// empty or already-trimmed buffer.
    private func trimWarmup(_ preds: inout [Float], numSpeakers: Int) {
        guard warmupFramesRemaining > 0, !preds.isEmpty, numSpeakers > 0 else { return }
        let newFrames = preds.count / numSpeakers
        let drop = min(warmupFramesRemaining, newFrames)
        // `drop * numSpeakers ≤ newFrames * numSpeakers ≤ preds.count`
        // is guaranteed by the `min` above + integer-division rounding,
        // so no defensive clamp on `removeFirst` is needed.
        preds.removeFirst(drop * numSpeakers)
        warmupFramesRemaining -= drop
    }

    // MARK: - Private: finalize

    @discardableResult
    public func finalize() throws -> DiarizerTimelineUpdate? {
        guard !finalized else { return nil }
        
        guard let preprocessor, let model else {
            throw LSEENDError.notInitialized
        }
        
        // Drain pending real audio, capture real-frame target.
        try preprocessor.finalize()
        let update = try process()
        
        timeline.finalize()
        finalized = true
        return update
    }
}
