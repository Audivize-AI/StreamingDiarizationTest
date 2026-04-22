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
    private var session: LSEENDFeatureProvider? = nil

    public var timeline: DiarizerTimeline

    // MARK: - Protocol properties

    public private(set) var isAvailable: Bool = false
    public private(set) var numFramesProcessed: Int = 0
    public let targetSampleRate: Int?
    public let modelFrameHz: Double?
    public let numSpeakers: Int?

    private var finalized: Bool = false

    private let logger = AppLogger(category: "LSEENDDiarizer")

    // MARK: - Init

    public init(model: LSEENDModel) throws {
        self.model = model
        let metadata = model.metadata
        self.session = try LSEENDFeatureProvider(from: metadata)

        self.timeline = DiarizerTimeline(
            config: .default(
                numSpeakers: metadata.maxSpeakers,
                frameDurationSeconds: metadata.frameDurationSeconds
            )
        )

        self.targetSampleRate = metadata.sampleRate
        self.modelFrameHz = Double(metadata.sampleRate) / Double(metadata.hopLength * metadata.subsampling)
        self.numSpeakers = metadata.maxSpeakers
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
        self.session = try LSEENDFeatureProvider(from: model.metadata)
        // Re-seed warmup counter + clear any prior streaming state — the
        // new model may have a different `convDelay`, so leaving stale
        // state around would mis-trim the first chunk after hot-swap.
        resetStreamingState()
    }

    // MARK: - Debug helpers (parity tests)

    #if DEBUG
    /// Drive `samples` through session → STFT → log10-mel → CMN →
    /// subsample+context stack, and return the flat `[N × featDim]`
    /// stacked features that would be fed to CoreML. Used by
    /// `testFeat345Parity` to byte-compare against the Python fixture
    /// without running inference.
    internal func debugExtractFeatures<C: Collection>(
        _ samples: C, sourceSampleRate: Double?
    ) throws -> [Float] where C.Element == Float {
        guard let session else { throw LSEENDError.notInitialized }
        session.reset()
        try session.enqueueAudio(samples, withSampleRate: sourceSampleRate)
        try session.drainRightContextWithSilence()

        var out: [Float] = []
        while let input = try session.emitNextChunk() {
            // `input.melFeatures` is preallocated + reused — copy out each
            // pass. Caller-allocated input arrays have tight strides, so a
            // flat read is safe (unlike model *output* arrays, which get
            // tile-padded strides; see CLAUDE.md gotcha #2).
            input.melFeatures.withUnsafeBufferPointer(ofType: Float.self) { buf in
                out.append(contentsOf: buf)
            }
        }
        return out
    }
    #endif

    // MARK: - Streaming API

    public func addAudio<C: Collection>(_ samples: C, sourceSampleRate: Double?) throws
    where C.Element == Float
    {
        guard !samples.isEmpty else { return }
        guard let session else {
            throw LSEENDError.notInitialized
        }
        try session.enqueueAudio(samples, withSampleRate: sourceSampleRate)
    }

    public func process() throws -> DiarizerTimelineUpdate? {
        try flush(progressCallback: nil)
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
        guard session != nil, model != nil else {
            throw LSEENDError.notInitialized
        }
        let keep = keepSpeakers ?? !timeline.hasSegments
        resetStreamingState()
        timeline.reset(keepingSpeakers: keep)

        try addAudio(samples, sourceSampleRate: sourceSampleRate)
        try flush(
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
        guard let session, model != nil else {
            throw LSEENDError.notInitialized
        }
        let keep = keepSpeakers ?? !timeline.hasSegments
        resetStreamingState()
        timeline.reset(keepingSpeakers: keep)

        try session.enqueueAudioFile(at: audioFileURL)
        try flush(
            finalizeOnCompletion: finalizeOnCompletion,
            progressCallback: progressCallback
        )
        return timeline
    }

    /// Shared drain path for both `processComplete` overloads. Runs
    /// session → model → timeline, optionally finalizing the stream.
    private func flush(
        recordFrames: Bool = true,
        finalizeOnCompletion: Bool,
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws {
        guard let session else {
            throw LSEENDError.notInitialized
        }

        if finalizeOnCompletion {
            try session.drainRightContextWithSilence()
        }

        _ = try flush(recordFrames: recordFrames, progressCallback: progressCallback)

        if finalizeOnCompletion {
            timeline.finalize()
            finalized = true
        }
    }

    /// Drain all ready chunks through `model.predict` → timeline. Returns the
    /// timeline update, or nil if the drain produced no frames. Warmup rows
    /// are stripped per-chunk inside `model.predict` (`input.warmupFrames`),
    /// so the accumulated stream is already 1:1 with real audio time.
    private func flush(
        recordFrames: Bool = true,
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws -> DiarizerTimelineUpdate? {
        guard let session, let model else {
            throw LSEENDError.notInitialized
        }

        let chunkSize = model.metadata.chunkSize
        let numSpeakers = model.metadata.maxSpeakers
        let rightContext = model.metadata.convDelay
        let totalChunks = session.readyChunks

        var processed = 0
        var newPreds: [Float] = []
        newPreds.reserveCapacity(totalChunks * numSpeakers * chunkSize)
        
        while let input = try session.emitNextChunk() {
            if recordFrames {
                input.warmupFrames = max(min(rightContext - numFramesProcessed, chunkSize), 0)
                numFramesProcessed += chunkSize
            }
            newPreds.append(contentsOf: try model.predict(from: input))
            processed += 1
            progressCallback?(processed, totalChunks, 1)
        }

        guard !newPreds.isEmpty else { return nil }

        return try timeline.addPredictions(
            finalizedPredictions: newPreds,
            tentativePredictions: []
        )
    }

    // MARK: - Lifecycle

    public func reset() {
        resetStreamingState()
        timeline.reset(keepingSpeakers: false)
    }

    public func cleanup() {
        resetStreamingState()
        self.model = nil
        self.session = nil
        isAvailable = false
    }

    public func enrollSpeaker<C: Collection>(
        withAudio samples: C,
        sourceSampleRate: Double?,
        named name: String?,
        overwritingAssignedSpeakerName overwriteAssignedSpeakerName: Bool
    ) throws -> DiarizerSpeaker? where C.Element == Float {
        guard let session else {
            throw LSEENDError.notInitialized
        }
        
        let sessionSnapshot = session.takeSnapshot()
        let timelineSnapshot = timeline.takeSnapshot()
        let isNamed = name != nil

        let requireNewSpeaker = isNamed && !overwriteAssignedSpeakerName
        
        if timeline.hasSegments {
            logger.warning("Enrolling speaker mid session. The timeline will be reset if successful.")
        }
        
        // Flush queued audio, including right context.
        try session.drainRightContextWithSilence()
        _ = try flush(progressCallback: nil)
        
        // Snapshot old speakers starting here after old audio has been flushed
        let oldSlots: Set<Int>
        
        if isNamed {
            oldSlots = Set(timeline.speakers.filter { $0.value.name != nil }.keys)
        } else {
            oldSlots = Set(timeline.speakers.keys)
        }
        
        try session.enqueueAudio(
            samples,
            withSampleRate: sourceSampleRate,
            eagerPreprocessing: false
        )
        
        // Flush enrollment audio queued in the right context
        try session.drainRightContextWithSilence()
        
        // Process enrollment audio. The new speaker will be extracted from this timeline update.
        guard let update = try flush(recordFrames: false, progressCallback: nil),
              !update.finalizedSegments.isEmpty
        else {
            session.rollback(to: sessionSnapshot)
            timeline.rollback(to: timelineSnapshot)
            return nil
        }
        
        // Get the new/unnamed speaker with the most speech if any exist.
        // Fallback to old speaker with the most speech if overwrites are allowed.
        var speechActivities: [Int : Float] = [:]
        for segment in update.finalizedSegments {
            speechActivities[segment.speakerIndex, default: 0] += segment.activity * Float(segment.length)
        }
        
        // Prioritized unnamed speakers; speech activity is secondary
        let bestSlot = speechActivities.max {
            let isFirstOld = oldSlots.contains($0.key)
            let isSecondOld = oldSlots.contains($1.key)
            if isFirstOld == isSecondOld {
                return $0.value < $1.value
            }
            return isFirstOld
        }?.key
        
        guard let bestSlot,
              let enrolledSpeaker = timeline.speakers[bestSlot],
              !requireNewSpeaker || !oldSlots.contains(bestSlot)
        else {
            session.rollback(to: sessionSnapshot)
            timeline.rollback(to: timelineSnapshot)
            return nil
        }
        
        // Rename speaker and report success
        enrolledSpeaker.name = name
        timeline.reset(keepingSpeakers: true)
        
        return enrolledSpeaker
    }

    // MARK: - Private: state

    private func resetStreamingState() {
        session?.reset()
        numFramesProcessed = 0
        finalized = false
    }

    // MARK: - Private: finalize

    @discardableResult
    public func finalize() throws -> DiarizerTimelineUpdate? {
        guard !finalized else { return nil }
        guard let session else {
            throw LSEENDError.notInitialized
        }

        // Drain pending real audio, capture real-frame target.
        try session.drainRightContextWithSilence()
        let update = try process()

        timeline.finalize()
        finalized = true
        return update
    }
}
