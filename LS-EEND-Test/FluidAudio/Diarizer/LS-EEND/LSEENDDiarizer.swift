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

public final class LSEENDDiarizer: Diarizer, @unchecked Sendable {

    // MARK: - Dependencies

    private let lsModel: LSEENDModel
    private let meta: LSEENDMetadata
    private let mel: AudioMelSpectrogram
    private let converter: AudioConverter

    public let timeline: DiarizerTimeline

    private var state: LSEENDState

    // MARK: - Protocol properties

    private var _isAvailable: Bool = true
    public var isAvailable: Bool { _isAvailable }
    public let targetSampleRate: Int?
    public let modelFrameHz: Double?
    public let numSpeakers: Int?
    public private(set) var numFramesProcessed: Int = 0

    // MARK: - Model-derived constants (cached for hot path)

    private let T: Int                 // chunk_size, frames per CoreML call
    private let featDim: Int           // (2*ctx+1) * nMels
    private let nMels: Int
    private let hopLength: Int
    private let nFft: Int
    private let halfNfft: Int
    private let contextSize: Int
    private let subsampling: Int
    private let convDelay: Int         // CNN look-ahead (warmup count)
    private let maxSpk: Int            // output speaker channels
    private let log10Scale: Float = 1.0 / Float(log(10.0))

    // MARK: - Streaming scratch (all reused; no hot-path allocation)

    /// Rolling audio buffer. Left-padded on cold start with halfNfft zeros so
    /// that audioRing[0] corresponds to the STFT start offset of stftFramesDone.
    private var audioRing: [Float] = []

    /// Head index into audioRing. We advance logically and compact lazily
    /// when audioHead exceeds a threshold.
    private var audioHead: Int = 0

    /// Global number of STFT frames already extracted.
    private var stftFramesDone: Int = 0

    /// CMN running sum + count.
    private var cmnSum: [Float]
    private var cmnCount: Int = 0
    private var cmnMean: [Float]       // scratch for current mean

    /// Flat log10-mel frames after CMN, [N * nMels].
    private var normalizedFrames: [Float] = []
    /// Global frame index of normalizedFrames[0].
    private var normalizedBase: Int = 0

    /// Flat subsampled+context feature frames, [N * featDim].
    private var subsampledFeatures: [Float] = []
    /// Global subsample index of subsampledFeatures[0].
    private var subsampledBase: Int = 0
    /// Next subsample index to emit.
    private var subsamplePointer: Int = 0
    /// Next subsample index to consume into a CoreML call.
    private var subsampleConsumed: Int = 0

    /// Partial-T accumulator — holds 0..T feature frames until T is reached.
    private var featBuffer: [Float]
    private var featBufferFill: Int = 0

    /// Total encoder output positions emitted across all CoreML calls.
    private var encOutputsEmitted: Int = 0
    private var finalized: Bool = false

    private var maskScratch: [Float]

    // MARK: - Init

    public init(model: LSEENDModel) throws {
        self.lsModel = model
        self.meta = model.metadata
        self.T = meta.chunkSize
        self.featDim = meta.featDim
        self.nMels = meta.nMels
        self.hopLength = meta.hopLength
        self.nFft = meta.nFFT
        self.halfNfft = meta.nFFT / 2
        self.contextSize = meta.contextSize
        self.subsampling = meta.subsampling
        self.convDelay = meta.convDelay
        self.maxSpk = meta.maxSpeakers

        self.mel = AudioMelSpectrogram(
            sampleRate: meta.sampleRate,
            nMels: meta.nMels,
            nFFT: meta.nFFT,
            hopLength: meta.hopLength,
            winLength: meta.winLength,
            preemph: 0,
            padTo: 0,
            logFloor: 1e-10,
            logFloorMode: .clamped,
            windowPeriodic: true  // librosa/scipy 'hann' with fftbins=True
        )
        self.converter = AudioConverter(sampleRate: Double(meta.sampleRate))

        self.state = try LSEENDState(from: meta)

        self.timeline = DiarizerTimeline(
            config: .default(
                numSpeakers: meta.maxSpeakers,
                frameDurationSeconds: meta.frameDurationSeconds
            )
        )

        self.targetSampleRate = meta.sampleRate
        self.modelFrameHz = Double(meta.sampleRate) / Double(meta.hopLength * meta.subsampling)
        self.numSpeakers = meta.maxSpeakers

        self.cmnSum = [Float](repeating: 0, count: nMels)
        self.cmnMean = [Float](repeating: 0, count: nMels)
        self.featBuffer = [Float](repeating: 0, count: T * featDim)
        self.maskScratch = [Float](repeating: 0, count: T)

        // Prime ring with left half-nFFT zeros so frame 0's STFT window starts
        // at audioRing[0]. Matches Python's center-padded STFT semantics.
        audioRing.reserveCapacity(halfNfft + meta.sampleRate) // ~1s headroom
        audioRing.append(contentsOf: repeatElement(0, count: halfNfft))
    }

    public static func loadFromHuggingFace(
        variant: LSEENDVariant = .dihard3,
        stepSize: LSEENDStepSize = .step100ms,
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuOnly,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> LSEENDDiarizer {
        let model = try await LSEENDModel.loadFromHuggingFace(
            variant: variant,
            stepSize: stepSize,
            cacheDirectory: cacheDirectory,
            computeUnits: computeUnits,
            progressHandler: progressHandler
        )
        return try LSEENDDiarizer(model: model)
    }

    // MARK: - Streaming API

    public func addAudio<C: Collection>(_ samples: C, sourceSampleRate: Double?) throws
    where C.Element == Float
    {
        try requireAvailable()
        guard !samples.isEmpty else { return }
        if let src = sourceSampleRate, Int(src.rounded()) != meta.sampleRate {
            let asArray = Array(samples)
            let resampled = try converter.resample(asArray, from: src)
            audioRing.reserveCapacity(audioRing.count + resampled.count)
            audioRing.append(contentsOf: resampled)
        } else {
            audioRing.reserveCapacity(audioRing.count + samples.count)
            audioRing.append(contentsOf: samples)
        }
    }

    public func process() throws -> DiarizerTimelineUpdate? {
        try requireAvailable()
        extractNewFeatures()
        return try runInference(finalizing: false, realTarget: nil)
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
        // Default heuristic: keep enrolled speakers only on a pristine
        // diarizer. `audioRing.count == halfNfft` and `stftFramesDone == 0`
        // together mean the ring still holds nothing but the left-pad
        // primed in init/reset (no samples yet consumed by STFT). If anyone
        // has fed audio, defaults to discarding old speaker bindings.
        let keep = keepSpeakers ?? (audioRing.count <= halfNfft && stftFramesDone == 0)
        resetStreamingState()
        timeline.reset(keepingSpeakers: keep)

        let resampled: [Float]
        if let src = sourceSampleRate, Int(src.rounded()) != meta.sampleRate {
            resampled = try converter.resample(Array(samples), from: src)
        } else {
            resampled = Array(samples)
        }

        // Match Python: single addAudio + single process() + finalize.
        // Progress is reported only at completion; mid-call progress would
        // require slicing (which breaks mic-style parity with Python).
        try addAudio(resampled, sourceSampleRate: nil)
        _ = try process()
        if finalizeOnCompletion {
            _ = try finalize()
        }
        progressCallback?(resampled.count, resampled.count, 1)
        return timeline
    }

    public func processComplete(
        audioFileURL: URL,
        keepingEnrolledSpeakers keepSpeakers: Bool?,
        finalizeOnCompletion: Bool,
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws -> DiarizerTimeline {
        let samples = try converter.resampleAudioFile(audioFileURL)
        return try processComplete(
            samples,
            sourceSampleRate: nil,
            keepingEnrolledSpeakers: keepSpeakers,
            finalizeOnCompletion: finalizeOnCompletion,
            progressCallback: progressCallback
        )
    }

    // MARK: - Lifecycle

    public func reset() {
        resetStreamingState()
        timeline.reset(keepingSpeakers: false)
    }

    public func cleanup() {
        resetStreamingState()
        _isAvailable = false
    }

    private func requireAvailable() throws {
        guard _isAvailable else {
            throw LSEENDError.inferenceFailed(
                "Diarizer cleaned up — cannot be used further."
            )
        }
    }

    #if DEBUG
    /// Debug-only: drive the feature extractor on the full audio buffer
    /// and return the flat `[N * featDim]` subsampled+context features
    /// that would be fed to CoreML. Used by parity tests.
    internal func debugExtractFeatures<C: Collection>(
        _ samples: C, sourceSampleRate: Double?
    ) throws -> [Float] where C.Element == Float {
        resetStreamingState()
        try addAudio(samples, sourceSampleRate: sourceSampleRate)
        extractNewFeatures()
        // Copy out only the emitted frames (subsamplePointer, offset from base 0).
        let count = (subsamplePointer - subsampledBase) * featDim
        return Array(subsampledFeatures[0..<count])
    }
    #endif

    public func enrollSpeaker<C: Collection>(
        withAudio samples: C,
        sourceSampleRate: Double?,
        named name: String?,
        overwritingAssignedSpeakerName overwriteAssignedSpeakerName: Bool
    ) throws -> DiarizerSpeaker? where C.Element == Float {
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
        let maxSpk = meta.maxSpeakers

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
        // Zero state tensors in place (preserves MLMultiArray identity).
        for arr in [state.encRetKv, state.encRetScale, state.encConvCache,
                    state.cnnWindow, state.decRetKv, state.decRetScale] {
            memset(arr.dataPointer, 0, arr.count * MemoryLayout<Float>.stride)
        }

        audioRing.removeAll(keepingCapacity: true)
        audioRing.append(contentsOf: repeatElement(0, count: halfNfft))
        audioHead = 0
        stftFramesDone = 0

        vDSP_vclr(&cmnSum, 1, vDSP_Length(nMels))
        cmnCount = 0

        normalizedFrames.removeAll(keepingCapacity: true)
        normalizedBase = 0

        subsampledFeatures.removeAll(keepingCapacity: true)
        subsampledBase = 0
        subsamplePointer = 0
        subsampleConsumed = 0

        featBufferFill = 0
        encOutputsEmitted = 0
        numFramesProcessed = 0
        finalized = false
    }

    // MARK: - Private: feature extraction

    /// Extract all STFT frames whose windows fit in the current audio buffer,
    /// update CMN, and emit subsample+context feature vectors.
    private func extractNewFeatures() {
        // 1. Count new STFT frames we can extract. Frame k starts at
        //    audioRing[audioHead + k*hop]; window is nFFT samples.
        let availableSamples = audioRing.count - audioHead
        let nNew = max(0, (availableSamples - nFft) / hopLength + 1)
        if nNew == 0 { return }

        // 2. Compute log-mel for those frames in one call.
        let slice = audioRing[audioHead..<audioRing.count]
        let result = mel.computeFlatTransposed(
            audio: slice,
            lastAudioSample: 0,
            paddingMode: .prePadded,
            expectedFrameCount: nNew
        )
        var melBlock = result.mel  // [nNew * nMels]

        // 3. Convert natural log → log10 in-place: multiply by 1/ln(10).
        var scale = log10Scale
        melBlock.withUnsafeMutableBufferPointer { buf in
            if let p = buf.baseAddress {
                vDSP_vsmul(p, 1, &scale, p, 1, vDSP_Length(nNew * nMels))
            }
        }

        // 4. Streaming CMN. Resize normalizedFrames ONCE to accommodate all
        //    new rows, then stride into the preallocated trailing region.
        //    Avoids per-frame append+resize heap traffic.
        let origCount = normalizedFrames.count
        normalizedFrames.append(
            contentsOf: repeatElement(0, count: nNew * nMels)
        )
        melBlock.withUnsafeBufferPointer { src in
            guard let srcBase = src.baseAddress else { return }
            normalizedFrames.withUnsafeMutableBufferPointer { dst in
                guard let dstBase = dst.baseAddress else { return }
                for i in 0..<nNew {
                    let framePtr = srcBase.advanced(by: i * nMels)

                    cmnSum.withUnsafeMutableBufferPointer { sumBuf in
                        vDSP_vadd(
                            sumBuf.baseAddress!, 1,
                            framePtr, 1,
                            sumBuf.baseAddress!, 1,
                            vDSP_Length(nMels)
                        )
                    }
                    cmnCount += 1
                    var invCount = 1.0 / Float(cmnCount)
                    vDSP_vsmul(
                        cmnSum, 1, &invCount,
                        &cmnMean, 1,
                        vDSP_Length(nMels)
                    )

                    vDSP_vsub(
                        cmnMean, 1,
                        framePtr, 1,
                        dstBase.advanced(by: origCount + i * nMels), 1,
                        vDSP_Length(nMels)
                    )
                }
            }
        }

        stftFramesDone += nNew

        // 5. Advance audioHead locally: next STFT frame starts nNew*hop
        //    samples after the current head within audioRing.
        audioHead += nNew * hopLength
        compactAudioRingIfNeeded()

        // 6. Emit subsampled+context features.
        emitSubsampledFeatures()
    }

    private func emitSubsampledFeatures() {
        let ctx = contextSize
        let featFloats = featDim
        let totalNormalized = normalizedBase + normalizedFrames.count / nMels

        while true {
            let k = subsamplePointer
            let rawIdx = k * subsampling
            let needIdx = rawIdx + ctx
            if needIdx >= totalNormalized { break }

            let startIdx = subsampledFeatures.count
            subsampledFeatures.append(contentsOf: repeatElement(0, count: featFloats))

            // Fill window [-ctx ... +ctx] from normalizedFrames; zeros when
            // out of range (cold-start left padding).
            normalizedFrames.withUnsafeBufferPointer { srcBuf in
                subsampledFeatures.withUnsafeMutableBufferPointer { dstBuf in
                    let dstBase = dstBuf.baseAddress!.advanced(by: startIdx)
                    let srcBase = srcBuf.baseAddress!
                    let byteStride = nMels * MemoryLayout<Float>.stride
                    for offset in -ctx...ctx {
                        let fi = rawIdx + offset
                        let outSlot = offset + ctx
                        let localFi = fi - normalizedBase
                        if localFi >= 0, localFi < normalizedFrames.count / nMels {
                            memcpy(
                                dstBase.advanced(by: outSlot * nMels),
                                srcBase.advanced(by: localFi * nMels),
                                byteStride
                            )
                        }
                        // else: already zero from the repeatElement append
                    }
                }
            }

            subsamplePointer += 1
        }

        // Trim normalizedFrames: oldest needed is rawIdx - ctx for next k.
        if subsamplePointer > 0 {
            let nextOldest = subsamplePointer * subsampling - ctx
            let trim = max(0, nextOldest - normalizedBase)
            // `<=` allows exact-fit trim; `<` left one stale row behind.
            if trim > 0 && trim * nMels <= normalizedFrames.count {
                normalizedFrames.removeFirst(trim * nMels)
                normalizedBase += trim
            }
        }
    }

    private func compactAudioRingIfNeeded() {
        // Keep ring compact; removeFirst is O(n) but bounded by audioHead.
        if audioHead > max(nFft, 4 * hopLength) {
            audioRing.removeFirst(audioHead)
            audioHead = 0
        }
    }

    // MARK: - Private: inference

    private func runInference(
        finalizing: Bool, realTarget: Int?
    ) throws -> DiarizerTimelineUpdate? {
        let target = realTarget ?? (finalizing ? subsamplePointer : Int.max)
        var newPreds: [Float] = []
        newPreds.reserveCapacity(T * maxSpk)

        // Drain ready subsampled features into featBuffer; fire when full.
        while subsampleConsumed < subsamplePointer {
            let srcFrame = subsampleConsumed - subsampledBase
            subsampledFeatures.withUnsafeBufferPointer { srcBuf in
                featBuffer.withUnsafeMutableBufferPointer { dstBuf in
                    _ = memcpy(
                        dstBuf.baseAddress!.advanced(by: featBufferFill * featDim),
                        srcBuf.baseAddress!.advanced(by: srcFrame * featDim),
                        featDim * MemoryLayout<Float>.stride
                    )
                }
            }
            featBufferFill += 1
            subsampleConsumed += 1

            if featBufferFill == T {
                let remaining = max(0, target - (numFramesProcessed + newPreds.count / maxSpk))
                let maxValid = finalizing ? remaining : nil
                let probs = try runOneCall(maxValid: maxValid)
                newPreds.append(contentsOf: probs)
                featBufferFill = 0
            }
        }

        // Trim consumed subsampled features.
        if subsampleConsumed > subsampledBase {
            let trim = subsampleConsumed - subsampledBase
            if trim * featDim <= subsampledFeatures.count {
                subsampledFeatures.removeFirst(trim * featDim)
                subsampledBase = subsampleConsumed
            }
        }

        // If finalizing, pad partial T with zeros until target is met.
        if finalizing {
            while numFramesProcessed + newPreds.count / maxSpk < target {
                featBuffer.withUnsafeMutableBufferPointer { buf in
                    while featBufferFill < T {
                        vDSP_vclr(
                            buf.baseAddress!.advanced(by: featBufferFill * featDim),
                            1, vDSP_Length(featDim)
                        )
                        featBufferFill += 1
                    }
                }
                let remaining = target - (numFramesProcessed + newPreds.count / maxSpk)
                let probs = try runOneCall(maxValid: remaining)
                newPreds.append(contentsOf: probs)
                featBufferFill = 0
            }
        }

        guard !newPreds.isEmpty else { return nil }
        numFramesProcessed += newPreds.count / maxSpk
        return try timeline.addPredictions(
            finalizedPredictions: newPreds,
            tentativePredictions: []
        )
    }

    private func runOneCall(maxValid: Int?) throws -> [Float] {
        let validStart = max(0, convDelay - encOutputsEmitted)
        var validEnd = T
        if let mv = maxValid {
            validEnd = min(T, validStart + max(0, mv))
        }

        // Build valid_mask: 0 elsewhere, 1 in [validStart, validEnd).
        maskScratch.withUnsafeMutableBufferPointer { buf in
            vDSP_vclr(buf.baseAddress!, 1, vDSP_Length(T))
            if validStart < validEnd {
                var one: Float = 1
                vDSP_vfill(
                    &one,
                    buf.baseAddress!.advanced(by: validStart), 1,
                    vDSP_Length(validEnd - validStart)
                )
            }
        }

        let probs = try lsModel.predict(
            state: &state,
            features: featBuffer,
            frameMask: maskScratch
        )
        encOutputsEmitted += T

        if validStart >= validEnd { return [] }
        // probs layout: [1, T, maxSpk]. Slice out valid rows.
        let startIdx = validStart * maxSpk
        let endIdx = validEnd * maxSpk
        return Array(probs[startIdx..<endIdx])
    }

    // MARK: - Private: finalize

    @discardableResult
    public func finalize() throws -> DiarizerTimelineUpdate? {
        guard !finalized else { return nil }
        // Drain pending real audio, capture real-frame target.
        extractNewFeatures()
        let realTarget = subsamplePointer

        // Append silence so the last `convDelay` real frames get right-context
        // through the CNN smoothing.
        let silenceLen =
            (convDelay * subsampling + contextSize) * hopLength + halfNfft
        let silence = [Float](repeating: 0, count: silenceLen)
        try addAudio(silence, sourceSampleRate: nil)
        extractNewFeatures()

        let update = try runInference(finalizing: true, realTarget: realTarget)
        timeline.finalize()
        finalized = true
        return update
    }
}
