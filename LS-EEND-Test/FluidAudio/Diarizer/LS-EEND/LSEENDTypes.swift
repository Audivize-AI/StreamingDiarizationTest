//
//  LSEENDTypes.swift
//  LS-EEND-Test
//
//  Created by Benjamin Lee on 4/16/26.
//

import Foundation
import CoreML
import Accelerate


public typealias LSEENDVariant = ModelNames.LSEEND.Variant
public typealias LSEENDStepSize = ModelNames.LSEEND.StepSize


public struct LSEENDMetadata: Codable {
    public let chunkSize: Int
    public let frameDurationSeconds: Float
    public let maxSpeakers: Int
    public let sampleRate: Int
    public let maxNspks: Int
    public let hopLength: Int
    public let winLength: Int
    public let nMels: Int
    public let contextSize: Int
    public let subsampling: Int
    public let convDelay: Int
    public let nUnits: Int
    public let nHeads: Int
    public let encNLayers: Int
    public let decNLayers: Int
    public let convKernelSize: Int
    
    public var headDim: Int { nUnits / nHeads }
    public var featDim: Int { (2 * contextSize + 1) * nMels }
    public var nFFT: Int {
        1 << (Int.bitWidth - (winLength - 1).leadingZeroBitCount)
    }
}

public struct LSEENDState {
    public var encRetKv: MLMultiArray
    public var encRetScale: MLMultiArray
    public var encConvCache: MLMultiArray
    public var cnnWindow: MLMultiArray
    public var decRetKv: MLMultiArray
    public var decRetScale: MLMultiArray
    
    public init(
        encRetKv: MLMultiArray,
        encRetScale: MLMultiArray,
        encConvCache: MLMultiArray,
        cnnWindow: MLMultiArray,
        decRetKv: MLMultiArray,
        decRetScale: MLMultiArray,
    ) {
        self.encRetKv = encRetKv
        self.encRetScale = encRetScale
        self.encConvCache = encConvCache
        self.cnnWindow = cnnWindow
        self.decRetKv = decRetKv
        self.decRetScale = decRetScale
    }
    
    public init(from metadata: borrowing LSEENDMetadata) throws {
        let Lenc = NSNumber(value: metadata.encNLayers)
        let Ldec = NSNumber(value: metadata.decNLayers)
        let H = NSNumber(value: metadata.nHeads)
        let hd = NSNumber(value: metadata.headDim)
        let D = NSNumber(value: metadata.nUnits)
        let K = NSNumber(value: metadata.convKernelSize)
        let Kcnn = NSNumber(value: 2 * metadata.convDelay)
        let nSpk = NSNumber(value: metadata.maxNspks)
        
        self.encRetKv = try MLMultiArray(shape: [Lenc, 1, H, hd, hd], dataType: .float32)
        self.encRetScale = try MLMultiArray(shape: [Lenc, 1], dataType: .float32)
        self.encConvCache = try MLMultiArray(shape: [Lenc, 1, K, D], dataType: .float32)
        self.cnnWindow = try MLMultiArray(shape: [1, D, Kcnn], dataType: .float32)
        self.decRetKv = try MLMultiArray(shape: [Ldec, nSpk, H, hd, hd], dataType: .float32)
        self.decRetScale = try MLMultiArray(shape: [Ldec, 1], dataType: .float32)
        // `MLMultiArray(shape:dataType:)` does not zero its backing buffer
        // — on cold start the model would read garbage KV / conv caches and
        // emit NaN probs. Zero each buffer so the first chunk sees the
        // expected identity state.
        //
        // `memset` against `ma.count * Float.stride` is only correct when
        // the array has tight strides (physical layout = logical layout).
        // Caller-allocated `MLMultiArray(shape:dataType:)` normally does,
        // but model-output arrays get tile-padded strides (see CLAUDE.md
        // gotcha #2) — a padded array here would leave physical-padding
        // bytes uninitialized. The precondition catches that case loudly
        // instead of returning NaN probs later.
        for ma in [encRetKv, encRetScale, encConvCache, cnnWindow, decRetKv, decRetScale] {
            precondition(
                ma.strides.last?.intValue == 1,
                "state tensor has non-1 innermost stride; memset-zero would miss padding bytes"
            )
            memset(ma.dataPointer, 0, ma.count * MemoryLayout<Float>.stride)
        }
    }
    
    public func copy() -> LSEENDState {
        LSEENDState(
            encRetKv: encRetKv.copy() as! MLMultiArray,
            encRetScale: encRetScale.copy() as! MLMultiArray,
            encConvCache: encConvCache.copy() as! MLMultiArray,
            cnnWindow: cnnWindow.copy() as! MLMultiArray,
            decRetKv: decRetKv.copy() as! MLMultiArray,
            decRetScale: decRetScale.copy() as! MLMultiArray,
        )
    }
    
    public func copy(to dst: inout LSEENDState) {
        ANEMemoryUtils.strideAwareCopy(from: encRetKv, to: dst.encRetKv)
        ANEMemoryUtils.strideAwareCopy(from: encRetScale, to: dst.encRetScale)
        ANEMemoryUtils.strideAwareCopy(from: encConvCache, to: dst.encConvCache)
        ANEMemoryUtils.strideAwareCopy(from: cnnWindow, to: dst.cnnWindow)
        ANEMemoryUtils.strideAwareCopy(from: decRetKv, to: dst.decRetKv)
        ANEMemoryUtils.strideAwareCopy(from: decRetScale, to: dst.decRetScale)
    }
}

public enum LSEENDError: Error, LocalizedError {
    case initializationFailed(String)
    case inferenceFailed(String)
    case invalidInputSize(String)
    case notInitialized
}

public class LSEENDPreprocessor {
    /// Number of mel chunks currently ready for `emitNextChunk()`.
    public var readyChunks: Int { lock.withLock { melQueue.readyChunks } }

    private let melSpectrogram: AudioMelSpectrogram
    private let converter: AudioConverter
    private let input: LSEENDInput

    private var melQueue: SlidingWindowBuffer
    private var audioQueue: SlidingWindowBuffer

    private var cmnMean: [Float]
    private var cmnCount: Int

    private var framesProcessed: Int

    private let lock = NSLock()
    private let log10Scale: Float = 1.0 / log(10.0)
    private let decoderMask: [Float]

    /// Audio samples required past the last real sample to flush every
    /// buffered real frame through STFT + mel ±context + CNN right-lookahead.
    private let flushSampleCount: Int
    private let chunkFrames: Int
    private let chunkMels: Int
    private let nMels: Int
    private let contextSize: Int
    private let subsampling: Int
    private let featDim: Int

    public init(from metadata: borrowing LSEENDMetadata) throws {
        self.nMels = metadata.nMels
        self.contextSize = metadata.contextSize
        self.subsampling = metadata.subsampling
        self.featDim = metadata.featDim

        let contextMels = metadata.contextSize
        let contextSamples = metadata.nFFT / 2
        let chunkMels = metadata.subsampling * metadata.chunkSize
        let chunkSamples = metadata.hopLength * chunkMels

        // (mel ±context + CNN right-lookahead) mels × hop + STFT last-window halfNfft
        self.flushSampleCount =
            (contextMels + metadata.convDelay * metadata.subsampling) * metadata.hopLength
            + contextSamples

        self.melQueue = SlidingWindowBuffer(
            chunkLength: chunkMels,
            halfContextLength: contextMels,
            stride: nMels
        )
        self.audioQueue = SlidingWindowBuffer(
            chunkLength: chunkSamples,
            halfContextLength: contextSamples,
            stride: 1
        )
        
        self.cmnMean = .init(repeating: 0, count: nMels)
        self.cmnCount = 0
        
        self.framesProcessed = 0
        self.chunkFrames = metadata.chunkSize
        self.chunkMels = chunkMels
        self.decoderMask = (Array(repeating: 0, count: metadata.convDelay) +
                            Array(repeating: 1, count: metadata.chunkSize))
        
        
        // Initialize preprocessor and converter
        self.input = try LSEENDInput(from: metadata)
        
        self.melSpectrogram = AudioMelSpectrogram(
            sampleRate: metadata.sampleRate,
            nMels: metadata.nMels,
            nFFT: metadata.nFFT,
            hopLength: metadata.hopLength,
            winLength: metadata.winLength,
            preemph: 0,
            padTo: 0,
            logFloor: 1e-10,
            logFloorMode: .clamped,
            windowPeriodic: true
        )
        
        self.converter = AudioConverter(sampleRate: Double(metadata.sampleRate))
    }
    
    /// Clear preprocessor buffers + model recurrence state + frame counter.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        vDSP.fill(&cmnMean, with: 0)
        cmnCount = 0
        framesProcessed = 0
        audioQueue.reset()
        melQueue.reset()
        input.reset()
    }
    
    /// Add audio to the processing queue
    /// - Parameters:
    ///   - samples: Audio samples to enqueue
    ///   - sourceSampleRate: Sample rate of audio input
    ///   - eagerPreprocessing: Whether to eagerly feed audio chunks to the mel spectrogram
    public func enqueueAudio(
        _ samples: [Float],
        withSampleRate sourceSampleRate: Double? = nil,
        eagerPreprocessing: Bool = true
    ) throws {
        lock.lock()
        defer { lock.unlock() }
        
        if let sourceSampleRate {
            try audioQueue.append(converter.resample(samples, from: sourceSampleRate))
        } else {
            audioQueue.append(samples)
        }
        
        if eagerPreprocessing {
            flushAudioQueue()
        }
    }
    
    /// Resample and enqueue a full audio file.
    /// - Parameter url: Audio file to read.
    /// - Returns: Number of samples enqueued (at the model's sample rate).
    @discardableResult
    public func enqueueAudioFile(at url: URL) throws -> Int {
        let samples = try converter.resampleAudioFile(url)
        lock.lock()
        defer { lock.unlock() }
        audioQueue.append(samples)
        flushAudioQueue()
        return samples.count
    }

    /// Drain all buffered real audio through STFT + CMN → melQueue by
    /// appending enough trailing silence to satisfy STFT, mel ±context, and
    /// CNN right-lookahead. Caller then drains `emitNextChunk()` until nil.
    ///
    /// Call once per stream. Re-enqueuing audio after finalize requires
    /// `reset()` first.
    public func finalize() throws {
        lock.lock()
        defer { lock.unlock() }

        // 1. Trailing silence covering STFT + mel ±context + CNN right-lookahead.
        audioQueue.append([Float](repeating: 0, count: flushSampleCount))

        // 2. Round up to the next audio-chunk boundary so popAllChunks consumes
        //    every real sample plus the silence we just pushed.
        let unread = audioQueue.unreadSize
        let chunk  = audioQueue.chunkSize
        let ctx    = audioQueue.contextSize
        let overCtx = max(0, unread - ctx)
        let shortfall = (chunk - overCtx % chunk) % chunk
        if shortfall > 0 {
            audioQueue.append([Float](repeating: 0, count: shortfall))
        }

        // 3. Drain audioQueue → STFT → log10 → CMN → melQueue.
        flushAudioQueue()
    }
    
    /// Read the next chunk from the mel thingy
    public func emitNextChunk() throws -> LSEENDInput? {
        lock.lock()
        defer { lock.unlock() }

        flushAudioQueue()
        guard let rawChunk = melQueue.popNextChunk() else { return nil }

        defer {
            framesProcessed += chunkFrames
            framesProcessed = min(framesProcessed, decoderMask.count - chunkFrames)
        }

        // `rawChunk` layout: `(chunkMels + 2·contextSize) * nMels` floats —
        // 7 left-context + 10 advance + 7 right-context frames for dih3
        // (`chunkFrames=1, subsampling=10, contextSize=7`).
        // For each of the `chunkFrames` output frames k, stack the 15-frame
        // window `[k·subsampling … k·subsampling + 2·contextSize]`
        // (inclusive) → `featDim` floats. Left context of the popped chunk
        // lines up so output k=0 starts at raw-frame offset 0. Model input
        // expects `[1, chunkFrames, featDim]` = `chunkFrames·featDim` floats.
        var stacked = [Float](repeating: 0, count: chunkFrames * featDim)
        let windowMels = 2 * contextSize + 1
        rawChunk.withUnsafeBufferPointer { srcBuf in
            stacked.withUnsafeMutableBufferPointer { dstBuf in
                guard let src = srcBuf.baseAddress, let dst = dstBuf.baseAddress else { return }
                for k in 0..<chunkFrames {
                    let srcFrame = k * subsampling
                    memcpy(
                        dst.advanced(by: k * featDim),
                        src.advanced(by: srcFrame * nMels),
                        windowMels * nMels * MemoryLayout<Float>.stride
                    )
                }
            }
        }

        // loadInputs is generic on a single `C: AccelerateBuffer` — both
        // arguments must share the same concrete type. `decoderMask[...]`
        // is `ArraySlice<Float>`; slice `stacked` the same way.
        try input.loadInputs(
            melFeatures: stacked[...],
            decoderMask: decoderMask[framesProcessed..<framesProcessed + chunkFrames]
        )

        return input
    }
        
    private func flushAudioQueue() {
        // One audio `popNextChunk` → exactly `chunkMels` mel frames. Using
        // `popAllChunks` here is wrong: the entire pending audio comes back
        // as one slice, but `computeFlatTransposed(expectedFrameCount:)`
        // truncates the output to `chunkMels` frames and discards the rest.
        // Loop per-chunk so every enqueued chunk gets drained.
        while let audioChunk = audioQueue.popNextChunk() {
            // .prePadded emits (L - nFFT)/hop + 1 frames; our audio pop is
            // one hop wider than librosa needs, so pin the count to the
            // expected chunkMels per pop. Trailing samples reappear in the
            // next pop via the standard inter-chunk overlap.
            var (melFeats, melFrames, _) = melSpectrogram.computeFlatTransposed(
                audio: audioChunk,
                lastAudioSample: 0,
                paddingMode: .prePadded,
                expectedFrameCount: chunkMels
            )

            // Rescale to use log10 instead of ln
            var scale = log10Scale
            vDSP_vsmul(melFeats, 1, &scale, &melFeats, 1, vDSP_Length(melFrames * nMels))

            // Cumulative mean normalization
            melFeats.withUnsafeMutableBufferPointer { melFeatsBuffer in
                let melFrameLength = vDSP_Length(nMels)
                guard let melBase = melFeatsBuffer.baseAddress else { return }

                for melFrame in stride(from: melBase, to: melBase + melFeatsBuffer.count, by: nMels) {
                    // µ[k+1] = µ[k] + (mel[k+1] - µ[k]) * 1/(k+1)
                    cmnCount += 1
                    var alpha = 1.0 / Float(cmnCount)
                    vDSP_vintb(cmnMean, 1, melFrame, 1, &alpha, &cmnMean, 1, melFrameLength)
                    // mel[k+1] <- mel[k+1] - µ[k+1].
                    // vDSP_vsub(A,_,B,_,C,_,N) computes C = B - A, so A is
                    // `cmnMean` and B is `melFrame` for a `mel - µ` result.
                    vDSP_vsub(cmnMean, 1, melFrame, 1, melFrame, 1, melFrameLength)
                }
            }

            melQueue.append(consume melFeats)
        }
    }
}

public class LSEENDInput: MLFeatureProvider {
    var state: LSEENDState
    let melFeatures: MLMultiArray
    let decoderMask: MLMultiArray
    private let metadata: LSEENDMetadata

    public var featureNames: Set<String> {[
        "features",
        "enc_kv", "enc_scale",
        "enc_conv_cache", "cnn_window",
        "dec_kv", "dec_scale",
        "valid_mask"
    ]}

    public init(from metadata: LSEENDMetadata) throws {
        self.metadata = metadata
        self.state = try .init(from: metadata)
        let T = NSNumber(value: metadata.chunkSize)
        let F = NSNumber(value: metadata.featDim)
        self.melFeatures = try MLMultiArray(shape: [1, T, F], dataType: .float32)
        self.decoderMask = try MLMultiArray(shape: [T], dataType: .float32)
    }

    /// Reset state + input buffers for a fresh stream. State is re-allocated
    /// (rather than memset) because after the first `predict()` the state
    /// arrays are CoreML-vended and may have non-contiguous strides —
    /// a flat memset would only zero a prefix.
    /// `try!` is acceptable here: shapes are fixed from metadata, so failure
    /// can only come from OOM, which is not recoverable.
    public func reset() {
        state = try! LSEENDState(from: metadata)
        memset(melFeatures.dataPointer, 0, melFeatures.count * MemoryLayout<Float>.stride)
        memset(decoderMask.dataPointer, 0, decoderMask.count * MemoryLayout<Float>.stride)
    }
    
    public func loadInputs<C: AccelerateBuffer>(
        melFeatures newMelFeatures: C,
        decoderMask newDecoderMask: C
    ) throws where C.Element == Float {
        try loadMelFeatures(from: newMelFeatures)
        try loadDecoderMask(from: newDecoderMask)
    }
    
    public func loadDecoderMask<C: AccelerateBuffer>(from newDecoderMask: C) throws
    where C.Element == Float {
        guard newDecoderMask.count == decoderMask.count else {
            throw LSEENDError.invalidInputSize(
                "decoder mask input size mismatch: new=\(newDecoderMask.count) expected=\(decoderMask.count)")
        }
        
        _ = newDecoderMask.withUnsafeBufferPointer { maskPtr in
            memcpy(decoderMask.dataPointer, maskPtr.baseAddress,
                   maskPtr.count * MemoryLayout<Float>.stride)
        }
    }
    
    public func loadMelFeatures<C: AccelerateBuffer>(from newMelFeatures: C) throws
    where C.Element == Float {
        guard newMelFeatures.count == melFeatures.count else {
            throw LSEENDError.invalidInputSize(
                "mel features input size mismatch: new=\(newMelFeatures.count) expected=\(melFeatures.count)")
        }
        
        _ = newMelFeatures.withUnsafeBufferPointer { melPtr in
            memcpy(melFeatures.dataPointer, melPtr.baseAddress,
                   melPtr.count * MemoryLayout<Float>.stride)
        }
    }
    
    public func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "features": return MLFeatureValue(multiArray: melFeatures)
        case "enc_kv": return MLFeatureValue(multiArray: state.encRetKv)
        case "enc_scale": return MLFeatureValue(multiArray: state.encRetScale)
        case "enc_conv_cache": return MLFeatureValue(multiArray: state.encConvCache)
        case "cnn_window": return MLFeatureValue(multiArray: state.cnnWindow)
        case "dec_kv": return MLFeatureValue(multiArray: state.decRetKv)
        case "dec_scale": return MLFeatureValue(multiArray: state.decRetScale)
        case "valid_mask": return MLFeatureValue(multiArray: decoderMask)
        default: return nil
        }
    }
}



struct SlidingWindowBuffer {
    /// Stride between elements if features are n-dimensional arrays
    let stride: Int
    
    /// Context size
    let contextSize: Int
    
    /// Unpadded chunk size
    let chunkSize: Int
    
    /// Padded chunk size
    let paddedChunkSize: Int
    
    /// Number of unread floats
    public var unreadSize: Int { buffer.count - head }
    
    /// Number of full chunks currently poppable via `popNextChunk` / `popAllChunks`.
    public var readyChunks: Int { max(0, (unreadSize - contextSize) / chunkSize) }
    
    /// Start index/index offset
    private var offset: Int
    
    /// Next index at which to start processing
    private var head: Int
    
    /// Data buffer
    private var buffer: [Float] = []
    
    public var hasChunk: Bool {
        buffer.count - head >= paddedChunkSize
    }
    
    public init(chunkLength: Int, halfContextLength: Int, stride: Int) {
        self.stride = stride
        self.chunkSize = chunkLength * stride
        self.contextSize = 2 * halfContextLength * stride
        self.paddedChunkSize = chunkSize + contextSize
        
        self.head = 0
        self.offset = 0
        
        self.buffer.reserveCapacity(paddedChunkSize * 2)
        self.buffer.append(contentsOf: repeatElement(0, count: contextSize / 2))
    }
    
    public mutating func append(_ newElements: [Float]) {
        // Lazy trimming
        if buffer.count + newElements.count > buffer.capacity {
            buffer.removeFirst(head)
            offset += head
            head = 0
        }
        
        // Allow Swift to reserve more memory if needed after the trimming
        buffer.append(contentsOf: newElements)
    }
    
    /// Pop the last chunk
    public mutating func popNextChunk() -> ArraySlice<Float>? {
        guard hasChunk else { return nil }
        let result = buffer[head..<head+paddedChunkSize]
        head += chunkSize
        return result
    }
    
    /// Pop all availables chunk as one buffer
    public mutating func popAllChunks() -> ArraySlice<Float>? {
        guard hasChunk else { return nil }
        let newHead = head + (buffer.count - head - contextSize) / chunkSize * chunkSize
        let result = buffer[head..<newHead+contextSize]
        head = newHead
        return result
    }
    
    /// Reset buffer
    public mutating func reset() {
        self.head = 0
        self.offset = 0
        self.buffer.removeAll(keepingCapacity: true)
        self.buffer.append(contentsOf: repeatElement(0, count: contextSize / 2))
    }
}
