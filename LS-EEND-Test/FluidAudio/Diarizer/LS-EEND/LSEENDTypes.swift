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
    
    public var melFrames: Int { (chunkSize - 1) * subsampling + 2 * contextSize + 1 }
    
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
        decRetScale: MLMultiArray
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
        
        self.encRetKv = try ANEMemoryUtils.createAlignedArray(
            shape: [Lenc, 1, H, hd, hd], dataType: .float32)
        self.encRetScale = try ANEMemoryUtils.createAlignedArray(
            shape: [Lenc, 1], dataType: .float32)
        self.encConvCache = try ANEMemoryUtils.createAlignedArray(
            shape: [Lenc, 1, K, D], dataType: .float32)
        self.cnnWindow = try ANEMemoryUtils.createAlignedArray(
            shape: [1, D, Kcnn], dataType: .float32)
        self.decRetKv = try ANEMemoryUtils.createAlignedArray(
            shape: [Ldec, nSpk, H, hd, hd], dataType: .float32)
        self.decRetScale = try ANEMemoryUtils.createAlignedArray(
            shape: [Ldec, 1], dataType: .float32)
    }
    
    public func copy() -> LSEENDState {
        LSEENDState(
            encRetKv: encRetKv.copy() as! MLMultiArray,
            encRetScale: encRetScale.copy() as! MLMultiArray,
            encConvCache: encConvCache.copy() as! MLMultiArray,
            cnnWindow: cnnWindow.copy() as! MLMultiArray,
            decRetKv: decRetKv.copy() as! MLMultiArray,
            decRetScale: decRetScale.copy() as! MLMultiArray
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
    
    public func reset() {
        clearMultiArray(encRetKv)
        clearMultiArray(encRetScale)
        clearMultiArray(encConvCache)
        clearMultiArray(cnnWindow)
        clearMultiArray(decRetKv)
        clearMultiArray(decRetScale)
    }
    
}

public enum LSEENDError: Error, LocalizedError {
    case initializationFailed(String)
    case inferenceFailed(String)
    case invalidInputSize(String)
    case notInitialized
}

public class LSEENDSession {
    public struct Snapshot {
        let state: LSEENDState
        let melQueue: SlidingWindowBuffer
        let audioQueue: SlidingWindowBuffer
        let cmnMean: [Float]
        let cmnCount: Int
        let decoderMaskEnd: Int
    }
    
    /// Number of mel chunks currently ready for `emitNextChunk()`.
    public var readyChunks: Int { lock.withLock { melQueue.readyChunks } }

    private let melSpectrogram: AudioMelSpectrogram
    private let converter: AudioConverter
    private let input: LSEENDInput

    private var melQueue: SlidingWindowBuffer
    private var audioQueue: SlidingWindowBuffer

    private var cmnMean: [Float]
    private var cmnCount: Int
    
    private var isRightContextEmpty: Bool = true

    private var decoderMaskEnd: Int

    private let lock = NSLock()
    private let log10Scale: Float = 1.0 / log(10.0)
    private let decoderMask: [Float]

    /// Audio samples required past the last real sample to flush every
    /// buffered real frame through STFT + mel ±context + CNN right-lookahead.
    private let flushSampleCount: Int
    private let chunkFrames: Int
    private let nMels: Int

    public init(from metadata: borrowing LSEENDMetadata, restoringFrom snapshot: consuming Snapshot? = nil) throws {
        self.nMels = metadata.nMels

        let contextMels = metadata.contextSize
        let contextSamples = metadata.nFFT / 2
        let chunkMels = metadata.subsampling * metadata.chunkSize
        let chunkSamples = metadata.hopLength * chunkMels
        
        // TODO: Validate that this can't be reduced further
        let rightSamples = metadata.nFFT / 2 - metadata.hopLength

        // (mel ±context + CNN right-lookahead) mels × hop + STFT last-window halfNfft
        self.flushSampleCount =
            (contextMels + metadata.convDelay * metadata.subsampling) * metadata.hopLength
            + contextSamples
        
        self.chunkFrames = metadata.chunkSize
        
        var decoderMaskTemp = Array<Float>(repeating: 1, count: metadata.convDelay + metadata.chunkSize)
        vDSP_vclr(&decoderMaskTemp, 1, vDSP_Length(metadata.convDelay))
        self.decoderMask = decoderMaskTemp
        
        // Initialize processors
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
        
        // Initialize state
        if let snapshot {
            self.melQueue = snapshot.melQueue
            self.audioQueue = snapshot.audioQueue
            self.cmnMean = snapshot.cmnMean
            self.cmnCount = snapshot.cmnCount
            self.decoderMaskEnd = snapshot.decoderMaskEnd
        } else {
            self.melQueue = SlidingWindowBuffer(
                chunkLength: chunkMels,
                leftContextLength: contextMels,
                rightContextLength: contextMels + 1 - metadata.subsampling,
                stride: nMels
            )
            self.audioQueue = SlidingWindowBuffer(
                chunkLength: chunkSamples,
                leftContextLength: contextSamples,
                rightContextLength: rightSamples,
                stride: 1
            )
            
            self.cmnMean = .init(repeating: 0, count: nMels)
            self.cmnCount = 0
            self.decoderMaskEnd = 0
            
            // Initialize preprocessor and converter
            self.input = try LSEENDInput(from: metadata)
        }
    }
    
    /// Clear preprocessor buffers + model recurrence state + frame counter.
    public func reset() {
        lock.lock()
        defer { lock.unlock() }
        vDSP.fill(&cmnMean, with: 0)
        cmnCount = 0
        decoderMaskEnd = 0
        audioQueue.reset()
        melQueue.reset()
        input.reset()
    }
    
    /// Add audio to the processing queue
    /// - Parameters:
    ///   - samples: Audio samples to enqueue
    ///   - sourceSampleRate: Sample rate of audio input
    ///   - eagerPreprocessing: Whether to eagerly feed audio chunks to the mel spectrogram
    public func enqueueAudio<C: Collection>(
        _ samples: C,
        withSampleRate sourceSampleRate: Double? = nil,
        eagerPreprocessing: Bool = true
    ) throws where C.Element == Float {
        lock.lock()
        defer { lock.unlock() }

        if let sourceSampleRate {
            // `converter.resample` requires `[Float]`; unavoidable copy on
            // this branch. The no-resample branch stays copy-free via
            // `audioQueue.append`'s own `<C: Collection>` generic.
            let array = (samples as? [Float]) ?? Array(samples)
            try audioQueue.append(converter.resample(array, from: sourceSampleRate))
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
    public func finalizeQueuedAudio(flush: Bool = true) throws {
        lock.lock()
        defer { lock.unlock() }

        // 1. Trailing silence covering STFT + mel ±context + CNN right-lookahead.
        audioQueue.append(repeatElement(0, count: flushSampleCount))

        // 2. Round up to the next audio-chunk boundary so popAllChunks consumes
        //    every real sample plus the silence we just pushed.
        let unread = audioQueue.unreadFloats
        let chunk  = audioQueue.chunkFloats
        let ctx    = audioQueue.contextFloats
        let overCtx = max(0, unread - ctx)
        let shortfall = (chunk - overCtx % chunk) % chunk
        if shortfall > 0 {
            audioQueue.append(repeatElement(0, count: shortfall))
        }

        // 3. Drain audioQueue → STFT → log10 → CMN → melQueue.
        if flush {
            flushAudioQueue()            
        }
    }
    
    /// Read the next chunk from the mel
    public func emitNextChunk() throws -> LSEENDInput? {
        lock.lock()
        defer { lock.unlock() }

        flushAudioQueue()
        guard let rawChunk = melQueue.popNextChunk() else { return nil }
        
        // Advance decoder mask
        decoderMaskEnd = min(decoderMaskEnd + chunkFrames, decoderMask.count)

        try input.loadInputs(
            melFeatures: rawChunk,
            decoderMask: decoderMask[decoderMaskEnd-chunkFrames..<decoderMaskEnd],
            warmupFrames: min(decoderMask.count - decoderMaskEnd, chunkFrames)
        )

        return input
    }
    
    public func takeSnapshot() -> Snapshot {
        return Snapshot(
            state: input.state.copy(),
            melQueue: melQueue,
            audioQueue: audioQueue,
            cmnMean: cmnMean,
            cmnCount: cmnCount,
            decoderMaskEnd: decoderMaskEnd
        )
    }
    
    public func rollback(to snapshot: consuming Snapshot, keepingState: Bool = false)  {
        if !keepingState { self.input.state = snapshot.state }
        self.melQueue = snapshot.melQueue
        self.audioQueue = snapshot.audioQueue
        self.cmnMean = snapshot.cmnMean
        self.cmnCount = snapshot.cmnCount
        self.decoderMaskEnd = snapshot.decoderMaskEnd
    }
    
    private func flushAudioQueue() {
        guard let audioChunk = audioQueue.popAllChunks() else { return }

        var (melFeats, melFrames, _) = melSpectrogram.computeFlatTransposed(
            audio: audioChunk,
            lastAudioSample: 0,
            paddingMode: .prePadded,
            expectedFrameCount: nil
        )

        // Rescale to use log10 instead of ln
        var scale = log10Scale
        vDSP_vsmul(melFeats, 1, &scale, &melFeats, 1, vDSP_Length(melFrames * nMels))

        // Cumulative mean normalization — sequential by definition.
        melFeats.withUnsafeMutableBufferPointer { melFeatsBuffer in
            let melFrameLength = vDSP_Length(nMels)
            guard let melBase = melFeatsBuffer.baseAddress else { return }

            for melFrame in stride(from: melBase, to: melBase + melFeatsBuffer.count, by: nMels) {
                // µ[k] = µ[k-1] + (mel[k] - µ[k-1]) * 1 / k
                cmnCount += 1
                var alpha = 1.0 / Float(cmnCount)
                vDSP_vintb(cmnMean, 1, melFrame, 1, &alpha, &cmnMean, 1, melFrameLength)
                // mel[k] <- mel[k] - µ[k]. vDSP_vsub(A,_,B,_,C,_,N) is C = B - A.
                vDSP_vsub(cmnMean, 1, melFrame, 1, melFrame, 1, melFrameLength)
            }
        }

        melQueue.append(consume melFeats)
    }
}

public class LSEENDInput: MLFeatureProvider {
    var state: LSEENDState
    let melFeatures: MLMultiArray
    let decoderMask: MLMultiArray
    var warmupFrames: Int = 0

    public var featureNames: Set<String> {[
        "features",
        "enc_kv", "enc_scale",
        "enc_conv_cache", "cnn_window",
        "dec_kv", "dec_scale",
        "valid_mask"
    ]}

    public init(from metadata: LSEENDMetadata) throws {
        self.state = try LSEENDState(from: metadata)
        let T = NSNumber(value: metadata.chunkSize)
        let M = NSNumber(value: metadata.melFrames)
        let N = NSNumber(value: metadata.nMels)
        self.melFeatures = try MLMultiArray(shape: [1, M, N], dataType: .float32)
        self.decoderMask = try MLMultiArray(shape: [T], dataType: .float32)
    }

    /// Reset state + input buffers for a fresh stream.
    public func reset() {
        state.reset()
        clearMultiArray(melFeatures)
        clearMultiArray(decoderMask)
    }
    
    @inline(__always)
    public func loadInputs<C: AccelerateBuffer>(
        melFeatures newMelFeatures: C,
        decoderMask newDecoderMask: C,
        warmupFrames: Int? = nil
    ) throws where C.Element == Float {
        try Self.load(decoderMask, from: newDecoderMask)
        try Self.load(melFeatures, from: newMelFeatures)
        self.warmupFrames = warmupFrames ??
            newDecoderMask.withUnsafeBufferPointer { $0.count(where: \.isZero) }
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
    
    @inline(__always)
    private static func load<C: AccelerateBuffer>(
        _ multiArray: MLMultiArray,
        from buffer: C,
    ) throws {
        guard buffer.count == multiArray.count else {
            throw LSEENDError.invalidInputSize(
                "Input size mismatch: new=\(buffer.count) expected=\(multiArray.count)")
        }
        
        _ = buffer.withUnsafeBufferPointer { buf in
            memcpy(multiArray.dataPointer, buf.baseAddress,
                   buf.count * MemoryLayout<Float>.stride)
        }
    }
}



struct SlidingWindowBuffer {
    /// Stride between elements if features are n-dimensional arrays
    let stride: Int

    /// Total context size in floats (`leftContextFloats + rightContextFloats`).
    /// Kept as a single value so `popAllChunks` / `readyChunks` arithmetic
    /// (`unreadSize - contextSize`) stays correct under the asymmetric split.
    let contextFloats: Int

    /// Unpadded chunk size
    let chunkFloats: Int

    /// Padded chunk size — width of a `popNextChunk` / `popAllChunks` slice.
    let paddedChunkFloats: Int
    
    /// Whether the buffer is empty
    var isEmpty: Bool { buffer.isEmpty }

    /// Pre-pad width in floats — how many leading zeros are seeded at init
    /// and restored by `reset()`. Under the asymmetric left/right split this
    /// equals `leftContextFloats`; `contextSize / 2` would be wrong when
    /// left ≠ right.
    private let leftContextFloats: Int

    /// Number of unread floats
    public var unreadFloats: Int { buffer.count - head }

    /// Number of full chunks currently poppable via `popNextChunk` / `popAllChunks`.
    public var readyChunks: Int { max(0, (unreadFloats - contextFloats) / chunkFloats) }

    /// Next index at which to start processing
    private var head: Int

    /// Data buffer
    private var buffer: [Float] = []

    /// Whether a chunk is ready
    public var hasChunk: Bool {
        buffer.count - head >= paddedChunkFloats
    }

    /// Asymmetric left/right context. `rightContextLength` may be negative,
    /// in which case `head` advances past the popped slice by
    /// `-rightContextLength` strides each pop — useful when the consumer's
    /// per-window read is shorter than the advance block. Caller must ensure
    /// `leftContextLength + rightContextLength >= 0` so `contextFloats`
    /// (used by `readyChunks` / `popAllChunks`) stays non-negative.
    public init(
        chunkLength: Int,
        leftContextLength: Int,
        rightContextLength: Int,
        stride: Int
    ) {
        self.stride = stride
        self.chunkFloats = chunkLength * stride
        self.leftContextFloats = leftContextLength * stride
        self.contextFloats = (leftContextLength + rightContextLength) * stride
        self.paddedChunkFloats = chunkFloats + contextFloats

        self.head = 0

        self.buffer.reserveCapacity(paddedChunkFloats * 2)
        self.buffer.append(contentsOf: repeatElement(0, count: leftContextFloats))
    }

    public mutating func append<C: Collection>(_ newElements: C)
    where C.Element == Float {
        // Lazy trimming
        if buffer.count + newElements.count > buffer.capacity {
            buffer.removeFirst(head)
            head = 0
        }

        // Allow Swift to reserve more memory if needed after the trimming
        buffer.append(contentsOf: newElements)
    }

    /// Pop the last chunk
    public mutating func popNextChunk() -> ArraySlice<Float>? {
        guard hasChunk else { return nil }
        let result = buffer[head..<head+paddedChunkFloats]
        head += chunkFloats
        return result
    }

    /// Pop all available chunks as one buffer
    public mutating func popAllChunks() -> ArraySlice<Float>? {
        guard hasChunk else { return nil }
        let newHead = head + (buffer.count - head - contextFloats) / chunkFloats * chunkFloats
        let result = buffer[head..<newHead+contextFloats]
        head = newHead
        return result
    }

    /// Reset buffer
    public mutating func reset() {
        self.head = 0
        self.buffer.removeAll(keepingCapacity: true)
        self.buffer.append(contentsOf: repeatElement(0, count: leftContextFloats))
    }
}


@inline(__always)
private func clearMultiArray(_ buffer: MLMultiArray) {
    buffer.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, strides in
        guard let base = buf.baseAddress else { return }
        vDSP_vclr(base, 1, vDSP_Length(buf.count))
    }
}
