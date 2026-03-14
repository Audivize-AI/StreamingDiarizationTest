import Foundation

// MARK: - Diarizer Protocol

/// Protocol for frame-based speaker diarization processors.
///
/// Both SortformerDiarizer and LS-EEND processors conform to this protocol,
/// providing a unified streaming and offline diarization API.
public protocol Diarizer: AnyObject {
    /// Accumulated diarization results
    var timeline: DiarizerTimeline { get }

    /// Whether the processor is initialized and ready
    var isAvailable: Bool { get }

    /// Number of confirmed frames processed so far
    var numFramesProcessed: Int { get }

    /// Model's target sample rate in Hz
    var targetSampleRate: Int? { get }

    /// Output frame rate in Hz
    var modelFrameHz: Double? { get }

    /// Number of real speaker output tracks
    var numSpeakers: Int? { get }

    // MARK: Streaming

    /// Add audio samples to the processing buffer
    func addAudio(_ samples: [Float])

    /// Process buffered audio and return any new results
    func process() throws -> DiarizerChunkResult?

    /// Add audio and process in one call
    func processSamples(_ samples: [Float]) throws -> DiarizerChunkResult?

    // MARK: Offline

    /// Process complete audio and return finalized timeline
    func processComplete(
        _ samples: [Float],
        progressCallback: ((Int, Int, Int) -> Void)?
    ) throws -> DiarizerTimeline

    // MARK: Lifecycle

    /// Reset streaming state while keeping model loaded
    func reset()

    /// Clean up all resources
    func cleanup()
}

// MARK: - Post-Processing Configuration

/// Configuration for post-processing diarizer predictions into segments.
///
/// Generalizes Sortformer's `SortformerPostProcessingConfig` for any frame-based
/// diarizer (Sortformer, LS-EEND, etc.).
public struct DiarizerPostProcessingConfig: Sendable {
    /// Number of speaker output tracks
    public let numSpeakers: Int

    /// Duration of one output frame in seconds
    public let frameDurationSeconds: Float

    /// Onset threshold for detecting the beginning of speech
    public var onsetThreshold: Float

    /// Offset threshold for detecting the end of speech
    public var offsetThreshold: Float

    /// Padding frames added before each speech segment
    public var onsetPadFrames: Int

    /// Padding frames added after each speech segment
    public var offsetPadFrames: Int

    /// Minimum segment length in frames (shorter segments are discarded)
    public var minFramesOn: Int

    /// Minimum gap length in frames (shorter gaps are closed)
    public var minFramesOff: Int

    /// Maximum number of finalized prediction frames to retain (nil = unlimited)
    public var maxStoredFrames: Int?

    // MARK: - Seconds Accessors

    public var onsetPadSeconds: Float {
        get { Float(onsetPadFrames) * frameDurationSeconds }
        set { onsetPadFrames = Int(round(newValue / frameDurationSeconds)) }
    }

    public var offsetPadSeconds: Float {
        get { Float(offsetPadFrames) * frameDurationSeconds }
        set { offsetPadFrames = Int(round(newValue / frameDurationSeconds)) }
    }

    public var minDurationOn: Float {
        get { Float(minFramesOn) * frameDurationSeconds }
        set { minFramesOn = Int(round(newValue / frameDurationSeconds)) }
    }

    public var minDurationOff: Float {
        get { Float(minFramesOff) * frameDurationSeconds }
        set { minFramesOff = Int(round(newValue / frameDurationSeconds)) }
    }

    // MARK: - Presets

    /// Default configuration with no post-processing (pass-through thresholding at 0.5)
    public static func `default`(numSpeakers: Int, frameDurationSeconds: Float) -> DiarizerPostProcessingConfig {
        DiarizerPostProcessingConfig(
            numSpeakers: numSpeakers,
            frameDurationSeconds: frameDurationSeconds,
            onsetThreshold: 0.5,
            offsetThreshold: 0.5,
            onsetPadFrames: 0,
            offsetPadFrames: 0,
            minFramesOn: 0,
            minFramesOff: 0
        )
    }

    // MARK: - Init

    public init(
        numSpeakers: Int,
        frameDurationSeconds: Float,
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.5,
        onsetPadFrames: Int = 0,
        offsetPadFrames: Int = 0,
        minFramesOn: Int = 0,
        minFramesOff: Int = 0,
        maxStoredFrames: Int? = nil
    ) {
        self.numSpeakers = numSpeakers
        self.frameDurationSeconds = frameDurationSeconds
        self.onsetThreshold = onsetThreshold
        self.offsetThreshold = offsetThreshold
        self.onsetPadFrames = onsetPadFrames
        self.offsetPadFrames = offsetPadFrames
        self.minFramesOn = minFramesOn
        self.minFramesOff = minFramesOff
        self.maxStoredFrames = maxStoredFrames
    }

    public init(
        numSpeakers: Int,
        frameDurationSeconds: Float,
        onsetThreshold: Float = 0.5,
        offsetThreshold: Float = 0.5,
        onsetPadSeconds: Float = 0,
        offsetPadSeconds: Float = 0,
        minDurationOn: Float = 0,
        minDurationOff: Float = 0,
        maxStoredFrames: Int? = nil
    ) {
        self.numSpeakers = numSpeakers
        self.frameDurationSeconds = frameDurationSeconds
        self.onsetThreshold = onsetThreshold
        self.offsetThreshold = offsetThreshold
        self.onsetPadFrames = Int(round(onsetPadSeconds / frameDurationSeconds))
        self.offsetPadFrames = Int(round(offsetPadSeconds / frameDurationSeconds))
        self.minFramesOn = Int(round(minDurationOn / frameDurationSeconds))
        self.minFramesOff = Int(round(minDurationOff / frameDurationSeconds))
        self.maxStoredFrames = maxStoredFrames
    }
}

// MARK: - Segment

/// A single speaker segment from any diarizer.
public struct DiarizerSegment: Sendable, Identifiable {
    public let id: UUID

    /// Speaker index in diarizer output
    public var speakerIndex: Int

    /// Index of segment start frame
    public var startFrame: Int

    /// Index of segment end frame
    public var endFrame: Int

    /// Length of the segment in frames
    public var length: Int { endFrame - startFrame }

    /// Whether this segment is finalized
    public var isFinalized: Bool

    /// Duration of one frame in seconds
    public let frameDurationSeconds: Float

    /// Start time in seconds
    public var startTime: Float { Float(startFrame) * frameDurationSeconds }

    /// End time in seconds
    public var endTime: Float { Float(endFrame) * frameDurationSeconds }

    /// Duration in seconds
    public var duration: Float { Float(endFrame - startFrame) * frameDurationSeconds }

    /// Speaker label
    public var speakerLabel: String { "Speaker \(speakerIndex)" }

    public init(
        speakerIndex: Int,
        startFrame: Int,
        endFrame: Int,
        finalized: Bool = true,
        frameDurationSeconds: Float
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
        frameDurationSeconds: Float
    ) {
        self.id = UUID()
        self.speakerIndex = speakerIndex
        self.startFrame = Int(round(startTime / frameDurationSeconds))
        self.endFrame = Int(round(endTime / frameDurationSeconds))
        self.isFinalized = finalized
        self.frameDurationSeconds = frameDurationSeconds
    }

    /// Check if this overlaps with another segment
    public func overlaps(with other: DiarizerSegment) -> Bool {
        (startFrame <= other.endFrame) && (other.startFrame <= endFrame)
    }

    /// Merge another segment into this one
    public mutating func absorb(_ other: DiarizerSegment) {
        startFrame = min(startFrame, other.startFrame)
        endFrame = max(endFrame, other.endFrame)
    }

    /// Extend the end of this segment
    public mutating func extendEnd(toFrame frame: Int) {
        endFrame = max(endFrame, frame)
    }

    /// Extend the start of this segment
    public mutating func extendStart(toFrame frame: Int) {
        startFrame = min(startFrame, frame)
    }
}

// MARK: - Chunk Result

/// Result from a single streaming diarization step (works with any diarizer).
///
/// Maps directly to `SortformerChunkResult` for Sortformer,
/// and wraps `LSEENDStreamingUpdate` for LS-EEND.
public struct DiarizerChunkResult: Sendable {
    /// Speaker probabilities for confirmed/committed frames.
    /// Flat array of shape [frameCount, numSpeakers].
    public let speakerPredictions: [Float]

    /// Number of confirmed frames in this result
    public let frameCount: Int

    /// Frame index of the first confirmed frame
    public let startFrame: Int

    /// Tentative/preview predictions (may change with future data).
    /// Flat array of shape [tentativeFrameCount, numSpeakers].
    public let tentativePredictions: [Float]

    /// Number of tentative frames
    public let tentativeFrameCount: Int

    /// Frame index of first tentative frame
    public var tentativeStartFrame: Int { startFrame + frameCount }

    public init(
        startFrame: Int,
        speakerPredictions: [Float],
        frameCount: Int,
        tentativePredictions: [Float] = [],
        tentativeFrameCount: Int = 0
    ) {
        self.startFrame = startFrame
        self.speakerPredictions = speakerPredictions
        self.frameCount = frameCount
        self.tentativePredictions = tentativePredictions
        self.tentativeFrameCount = tentativeFrameCount
    }

    /// Get probability for a specific speaker at a confirmed frame
    public func probability(speaker: Int, frame: Int, numSpeakers: Int) -> Float {
        guard frame < frameCount, speaker < numSpeakers else { return 0 }
        return speakerPredictions[frame * numSpeakers + speaker]
    }

    /// Get probability for a specific speaker at a tentative frame
    public func tentativeProbability(speaker: Int, frame: Int, numSpeakers: Int) -> Float {
        guard frame < tentativeFrameCount, speaker < numSpeakers else { return 0 }
        return tentativePredictions[frame * numSpeakers + speaker]
    }
}

// MARK: - Timeline

/// Complete diarization timeline managing streaming predictions and segments.
///
/// Generalizes `SortformerTimeline` for any frame-based diarizer. Works with
/// both Sortformer (fixed 4 speakers) and LS-EEND (variable speaker count).
public class DiarizerTimeline {
    /// Post-processing configuration
    public let config: DiarizerPostProcessingConfig

    /// Finalized frame-wise speaker predictions.
    /// Flat array of shape [numFrames, numSpeakers].
    public private(set) var framePredictions: [Float] = []

    /// Tentative predictions.
    /// Flat array of shape [numTentative, numSpeakers].
    public private(set) var tentativePredictions: [Float] = []

    /// Total number of finalized frames
    public private(set) var numFrames: Int = 0

    /// Number of tentative frames
    public var numTentative: Int {
        tentativePredictions.count / config.numSpeakers
    }

    /// Finalized segments per speaker
    public private(set) var segments: [(confirmed: [DiarizerSegment], tentative: [DiarizerSegment])] = []
//
//    /// Tentative segments per speaker (may change as more predictions arrive)
//    public private(set) var tentativeSegments: [[DiarizerSegment]] = []

    /// Duration of finalized predictions in seconds
    public var duration: Float {
        Float(numFrames) * config.frameDurationSeconds
    }

    /// Duration including tentative predictions in seconds
    public var tentativeDuration: Float {
        Float(numFrames + numTentative) * config.frameDurationSeconds
    }

    // Segment builder state
    private var activeSpeakers: [Bool]
    private var activeStarts: [Int]
    private var recentSegments: [(start: Int, end: Int)]

    private static let logger = AppLogger(category: "DiarizerTimeline")

    // MARK: - Init

    /// Initialize for streaming usage
    public init(config: DiarizerPostProcessingConfig) {
        self.config = config
        activeStarts = Array(repeating: 0, count: config.numSpeakers)
        recentSegments = Array(repeating: (0, 0), count: config.numSpeakers)
        activeSpeakers = Array(repeating: false, count: config.numSpeakers)
        segments = Array(repeating: ([], []), count: config.numSpeakers)
    }

    /// Initialize with existing probabilities (batch processing or restored state)
    public convenience init(
        allPredictions: [Float],
        config: DiarizerPostProcessingConfig,
        isComplete: Bool = true
    ) {
        self.init(config: config)
        let numFrames = allPredictions.count / config.numSpeakers
        updateSegments(
            predictions: allPredictions,
            numFrames: numFrames,
            isFinalized: true,
            addTrailingTentative: true
        )
        framePredictions = allPredictions
        self.numFrames = numFrames
        trimPredictions()

        if isComplete {
            finalize()
        }
    }

    // MARK: - Streaming API

    /// Add a new chunk of predictions from the diarizer
    public func addChunk(_ chunk: DiarizerChunkResult) {
        framePredictions.append(contentsOf: chunk.speakerPredictions)
        tentativePredictions = chunk.tentativePredictions
        for i in 0..<config.numSpeakers {
            segments[i].tentative.removeAll(keepingCapacity: true)
        }

        updateSegments(
            predictions: chunk.speakerPredictions,
            numFrames: chunk.frameCount,
            isFinalized: true,
            addTrailingTentative: false
        )
        numFrames += chunk.frameCount

        updateSegments(
            predictions: chunk.tentativePredictions,
            numFrames: chunk.tentativeFrameCount,
            isFinalized: false,
            addTrailingTentative: true
        )
        trimPredictions()
    }

    /// Finalize all tentative data at end of recording
    public func finalize() {
        framePredictions.append(contentsOf: tentativePredictions)
        numFrames += numTentative
        tentativePredictions.removeAll()
        for i in 0..<config.numSpeakers {
            segments[i].confirmed.append(contentsOf: segments[i].tentative)
            segments[i].tentative.removeAll()
            if let lastSegment = segments[i].confirmed.last, lastSegment.length < config.minFramesOn {
                segments[i].confirmed.removeLast()
            }
        }
        trimPredictions()
    }

    /// Reset to initial state
    public func reset() {
        framePredictions.removeAll()
        tentativePredictions.removeAll()
        numFrames = 0
        activeStarts = Array(repeating: 0, count: config.numSpeakers)
        activeSpeakers = Array(repeating: false, count: config.numSpeakers)
        recentSegments = Array(repeating: (0, 0), count: config.numSpeakers)
        segments = Array(repeating: ([], []), count: config.numSpeakers)
    }

    // MARK: - Query

    /// Get probability for a specific speaker at a finalized frame
    public func probability(speaker: Int, frame: Int) -> Float {
        guard frame < numFrames, speaker < config.numSpeakers else { return 0 }
        return framePredictions[frame * config.numSpeakers + speaker]
    }

    /// Get probability for a specific speaker at a tentative frame
    public func tentativeProbability(speaker: Int, frame: Int) -> Float {
        guard frame < numTentative, speaker < config.numSpeakers else { return 0 }
        return tentativePredictions[frame * config.numSpeakers + speaker]
    }

    // MARK: - Segment Detection

    private func updateSegments(
        predictions: [Float],
        numFrames: Int,
        isFinalized: Bool,
        addTrailingTentative: Bool
    ) {
        guard numFrames > 0 else { return }

        let frameOffset = self.numFrames
        let numSpeakers = config.numSpeakers
        let onset = config.onsetThreshold
        let offset = config.offsetThreshold
        let padOnset = config.onsetPadFrames
        let padOffset = config.offsetPadFrames
        let minFramesOn = config.minFramesOn
        let minFramesOff = config.minFramesOff
        let frameDuration = config.frameDurationSeconds

        let tentativeBuffer = padOnset + padOffset + minFramesOff
        let tentativeStartFrame = isFinalized ? (frameOffset + numFrames) - tentativeBuffer : 0

        for speakerIndex in 0..<numSpeakers {
            var start = activeStarts[speakerIndex]
            var speaking = activeSpeakers[speakerIndex]
            var lastSegment = recentSegments[speakerIndex]
            var wasLastSegmentFinal = isFinalized

            for i in 0..<numFrames {
                let index = speakerIndex + i * numSpeakers

                if speaking {
                    if predictions[index] >= offset {
                        continue
                    }

                    speaking = false
                    let end = frameOffset + i + padOffset

                    guard end - start > minFramesOn else { continue }

                    wasLastSegmentFinal = isFinalized && (end < tentativeStartFrame)

                    let newSegment = DiarizerSegment(
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: wasLastSegmentFinal,
                        frameDurationSeconds: frameDuration
                    )

                    if wasLastSegmentFinal {
                        segments[speakerIndex].confirmed.append(newSegment)
                    } else {
                        segments[speakerIndex].tentative.append(newSegment)
                    }
                    lastSegment = (start, end)

                } else if predictions[index] > onset {
                    start = max(0, frameOffset + i - padOnset)
                    speaking = true

                    if start - lastSegment.end <= minFramesOff {
                        start = lastSegment.start
                        if wasLastSegmentFinal {
                            _ = segments[speakerIndex].confirmed.popLast()
                        } else {
                            _ = segments[speakerIndex].tentative.popLast()
                        }
                    }
                }
            }

            if isFinalized {
                activeSpeakers[speakerIndex] = speaking
                activeStarts[speakerIndex] = start
                recentSegments[speakerIndex] = lastSegment
            }

            if addTrailingTentative {
                let end = frameOffset + numFrames + padOffset
                if speaking && (end > start) {
                    let newSegment = DiarizerSegment(
                        speakerIndex: speakerIndex,
                        startFrame: start,
                        endFrame: end,
                        finalized: false,
                        frameDurationSeconds: frameDuration
                    )
                    segments[speakerIndex].tentative.append(newSegment)
                }
            }
        }
    }

    private func trimPredictions() {
        guard let maxStoredFrames = config.maxStoredFrames else { return }
        let numToRemove = framePredictions.count - maxStoredFrames * config.numSpeakers
        if numToRemove > 0 {
            framePredictions.removeFirst(numToRemove)
        }
    }
}
