//
//  EmbeddingManager.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/15/26.
//

import Foundation
import Accelerate
import OrderedCollections

/// Manages speaker embedding extraction asynchronously
/// Handles segment merging, pruning, and embedding lifecycle
public class EmbeddingManager {
    
    // MARK: - Configuration
    
    /// Embedding extraction configuration
    public let config: EmbeddingConfig
    
    // MARK: - State
    
    /// The embedding extractor
    private var extractor: TitaNetEmbeddingExtractor?
    private var preprocessor: NeMoMelSpectrogram
    
    /// Audio buffer for extraction (circular buffer of audio samples)
    private var melFeatures: [Float] = []
    /// First Sortformer frame available in the mel buffer (in Sortformer frame units, not mel features)
    private var firstMelFrame: Int = 0
    
    /// Lock for thread-safe access
    private let queue = DispatchQueue(label: "FluidAudio.EmbeddingManager")
    
    private var availibleEmbeddings: [TitaNetEmbedding] = []
    
    /// Logger
    private static let logger = AppLogger(category: "EmbeddingManager")
    
    // MARK: - Initialization
    
    public init(
        config: EmbeddingConfig = EmbeddingConfig(),
        frameDurationSeconds: Float = 0.08  // Match Sortformer's frame duration
    ) {
        self.config = config
        self.preprocessor = NeMoMelSpectrogram(nMels: config.melFeatures, padTo: config.melPadTo)
        // Initialize extractor lazily on first use
    }
    
    /// Initialize the extractor (call before first use)
    public func initialize() throws {
        try queue.sync(flags: .barrier) {
            if extractor == nil {
                extractor = try TitaNetEmbeddingExtractor(config:  config)
            }
        }
    }
    
    // MARK: - Audio Management
    
    /// Add audio to the mel feature buffer
    public func addAudio(from buffer: ArraySlice<Float>, lastAudioSample: Float = 0) {
        queue.sync(flags: .barrier) {
            let (mels, melFrameCount, _) = preprocessor.computeFlatTransposed(audio: buffer, lastAudioSample: lastAudioSample)
            
            // Append new mel features (un-padded length)
            let unpaddedMelLength = buffer.count / config.melStride
            let unpaddedFeatureCount = unpaddedMelLength * config.melFeatures
            
            // Only keep the unpadded portion
            melFeatures.append(contentsOf: mels.prefix(unpaddedFeatureCount))
        }
    }
    
    /// Number of Sortformer frames currently available in the mel buffer
    private var availableMelFrames: Int {
        melFeatures.count / config.melFeatures / config.subsamplingFactor
    }
    
    /// Last Sortformer frame available in the mel buffer (exclusive)
    private var lastMelFrame: Int {
        firstMelFrame + availableMelFrames
    }
    
    /// Range of Sortformer frames available in the mel buffer
    /// Use this to clip embedding requests to available frames
    public var availableFrameRange: Range<Int> {
        queue.sync {
            firstMelFrame..<lastMelFrame
        }
    }
    
    // MARK: - Embedding Extraction
    
    /// Process pending embedding requests from a timeline asynchronously
    public func processRequests(_ requests: [EmbeddingExtractionRequest]) throws -> [TitaNetEmbedding] {
        queue.sync(flags: .barrier) {
            guard let extractor else {
                return []
            }
            
            guard !requests.isEmpty else {
                return []
            }
            
            Self.logger.debug("Processing \(requests.count) embedding requests (mel buffer: frames \(self.firstMelFrame)-\(self.lastMelFrame))")
            
            var embeddings: [TitaNetEmbedding] = []
            
            for request in requests {
                // Check if request length is valid
                guard request.length <= config.maxEmbeddingFrames else {
                    Self.logger.debug("Request [\(request.startFrame)-\(request.endFrame)] too long (\(request.length) > \(config.maxEmbeddingFrames)), skipping")
                    continue
                }
                
                guard request.length >= config.minEmbeddingFrames else {
                    Self.logger.debug("Request [\(request.startFrame)-\(request.endFrame)] too short (\(request.length) < \(config.minEmbeddingFrames)), skipping")
                    continue
                }
                
                // Check if request is within available mel buffer range
                guard request.startFrame >= firstMelFrame else {
                    Self.logger.debug("Request [\(request.startFrame)-\(request.endFrame)] starts before mel buffer (first frame: \(firstMelFrame)), skipping")
                    continue
                }
                
                guard request.endFrame <= lastMelFrame else {
                    Self.logger.debug("Request [\(request.startFrame)-\(request.endFrame)] ends after mel buffer (last frame: \(lastMelFrame)), skipping")
                    continue
                }
                
                // Calculate mel feature indices relative to buffer start
                let relativeStartFrame = request.startFrame - firstMelFrame
                let melStartIndex = relativeStartFrame * config.subsamplingFactor * config.melFeatures
                
                // Calculate padded mel length for the model
                let melLength = IndexUtils.nextMultiple(
                    of: config.melPadTo,
                    for: request.length * config.subsamplingFactor
                )
                let melEndIndex = melStartIndex + melLength * config.melFeatures
                
                // Bounds check on mel features array
                guard melStartIndex >= 0 && melEndIndex <= melFeatures.count else {
                    Self.logger.warning("Mel indices out of bounds: [\(melStartIndex)-\(melEndIndex)] for buffer of \(melFeatures.count) features")
                    continue
                }
                
                let mels = Array(melFeatures[melStartIndex..<melEndIndex])
                
                do {
                    let embeddingVector = try extractor.extractEmbedding(mels: mels, melLength: melLength)
                    let embedding = TitaNetEmbedding(
                        embedding: embeddingVector,
                        startFrame: request.startFrame,
                        endFrame: request.endFrame
                    )
                    embeddings.append(embedding)
                    Self.logger.debug("Extracted embedding [\(request.startFrame)-\(request.endFrame)]")
                } catch {
                    Self.logger.error("Embedding extraction failed for [\(request.startFrame)-\(request.endFrame)]: \(error)")
                }
            }
            
            return embeddings
        }
    }
    
    public func takeMatches(for segment: EmbeddingSegment) -> [TitaNetEmbedding] {
        queue.sync(flags: .barrier) {
            guard availibleEmbeddings.count > 0 else {
                return []
            }
            
            let count = availibleEmbeddings.count
            var embeddings: [TitaNetEmbedding] = []
            for i in (0..<count).reversed() {
                if availibleEmbeddings[i].framesOutside(of: segment) <= config.maxOutsideFrames {
                    embeddings.append(availibleEmbeddings.remove(at: i))
                }
            }
            
            return embeddings
        }
    }

    public func returnEmbeddings(from segment: EmbeddingSegment) {
        queue.sync(flags: .barrier) {
            availibleEmbeddings.append(contentsOf: segment.embeddings)
        }
    }
    
    public func returnEmbeddings(from segments: [EmbeddingSegment]) {
        queue.sync(flags: .barrier) {
            for segment in segments {
                availibleEmbeddings.append(contentsOf: segment.embeddings)
            }
        }
    }
    
    public func dropFrames(before firstTentativeFrame: Int) {
        queue.sync(flags: .barrier) {
            // Don't drop if the new frame is not ahead of current start
            guard firstTentativeFrame > firstMelFrame else {
                return
            }
            
            // Calculate how many Sortformer frames to drop
            let framesToDrop = firstTentativeFrame - firstMelFrame
            
            // Calculate how many mel features to drop
            // Each Sortformer frame = subsamplingFactor mel frames
            // Each mel frame = melFeatures features
            let featuresToDrop = framesToDrop * config.subsamplingFactor * config.melFeatures
            
            // Don't drop more than we have
            let actualFeaturesToDrop = min(featuresToDrop, melFeatures.count)
            
            if actualFeaturesToDrop > 0 {
                melFeatures.removeFirst(actualFeaturesToDrop)
                firstMelFrame = firstTentativeFrame
                Self.logger.debug("Dropped \(framesToDrop) frames, mel buffer now starts at frame \(firstMelFrame)")
            }
        }
    }
}

// MARK: - Errors

public enum EmbeddingManagerError: Error, LocalizedError {
    case audioNotAvailable
    case extractorNotInitialized
    
    public var errorDescription: String? {
        switch self {
        case .audioNotAvailable:
            return "Audio not available in buffer for requested frame range"
        case .extractorNotInitialized:
            return "Embedding extractor not initialized. Call initialize() first."
        }
    }
}
