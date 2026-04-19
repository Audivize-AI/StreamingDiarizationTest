//
//  LSEENDModels.swift
//  LS-EEND-Test
//
//  Created by Benjamin Lee on 4/16/26.
//

import Foundation
import CoreML
import Accelerate

public class LSEENDModel {
    public let metadata: LSEENDMetadata
    
    private let model: MLModel
    
    private let lock = NSLock()
    
    private static let logger = AppLogger(category: "LS-EEND Model")
    
    // MARK: - Init
    
    public init(modelURL: URL, computeUnits: MLComputeUnits = .cpuOnly) throws {
        // Load the model from the URL
        let modelConfig = MLModelConfiguration()
        modelConfig.computeUnits = computeUnits
        self.model = try MLModel(contentsOf: modelURL, configuration: modelConfig)
        
        // Load the config from metadata
        guard let userMetadata = self.model.modelDescription.metadata[.creatorDefinedKey] as? [String: Any],
              let json = userMetadata["config"] as? String
        else {
            throw LSEENDError.initializationFailed("No `config` found in model metadata")
        }
        
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        self.metadata = try decoder.decode(LSEENDMetadata.self, from: Data(json.utf8))
    }
    
    /// Download LS-EEND models from HuggingFace.
    ///
    /// - Parameters:
    ///   - variant: The model variant to load (default: `.dihard3`).
    ///   - stepSize: The model step size to load (default: `.step100ms`).
    ///   - cacheDirectory: Directory to cache downloaded models (defaults to app support)
    ///   - computeUnits: Model compute units (`.cpuOnly` seems to be fastest for this model)
    /// - Returns: LS-EEND Model Wrapper
    public static func loadFromHuggingFace(
        variant: LSEENDVariant = .dihard3,
        stepSize: LSEENDStepSize = .step100ms,
        cacheDirectory: URL? = nil,
        computeUnits: MLComputeUnits = .cpuOnly,
        progressHandler: DownloadUtils.ProgressHandler? = nil
    ) async throws -> LSEENDModel {
        let directory =
        cacheDirectory
        ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidAudio/Models")
        
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        
        let repo = variant.repo
        let repoPath = directory.appendingPathComponent(repo.folderName)
        let modelRelPath = variant.fileName(forStep: stepSize)
        // LS-EEND repos live under a `subPath` inside the HF repo
        // (e.g. `optimized/dih3`). Both the remote listing path and the
        // local save path must include it — `downloadSubdirectory` saves
        // files at `repoPath + <repo-relative path>`, so `modelURL` has to
        // mirror that layout or the exists-check never fires.
        let fullRelPath = repo.subPath.map { "\($0)/\(modelRelPath)" } ?? modelRelPath
        let modelURL = repoPath.appendingPathComponent(fullRelPath)

        let modelExists = FileManager.default.fileExists(atPath: modelURL.path)

        if !modelExists {
            // Narrow to just the one mlmodelc — listing the whole step dir
            // is fine here since each step dir contains only its own mlmodelc.
            logger.info("Models not found in cache at \(modelURL.path); downloading \(fullRelPath)…")
            try await DownloadUtils.downloadSubdirectory(
                repo, subdirectory: fullRelPath, to: repoPath
            )
        }
        
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw LSEENDError.initializationFailed(
                "HF download completed but mlmodelc missing at \(modelURL.path). "
                + "Expected HF path: \(modelRelPath)"
            )
        }
        
        return try LSEENDModel(modelURL: modelURL, computeUnits: computeUnits)
    }
    
    // MARK: - Inference
    
    public func predict(from input: LSEENDInput) throws -> [Float] {
        try autoreleasepool {
            lock.lock()
            defer { lock.unlock() }
            
            let prediction = try model.prediction(from: input)
            
            guard let probsMA = prediction.featureValue(for: "probs")?.multiArrayValue,
                  let encKvMA = prediction.featureValue(for: "enc_kv_new")?.multiArrayValue,
                  let encScaleMA = prediction.featureValue(for: "enc_scale_new")?.multiArrayValue,
                  let encConvCacheMA = prediction.featureValue(for: "enc_conv_cache_new")?.multiArrayValue,
                  let cnnWindowMA = prediction.featureValue(for: "cnn_window_new")?.multiArrayValue,
                  let decKvMA = prediction.featureValue(for: "dec_kv_new")?.multiArrayValue,
                  let decScaleMA = prediction.featureValue(for: "dec_scale_new")?.multiArrayValue
            else {
                throw LSEENDError.inferenceFailed("Failed to extract predictions from CoreML model.")
            }
            
            // Update state
            input.state.encRetKv = encKvMA
            input.state.encRetScale = encScaleMA
            input.state.encConvCache = encConvCacheMA
            input.state.cnnWindow = cnnWindowMA
            input.state.decRetKv = decKvMA
            input.state.decRetScale = decScaleMA
            
            // Copy speaker sigmoids and skip warmup frames
            let warmup = input.warmupFrames
            let outputFrames = metadata.chunkSize - warmup
            let outputSpeakers = metadata.maxSpeakers
            guard outputFrames > 0, outputSpeakers > 0 else { return [] }
            
            guard probsMA.strides.last?.intValue == 1 else {
                throw LSEENDError.inferenceFailed(
                    "Probs innermost stride must be 1. CoreML model produced strides: \(probsMA.strides).")
            }
            let frameStride = probsMA.strides[1].intValue

            var probsOut = [Float](repeating: 0, count: outputFrames * outputSpeakers)
            let maBase = probsMA.dataPointer.assumingMemoryBound(to: Float.self)
            
            probsOut.withUnsafeMutableBufferPointer { flatPtr in
                vDSP_mmov(
                    maBase + warmup * frameStride,
                    flatPtr.baseAddress!,
                    vDSP_Length(outputSpeakers),
                    vDSP_Length(outputFrames),
                    vDSP_Length(frameStride),
                    vDSP_Length(outputSpeakers)
                )
            }

            return probsOut
        }
    }
}

