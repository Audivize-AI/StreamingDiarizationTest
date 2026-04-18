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
        //        await SystemInfo.logOnce(using: logger)
        
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
        lock.lock()
        defer { lock.unlock() }
        
        return try autoreleasepool {
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
            
            input.state.encRetKv = encKvMA
            input.state.encRetScale = encScaleMA
            input.state.encConvCache = encConvCacheMA
            input.state.cnnWindow = cnnWindowMA
            input.state.decRetKv = decKvMA
            input.state.decRetScale = decScaleMA

            return Self.readProbsStrideAware(probsMA)
        }
    }

    /// CoreML returns `probs` with tile-padded inner strides (e.g. logical
    /// shape `[1, T, 10]`, strides `[16, 16, 1]`) — baked at mlpackage
    /// compile time even when `.cpuOnly` is requested. A flat
    /// `withUnsafeBufferPointer` reads the physical `T × 16` footprint and
    /// leaves 6 garbage lanes per row, which the timeline rejects with
    /// `misalignedFinalizedPredictions`. Copy row-by-row using the
    /// published strides so the returned array is exactly `T × S` logical
    /// elements.
    private static func readProbsStrideAware(_ probsMA: MLMultiArray) -> [Float] {
        let shape = probsMA.shape.map { $0.intValue }
        let strides = probsMA.strides.map { $0.intValue }
        guard let innerCount = shape.last, strides.last == 1 else {
            preconditionFailure(
                "probs must be non-empty with innermost stride 1; got shape=\(shape) strides=\(strides)"
            )
        }
        let outerCount = shape.dropLast().reduce(1, *)
        var out = [Float](repeating: 0, count: outerCount * innerCount)
        probsMA.withUnsafeBufferPointer(ofType: Float.self) { buf in
            guard let src = buf.baseAddress else { return }
            let rowBytes = innerCount * MemoryLayout<Float>.stride
            var idx = [Int](repeating: 0, count: max(shape.count - 1, 0))
            for outer in 0..<outerCount {
                var srcOff = 0
                for d in 0..<idx.count { srcOff += idx[d] * strides[d] }
                out.withUnsafeMutableBufferPointer { dst in
                    memcpy(dst.baseAddress!.advanced(by: outer * innerCount),
                           src.advanced(by: srcOff),
                           rowBytes)
                }
                for d in stride(from: idx.count - 1, through: 0, by: -1) {
                    idx[d] += 1
                    if idx[d] < shape[d] { break }
                    idx[d] = 0
                }
            }
        }
        return out
    }
}

