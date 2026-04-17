//
//  LSEENDModels.swift
//  LS-EEND-Test
//
//  Created by Benjamin Lee on 4/16/26.
//

import Foundation
import CoreML

public class LSEENDModel {
    public let metadata: LSEENDMetadata
    private let model: MLModel
    
    private let melBuffer: MLMultiArray
    private let validBuffer: MLMultiArray
    private let probsBuffer: MLMultiArray  // contiguous [1, T, maxSpk] for stride-aware readback
    
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
        
        // Initialize preallocated input buffers
        let T = NSNumber(value: metadata.chunkSize)
        let F = NSNumber(value: metadata.featDim)
        let S = NSNumber(value: metadata.maxSpeakers)
        self.melBuffer = try MLMultiArray(shape: [1, T, F], dataType: .float32)
        self.validBuffer = try MLMultiArray(shape: [T], dataType: .float32)
        self.probsBuffer = try MLMultiArray(shape: [1, T, S], dataType: .float32)
    }
    
    /// Download LS-EEND models from HuggingFace and construct a descriptor.
    ///
    /// Downloads all variant files on first call; subsequent calls use the cache.
    /// The returned descriptor points at the cached `.mlmodelc` and `.json` files.
    ///
    /// - Parameters:
    ///   - variant: The model variant to load (default: `.dihard3`).
    ///   - cacheDirectory: Directory to cache downloaded models (defaults to app support)
    ///   - computeUnits: Model compute units (.cpuOnly seems to be fastest for this model)
    /// - Returns: A descriptor ready for ``LSEENDInferenceEngine/init(descriptor:computeUnits:)``.
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

        let repo = Repo.lseend
        let repoPath = directory.appendingPathComponent(repo.folderName)
        let requiredModels = ModelNames.getRequiredModelNames(for: repo, variant: variant.fileName(forStep: stepSize))

        let allModelsExist = requiredModels.allSatisfy { model in
            let modelPath = repoPath.appendingPathComponent(model)
            return FileManager.default.fileExists(atPath: modelPath.path)
        }

        if !allModelsExist {
            logger.info("Models not found in cache at \(repoPath.path)")
            try await DownloadUtils.downloadSubdirectory(repo, subdirectory: variant.subPath, to: directory)
        }

        let modelURL = repoPath.appendingPathComponent(variant.fileName(forStep: stepSize))
        
        return try LSEENDModel(modelURL: modelURL, computeUnits: computeUnits)
    }
    
    // MARK: Inference
    
    public func predict(
        state: inout LSEENDState,
        features: [Float],
        frameMask: [Float]
    ) throws -> [Float] {
        guard features.count == melBuffer.count else {
            throw LSEENDError.inferenceFailed(
                "Invalid feature count: got \(features.count), expected \(melBuffer.count)")
        }
        guard frameMask.count == validBuffer.count else {
            throw LSEENDError.inferenceFailed(
                "Invalid frame mask size: got \(frameMask.count), expected \(validBuffer.count)")
        }

        let floatStride = MemoryLayout<Float>.stride
        features.withUnsafeBufferPointer { featPtr in
            if let base = featPtr.baseAddress {
                memcpy(melBuffer.dataPointer, base, melBuffer.count * floatStride)
            }
        }
        frameMask.withUnsafeBufferPointer { maskPtr in
            if let base = maskPtr.baseAddress {
                memcpy(validBuffer.dataPointer, base, validBuffer.count * floatStride)
            }
        }

        let input = LSEENDInput(
            state: state,
            melFeatures: melBuffer,
            validMask: validBuffer
        )
        let prediction = try model.prediction(from: input)

        func extract(_ name: String) throws -> MLMultiArray {
            guard let v = prediction.featureValue(for: name)?.multiArrayValue else {
                throw LSEENDError.inferenceFailed("Missing output feature '\(name)'")
            }
            return v
        }
        let probs = try extract("probs")
        let newState: [(String, MLMultiArray, WritableKeyPath<LSEENDState, MLMultiArray>)] = [
            ("enc_kv_new",         try extract("enc_kv_new"),         \.encRetKv),
            ("enc_scale_new",      try extract("enc_scale_new"),      \.encRetScale),
            ("enc_conv_cache_new", try extract("enc_conv_cache_new"), \.encConvCache),
            ("cnn_window_new",     try extract("cnn_window_new"),     \.cnnWindow),
            ("dec_kv_new",         try extract("dec_kv_new"),         \.decRetKv),
            ("dec_scale_new",      try extract("dec_scale_new"),      \.decRetScale),
        ]

        // Swap refs when strides are contiguous (common on CPU, zero-copy);
        // fall back to stride-aware memcpy otherwise. Either way, the inner
        // buffer for the next call is the latest CoreML-vended MLMultiArray.
        for (_, src, kp) in newState {
            if src.strides.last?.intValue == 1,
               src.strides == state[keyPath: kp].strides
            {
                state[keyPath: kp] = src  // zero-copy ref swap
            } else {
                ANEMemoryUtils.strideAwareCopy(from: src, to: state[keyPath: kp])
            }
        }

        // probs strides may be non-contiguous (ANE tile padding). Copy into
        // our contiguous probsBuffer via ANEMemoryUtils.strideAwareCopy, then
        // flat-memcpy out into a Swift array.
        ANEMemoryUtils.strideAwareCopy(from: probs, to: probsBuffer)
        let count = probsBuffer.count
        return [Float](unsafeUninitializedCapacity: count) { buf, outCount in
            memcpy(buf.baseAddress!, probsBuffer.dataPointer, count * floatStride)
            outCount = count
        }
    }
}

struct LSEENDOutput {
    let state: LSEENDState
    let preds: [Float]
}

private class LSEENDInput: MLFeatureProvider {
    let state: LSEENDState
    let melFeatures: MLMultiArray
    let validMask: MLMultiArray
    
    var featureNames: Set<String> {[
        "features",
        "enc_kv", "enc_scale",
        "enc_conv_cache", "cnn_window",
        "dec_kv", "dec_scale",
        "valid_mask"
    ]}
    
    init(state: LSEENDState, melFeatures: MLMultiArray, validMask: MLMultiArray) {
        self.state = state
        self.melFeatures = melFeatures
        self.validMask = validMask
    }
    
    func featureValue(for featureName: String) -> MLFeatureValue? {
        switch featureName {
        case "features": return MLFeatureValue(multiArray: melFeatures)
        case "enc_kv": return MLFeatureValue(multiArray: state.encRetKv)
        case "enc_scale": return MLFeatureValue(multiArray: state.encRetScale)
        case "enc_conv_cache": return MLFeatureValue(multiArray: state.encConvCache)
        case "cnn_window": return MLFeatureValue(multiArray: state.cnnWindow)
        case "dec_kv": return MLFeatureValue(multiArray: state.decRetKv)
        case "dec_scale": return MLFeatureValue(multiArray: state.decRetScale)
        case "valid_mask": return MLFeatureValue(multiArray: validMask)
        default: return nil
        }
    }
}
