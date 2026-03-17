import AVFoundation
import CoreML
import Foundation

private struct ProbeMatrix: Codable {
    let rows: Int
    let columns: Int
    let values: [Float]

    init(_ matrix: LSEENDMatrix) {
        rows = matrix.rows
        columns = matrix.columns
        values = matrix.values
    }
}

private struct ProbeInferenceResult: Codable {
    let logits: ProbeMatrix
    let probabilities: ProbeMatrix
    let fullLogits: ProbeMatrix
    let fullProbabilities: ProbeMatrix
    let frameHz: Double
    let durationSeconds: Double

    init(_ result: LSEENDInferenceResult) {
        logits = ProbeMatrix(result.logits)
        probabilities = ProbeMatrix(result.probabilities)
        fullLogits = ProbeMatrix(result.fullLogits)
        fullProbabilities = ProbeMatrix(result.fullProbabilities)
        frameHz = result.frameHz
        durationSeconds = result.durationSeconds
    }
}

private struct ProbeStreamingProgress: Codable {
    let chunkIndex: Int
    let bufferSeconds: Double
    let numFramesEmitted: Int
    let totalFramesEmitted: Int
    let flush: Bool

    init(_ progress: LSEENDStreamingProgress) {
        chunkIndex = progress.chunkIndex
        bufferSeconds = progress.bufferSeconds
        numFramesEmitted = progress.numFramesEmitted
        totalFramesEmitted = progress.totalFramesEmitted
        flush = progress.flush
    }
}

private struct ProbeStreamingResult: Codable {
    let result: ProbeInferenceResult
    let updates: [ProbeStreamingProgress]

    init(_ simulation: LSEENDStreamingSimulationResult) {
        result = ProbeInferenceResult(simulation.result)
        updates = simulation.updates.map(ProbeStreamingProgress.init)
    }
}

private struct ProbeSessionCheckResult: Codable {
    let firstUpdateRows: Int
    let firstUpdateTotalEmittedFrames: Int
    let committedProbabilities: ProbeMatrix
    let snapshotProbabilities: ProbeMatrix
    let repeatedFinalizeReturnedUpdate: Bool
    let repeatedSnapshotProbabilities: ProbeMatrix
}

/// Lightweight variant info for the standalone probe (avoids pulling in ModelNames + its dependencies).
private enum ProbeVariant: String {
    case ami, callhome, dihard2, dihard3

    /// Subfolder name in the HuggingFace cache (e.g. "AMI", "DIHARD III").
    var folderName: String {
        switch self {
        case .ami: return "AMI"
        case .callhome: return "CALLHOME"
        case .dihard2: return "DIHARD II"
        case .dihard3: return "DIHARD III"
        }
    }

    /// Base file stem (e.g. "ls_eend_ami_step").
    var stem: String {
        switch self {
        case .ami: return "ls_eend_ami_step"
        case .callhome: return "ls_eend_callhome_step"
        case .dihard2: return "ls_eend_dih2_step"
        case .dihard3: return "ls_eend_dih3_step"
        }
    }

    /// Resolve a descriptor from the standard HuggingFace model cache directory.
    func resolveDescriptor() throws -> LSEENDModelDescriptor {
        let cacheDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidAudio/Models/ls-eend")
            .appendingPathComponent(folderName)
        let modelURL = cacheDir.appendingPathComponent("\(stem).mlmodelc")
        let metadataURL = cacheDir.appendingPathComponent("\(stem).json")

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw ProbeError.invalidArguments("Model not found at \(modelURL.path). Run the app first to download models.")
        }
        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            throw ProbeError.invalidArguments("Metadata not found at \(metadataURL.path).")
        }

        // We need an LSEENDVariant for the descriptor init, but we can't use it directly
        // without pulling in ModelNames. Instead we use the variant that matches our folder.
        // The descriptor only uses variant for labeling, not for path resolution.
        return LSEENDModelDescriptor(variant: lseendVariant, modelURL: modelURL, metadataURL: metadataURL)
    }

    /// Map to the actual LSEENDVariant enum value.
    var lseendVariant: LSEENDVariant {
        switch self {
        case .ami: return .ami
        case .callhome: return .callhome
        case .dihard2: return .dihard2
        case .dihard3: return .dihard3
        }
    }
}

@main
private struct LSEENDRuntimeProbe {
    static func main() throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let outputURL = try parseOptionalOutputURL(from: arguments)
        guard let command = arguments.first else {
            throw ProbeError.invalidArguments("Missing command.")
        }

        let payload: Data
        switch command {
        case "offline-features":
            let variant = try parseVariant(from: arguments)
            let audioURL = try parseAudioURL(from: arguments)
            let descriptor = try variant.resolveDescriptor()
            let metadataData = try Data(contentsOf: descriptor.metadataURL)
            let metadata = try JSONDecoder().decode(LSEENDModelMetadata.self, from: metadataData)
            let converter = AudioConverter(
                targetFormat: AVAudioFormat(
                    commonFormat: .pcmFormatFloat32,
                    sampleRate: Double(metadata.resolvedSampleRate),
                    channels: 1,
                    interleaved: false
                )!
            )
            let audio = try converter.resampleAudioFile(audioURL)
            let extractor = LSEENDOfflineFeatureExtractor(metadata: metadata)
            payload = try encodeJSON(ProbeMatrix(extractor.extractFeatures(audio: audio)))
        case "streaming-features":
            let variant = try parseVariant(from: arguments)
            let audioURL = try parseAudioURL(from: arguments)
            let chunkSeconds = try parseDouble(flag: "--chunk-seconds", from: arguments)
            let descriptor = try variant.resolveDescriptor()
            let engine = try LSEENDInferenceHelper(descriptor: descriptor, computeUnits: .cpuOnly)
            let converter = AudioConverter(
                targetFormat: AVAudioFormat(
                    commonFormat: .pcmFormatFloat32,
                    sampleRate: Double(engine.targetSampleRate),
                    channels: 1,
                    interleaved: false
                )!
            )
            let audio = try converter.resampleAudioFile(audioURL)
            let extractor = LSEENDStreamingFeatureExtractor(metadata: engine.metadata)
            let chunkSize = max(1, Int(round(chunkSeconds * Double(engine.targetSampleRate))))
            var features = LSEENDMatrix.empty(columns: engine.metadata.inputDim)
            var start = 0
            while start < audio.count {
                let stop = min(audio.count, start + chunkSize)
                let chunkFeatures = try extractor.pushAudio(Array(audio[start..<stop]))
                if !chunkFeatures.isEmpty {
                    features = features.appendingRows(chunkFeatures)
                }
                start = stop
            }
            let finalFeatures = try extractor.finalize()
            if !finalFeatures.isEmpty {
                features = features.appendingRows(finalFeatures)
            }
            payload = try encodeJSON(ProbeMatrix(features))
        case "offline":
            let variant = try parseVariant(from: arguments)
            let audioURL = try parseAudioURL(from: arguments)
            let engine = try LSEENDInferenceHelper(
                descriptor: variant.resolveDescriptor(), computeUnits: .cpuOnly)
            payload = try encodeJSON(ProbeInferenceResult(engine.infer(audioFileURL: audioURL)))
        case "streaming":
            let variant = try parseVariant(from: arguments)
            let audioURL = try parseAudioURL(from: arguments)
            let chunkSeconds = try parseDouble(flag: "--chunk-seconds", from: arguments)
            let engine = try LSEENDInferenceHelper(
                descriptor: variant.resolveDescriptor(), computeUnits: .cpuOnly)
            payload = try encodeJSON(
                ProbeStreamingResult(engine.simulateStreaming(audioFileURL: audioURL, chunkSeconds: chunkSeconds)))
        case "session-check":
            let variant = try parseVariant(from: arguments)
            let audioURL = try parseAudioURL(from: arguments)
            let chunkSeconds = try parseDouble(flag: "--chunk-seconds", from: arguments)
            payload = try encodeJSON(
                try runSessionCheck(variant: variant, audioURL: audioURL, chunkSeconds: chunkSeconds))
        default:
            throw ProbeError.invalidArguments("Unknown command: \(command)")
        }

        if let outputURL {
            try payload.write(to: outputURL, options: .atomic)
        } else {
            FileHandle.standardOutput.write(payload)
        }
    }

    private static func runSessionCheck(
        variant: ProbeVariant,
        audioURL: URL,
        chunkSeconds: Double
    ) throws -> ProbeSessionCheckResult {
        let engine = try LSEENDInferenceHelper(
            descriptor: variant.resolveDescriptor(), computeUnits: .cpuOnly)
        let converter = AudioConverter(
            targetFormat: AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(engine.targetSampleRate),
                channels: 1,
                interleaved: false
            )!
        )
        let audio = try converter.resampleAudioFile(audioURL)
        let chunkSize = max(1, Int(round(chunkSeconds * Double(engine.targetSampleRate))))
        let session = try engine.createSession(inputSampleRate: engine.targetSampleRate)

        let firstChunk = Array(audio.prefix(chunkSize))
        let firstUpdate = try session.pushAudio(firstChunk)

        var committed = LSEENDMatrix.empty(columns: engine.metadata.realOutputDim)
        if let firstUpdate, !firstUpdate.probabilities.isEmpty {
            committed = committed.appendingRows(firstUpdate.probabilities)
        }

        var startIndex = firstChunk.count
        while startIndex < audio.count {
            let stopIndex = min(audio.count, startIndex + chunkSize)
            if let update = try session.pushAudio(Array(audio[startIndex..<stopIndex])),
                !update.probabilities.isEmpty
            {
                committed = committed.appendingRows(update.probabilities)
            }
            startIndex = stopIndex
        }

        if let finalUpdate = try session.finalize(), !finalUpdate.probabilities.isEmpty {
            committed = committed.appendingRows(finalUpdate.probabilities)
        }

        let repeatedFinalize = try session.finalize()
        return ProbeSessionCheckResult(
            firstUpdateRows: firstUpdate?.probabilities.rows ?? 0,
            firstUpdateTotalEmittedFrames: firstUpdate?.totalEmittedFrames ?? 0,
            committedProbabilities: ProbeMatrix(committed),
            snapshotProbabilities: ProbeMatrix(session.snapshot().probabilities),
            repeatedFinalizeReturnedUpdate: repeatedFinalize != nil,
            repeatedSnapshotProbabilities: ProbeMatrix(session.snapshot().probabilities)
        )
    }

    private static func encodeJSON<T: Encodable>(_ value: T) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        return try encoder.encode(value)
    }

    private static func parseVariant(from arguments: [String]) throws -> ProbeVariant {
        let raw = try parseString(flag: "--variant", from: arguments)
        guard let variant = ProbeVariant(rawValue: raw.lowercased()) else {
            throw ProbeError.invalidArguments("Unsupported variant: \(raw)")
        }
        return variant
    }

    private static func parseAudioURL(from arguments: [String]) throws -> URL {
        URL(fileURLWithPath: try parseString(flag: "--audio", from: arguments))
    }

    private static func parseOptionalOutputURL(from arguments: [String]) throws -> URL? {
        guard let index = arguments.firstIndex(of: "--output") else {
            return nil
        }
        guard arguments.indices.contains(index + 1) else {
            throw ProbeError.invalidArguments("Missing --output path.")
        }
        return URL(fileURLWithPath: arguments[index + 1])
    }

    private static func parseDouble(flag: String, from arguments: [String]) throws -> Double {
        guard let value = Double(try parseString(flag: flag, from: arguments)) else {
            throw ProbeError.invalidArguments("Invalid numeric value for \(flag).")
        }
        return value
    }

    private static func parseString(flag: String, from arguments: [String]) throws -> String {
        guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else {
            throw ProbeError.invalidArguments("Missing \(flag).")
        }
        return arguments[index + 1]
    }
}

private enum ProbeError: LocalizedError {
    case invalidArguments(String)

    var errorDescription: String? {
        switch self {
        case .invalidArguments(let message):
            return message
        }
    }
}
