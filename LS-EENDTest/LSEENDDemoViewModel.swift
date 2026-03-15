import AppKit
import AVFoundation
import Combine
import Foundation
import SwiftUI

private extension ProcessInfo {
    var isRunningXCTest: Bool {
        environment["XCTestConfigurationFilePath"] != nil
    }

    var isRunningUITestSmokeMode: Bool {
        arguments.contains("--uitest-no-model-load")
    }
}

private protocol LSEENDAudioSource: AnyObject {
    var label: String { get }
    func start() throws
    func stop()
}

private final class LSEENDMicrophoneSource: NSObject, LSEENDAudioSource {
    private(set) var label: String

    private let engine = AVAudioEngine()
    private let audioStream: AudioStream
    private let onChunk: ([Float]) -> Void
    private let onStatus: (String) -> Void

    init(
        targetSampleRate: Double,
        blockSeconds: Double,
        onChunk: @escaping ([Float]) -> Void,
        onStatus: @escaping (String) -> Void
    ) throws {
        self.onChunk = onChunk
        self.onStatus = onStatus
        audioStream = try AudioStream(
            chunkDuration: blockSeconds,
            chunkSkip: blockSeconds,
            chunkingStrategy: .useFixedSkip,
            startupStrategy: .waitForFullChunk,
            sampleRate: targetSampleRate
        )
        label = "Microphone -> \(Int(targetSampleRate)) Hz"
        super.init()
        audioStream.bind { [weak self] (chunk: [Float], _) in
            self?.onChunk(chunk)
        }
    }

    func start() throws {
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        label = "Microphone: \(Int(format.sampleRate)) Hz -> \(Int(audioStream.sampleRate)) Hz"
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { [weak self] buffer, _ in
            guard let self else { return }
            do {
                try self.audioStream.write(from: buffer)
            } catch {
                self.onStatus("Microphone write failed: \(error.localizedDescription)")
            }
        }
        engine.prepare()
        try engine.start()
    }

    func stop() {
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
    }
}

private final class LSEENDAudioFileSource: LSEENDAudioSource {
    let label: String

    private let queue = DispatchQueue(label: "LS-EEND.AudioFileSource", qos: .userInitiated)
    private let audio: [Float]
    private let sampleRate: Int
    private let chunkSize: Int
    private let speed: Double
    private let onChunk: ([Float]) -> Void
    private let onCompletion: () -> Void
    private var stopped = false

    init(
        fileURL: URL,
        targetSampleRate: Int,
        blockSeconds: Double,
        speed: Double,
        onChunk: @escaping ([Float]) -> Void,
        onCompletion: @escaping () -> Void
    ) throws {
        let converter = AudioConverter(
            targetFormat: AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: Double(targetSampleRate),
                channels: 1,
                interleaved: false
            )!
        )
        audio = try converter.resampleAudioFile(fileURL)
        sampleRate = targetSampleRate
        chunkSize = max(1, Int(round(blockSeconds * Double(targetSampleRate))))
        self.speed = max(speed, 0.01)
        self.onChunk = onChunk
        self.onCompletion = onCompletion
        label = "Simulated: \(fileURL.lastPathComponent)"
    }

    func start() throws {
        stopped = false
        queue.async { [weak self] in
            guard let self else { return }
            var startIndex = 0
            while startIndex < self.audio.count, !self.stopped {
                let stopIndex = min(self.audio.count, startIndex + self.chunkSize)
                self.onChunk(Array(self.audio[startIndex..<stopIndex]))
                let sleepSeconds = Double(stopIndex - startIndex) / Double(self.sampleRate) / self.speed
                Thread.sleep(forTimeInterval: sleepSeconds)
                startIndex = stopIndex
            }
            if !self.stopped {
                self.onCompletion()
            }
        }
    }

    func stop() {
        stopped = true
    }
}

struct LSEENDHeatmapSnapshot {
    let title: String
    let matrix: LSEENDMatrix
    let speakerLabels: [String]
    let startSeconds: Double
    let endSeconds: Double
    let previewStartSeconds: Double?
    let previewEndSeconds: Double?
    let binary: Bool
}

final class LSEENDDemoViewModel: ObservableObject {
    @Published var selectedVariant: LSEENDVariant = .dihard3
    @Published var useCustomPaths = false
    @Published var modelPath = ""
    @Published var metadataPath = ""
    @Published var threshold: Double = 0.5
    @Published var medianWidth: Int = 11
    @Published var windowSeconds: Double = 120
    @Published var blockSeconds: Double = 0.5
    @Published var refreshSeconds: Double = 0.25
    @Published var simulateSpeed: Double = 1.0
    @Published var simulationPath = ""

    @Published private(set) var statusText = "Loading model..."
    @Published private(set) var sourceText = "Not started"
    @Published private(set) var bufferText = "Buffered: 0.0 s"
    @Published private(set) var inferenceText = "Inference: idle"
    @Published private(set) var modelInfoText = ""

    @Published private(set) var committedProbabilities = LSEENDMatrix.empty(columns: 0)
    @Published private(set) var previewProbabilities = LSEENDMatrix.empty(columns: 0)
    @Published private(set) var previewStartFrame = 0
    @Published var speakerLabels: [String] = []
    @Published var displayOrder: [Int] = []

    private let processingQueue = DispatchQueue(label: "LS-EEND.Processing", qos: .userInitiated)
    private let processor = LSEENDDiarizer()
    private var audioSource: LSEENDAudioSource?
    private var currentDescriptor: LSEENDModelDescriptor?
    private var pendingAudioCount = 0
    private var totalSamplesReceived = 0
    private let outputDirectory: URL = {
        if let override = ProcessInfo.processInfo.environment["LSEEND_WORKSPACE_ROOT"], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true).appendingPathComponent("artifacts/mic_gui_swift", isDirectory: true)
        }
        var url = URL(fileURLWithPath: #filePath)
        while url.lastPathComponent != "LS-EEND" && url.path != "/" {
            url.deleteLastPathComponent()
        }
        return url.appendingPathComponent("artifacts/mic_gui_swift", isDirectory: true)
    }()

    init() {
        applyVariantDefaults()
        if ProcessInfo.processInfo.isRunningXCTest || ProcessInfo.processInfo.isRunningUITestSmokeMode {
            statusText = "Tests running."
            sourceText = "UI demo disabled during tests"
        } else {
            reloadModel()
        }
    }

    var probabilitySnapshot: LSEENDHeatmapSnapshot {
        let combined = combinedProbabilities()
        let ordered = reordered(matrix: combined)
        let frameRate = processor.modelFrameHz ?? 10
        let windowFrames = max(1, Int(round(windowSeconds * frameRate)))
        let startFrame = max(0, ordered.rows - windowFrames)
        let shown = ordered.slicingRows(start: startFrame, end: ordered.rows)
        let previewRange = previewRangeInDisplayedMatrix(totalRows: combined.rows, startFrame: startFrame)
        return LSEENDHeatmapSnapshot(
            title: "Speaker Probability",
            matrix: shown,
            speakerLabels: speakerLabelsForDisplay(),
            startSeconds: Double(startFrame) / frameRate,
            endSeconds: Double(ordered.rows) / frameRate,
            previewStartSeconds: previewRange.start,
            previewEndSeconds: previewRange.end,
            binary: false
        )
    }

    var binarySnapshot: LSEENDHeatmapSnapshot {
        let ordered = reordered(matrix: currentBinary())
        let frameRate = processor.modelFrameHz ?? 10
        let windowFrames = max(1, Int(round(windowSeconds * frameRate)))
        let startFrame = max(0, ordered.rows - windowFrames)
        let shown = ordered.slicingRows(start: startFrame, end: ordered.rows)
        let previewRange = previewRangeInDisplayedMatrix(totalRows: combinedProbabilities().rows, startFrame: startFrame)
        return LSEENDHeatmapSnapshot(
            title: "Binary Activity",
            matrix: shown,
            speakerLabels: speakerLabelsForDisplay(),
            startSeconds: Double(startFrame) / frameRate,
            endSeconds: Double(ordered.rows) / frameRate,
            previewStartSeconds: previewRange.start,
            previewEndSeconds: previewRange.end,
            binary: true
        )
    }

    func applyVariantDefaults() {
        guard !useCustomPaths else { return }
        let variant = selectedVariant
        let cacheDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FluidAudio/Models/ls-eend")
        modelPath = cacheDir.appendingPathComponent(variant.modelFile).path
        metadataPath = cacheDir.appendingPathComponent(variant.configFile).path
    }

    func reloadModel() {
        resetTimeline(clearStatus: false)
        applyVariantDefaults()
        let variant = selectedVariant
        statusText = "Loading \(variant.rawValue) model..."

        if useCustomPaths {
            let descriptor = descriptorFromCurrentSelection()
            processingQueue.async { [weak self] in
                self?.initializeProcessor(descriptor: descriptor)
            }
        } else {
            Task { [weak self] in
                guard let self else { return }
                do {
                    let descriptor = try await LSEENDModelDescriptor.loadFromHuggingFace(variant: variant)
                    DispatchQueue.main.async {
                        self.modelPath = descriptor.modelURL.path
                        self.metadataPath = descriptor.metadataURL.path
                        self.currentDescriptor = descriptor
                    }
                    self.processingQueue.async {
                        self.initializeProcessor(descriptor: descriptor)
                    }
                } catch {
                    DispatchQueue.main.async {
                        self.statusText = "Model load failed: \(error.localizedDescription)"
                        self.modelInfoText = ""
                    }
                }
            }
        }
    }

    private func initializeProcessor(descriptor: LSEENDModelDescriptor) {
        do {
            try self.processor.initialize(descriptor: descriptor)
            let numSpeakers = self.processor.numSpeakers ?? 0
            let latency = self.processor.streamingLatencySeconds ?? 0
            DispatchQueue.main.async {
                self.currentDescriptor = descriptor
                self.statusText = "Ready."
                self.modelInfoText = "\(descriptor.variant.rawValue) | \(numSpeakers) speaker tracks | \(String(format: "%.2f", latency)) s latency"
                self.resetTimelineUI(clearStatus: false, receivedSamples: 0)
            }
        } catch {
            DispatchQueue.main.async {
                self.statusText = "Model load failed: \(error.localizedDescription)"
                self.modelInfoText = ""
            }
        }
    }

    func selectVariant(_ variant: LSEENDVariant) {
        selectedVariant = variant
        applyVariantDefaults()
        reloadModel()
    }

    func browseModelPath() {
        if let url = chooseFile(allowedExtensions: ["mlpackage", "mlmodel", "mlmodelc"]) {
            modelPath = url.path
            useCustomPaths = true
        }
    }

    func browseMetadataPath() {
        if let url = chooseFile(allowedExtensions: ["json"]) {
            metadataPath = url.path
            useCustomPaths = true
        }
    }

    func browseSimulationFile() {
        if let url = chooseFile(allowedExtensions: ["wav", "flac", "mp3", "m4a", "aiff", "aif"]) {
            simulationPath = url.path
        }
    }

    func startMicrophone() {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            startMicrophoneCapture()
        case .notDetermined:
            statusText = "Requesting microphone access..."
            AVCaptureDevice.requestAccess(for: .audio) { [weak self] granted in
                DispatchQueue.main.async {
                    guard let self else { return }
                    if granted {
                        self.startMicrophoneCapture()
                    } else {
                        self.statusText = "Microphone access was denied."
                    }
                }
            }
        case .denied, .restricted:
            statusText = "Microphone access is unavailable. Enable it in System Settings and restart the app."
        @unknown default:
            statusText = "Microphone authorization status is unavailable."
        }
    }

    private func startMicrophoneCapture() {
        guard processor.isAvailable else {
            statusText = "Model is not loaded yet."
            return
        }
        guard let targetSampleRate = processor.targetSampleRate else { return }
        resetTimeline(clearStatus: false)
        processingQueue.async { [weak self] in
            guard let self else { return }
            do {
                self.processor.reset()
                let source = try LSEENDMicrophoneSource(
                    targetSampleRate: Double(targetSampleRate),
                    blockSeconds: self.blockSeconds,
                    onChunk: { [weak self] chunk in
                        self?.enqueueChunk(chunk)
                    },
                    onStatus: { [weak self] message in
                        DispatchQueue.main.async {
                            self?.statusText = message
                        }
                    }
                )
                self.audioSource = source
                try source.start()
                DispatchQueue.main.async {
                    self.sourceText = source.label
                    self.statusText = "Capture started."
                    self.updateBufferText(receivedSamples: self.totalSamplesReceived)
                }
            } catch {
                self.audioSource?.stop()
                self.audioSource = nil
                self.processor.reset()
                DispatchQueue.main.async {
                    self.statusText = "Failed to start microphone: \(error.localizedDescription)"
                }
            }
        }
    }

    func startSimulation() {
        guard processor.isAvailable else {
            statusText = "Model is not loaded yet."
            return
        }
        guard let targetSampleRate = processor.targetSampleRate else { return }
        guard !simulationPath.isEmpty else {
            statusText = "Choose an audio file to simulate."
            return
        }
        resetTimeline(clearStatus: false)
        processingQueue.async { [weak self] in
            guard let self else { return }
            do {
                self.processor.reset()
                let source = try LSEENDAudioFileSource(
                    fileURL: URL(fileURLWithPath: self.simulationPath),
                    targetSampleRate: targetSampleRate,
                    blockSeconds: self.blockSeconds,
                    speed: self.simulateSpeed,
                    onChunk: { [weak self] chunk in
                        self?.enqueueChunk(chunk)
                    },
                    onCompletion: { [weak self] in
                        self?.stopCapture(flush: true)
                    }
                )
                self.audioSource = source
                try source.start()
                DispatchQueue.main.async {
                    self.sourceText = source.label
                    self.statusText = "Simulation started."
                    self.updateBufferText(receivedSamples: self.totalSamplesReceived)
                }
            } catch {
                self.audioSource?.stop()
                self.audioSource = nil
                self.processor.reset()
                DispatchQueue.main.async {
                    self.statusText = "Failed to start simulation: \(error.localizedDescription)"
                }
            }
        }
    }

    func stopCapture(flush: Bool = true) {
        let source = audioSource
        audioSource = nil
        source?.stop()
        processingQueue.async { [weak self] in
            guard let self else { return }

            guard flush else {
                self.processor.reset()
                self.totalSamplesReceived = 0
                self.pendingAudioCount = 0
                DispatchQueue.main.async {
                    self.sourceText = "Not started"
                    self.updateBufferText(receivedSamples: 0)
                }
                return
            }

            guard self.processor.isAvailable else {
                DispatchQueue.main.async {
                    self.sourceText = "Not started"
                }
                return
            }

            do {
                // Flush pending audio and finalize
                let _ = try self.processor.finalizeSession()
                let timeline = self.processor.timeline
                let numSpeakers = self.processor.numSpeakers ?? 0
                DispatchQueue.main.async {
                    self.syncHeatmapFromTimeline(timeline, numSpeakers: numSpeakers)
                    self.previewProbabilities = .empty(columns: self.committedProbabilities.columns)
                    self.previewStartFrame = self.committedProbabilities.rows
                    self.inferenceText = "Inference: finalized"
                    self.statusText = "Streaming tail flushed."
                    self.sourceText = "Not started"
                    self.updateBufferText(receivedSamples: self.totalSamplesReceived)
                }
            } catch {
                self.processor.reset()
                DispatchQueue.main.async {
                    self.statusText = "Finalize failed: \(error.localizedDescription)"
                    self.sourceText = "Not started"
                }
            }
        }
    }

    func resetTimeline(clearStatus: Bool = true) {
        let source = audioSource
        audioSource = nil
        source?.stop()
        processingQueue.async { [weak self] in
            guard let self else { return }
            self.processor.reset()
            self.totalSamplesReceived = 0
            self.pendingAudioCount = 0
            DispatchQueue.main.async {
                self.sourceText = "Not started"
                self.resetTimelineUI(clearStatus: clearStatus, receivedSamples: 0)
            }
        }
    }

    func moveSpeaker(row: Int, offset: Int) {
        let target = row + offset
        guard row >= 0, row < displayOrder.count, target >= 0, target < displayOrder.count else {
            return
        }
        displayOrder.swapAt(row, target)
    }

    func resetSpeakerOrder() {
        displayOrder = Array(0..<speakerLabels.count)
    }

    func updateSpeakerLabel(row: Int, label: String) {
        guard row >= 0, row < displayOrder.count else { return }
        let trackIndex = displayOrder[row]
        guard trackIndex >= 0, trackIndex < speakerLabels.count else { return }
        speakerLabels[trackIndex] = label
    }

    func saveRTTM() {
        guard let frameRate = processor.modelFrameHz else { return }
        let binary = currentBinary()
        guard binary.rows > 0 else {
            statusText = "No inference available to save."
            return
        }
        do {
            try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
            let outputURL = outputDirectory.appendingPathComponent(nextOutputStem()).appendingPathExtension("rttm")
            try LSEENDEvaluation.writeRTTM(
                recordingID: outputURL.deletingPathExtension().lastPathComponent,
                binaryPrediction: reordered(matrix: binary),
                outputURL: outputURL,
                frameRate: frameRate,
                speakerLabels: speakerLabelsForDisplay()
            )
            statusText = "Saved RTTM: \(outputURL.lastPathComponent)"
        } catch {
            statusText = "Failed to save RTTM: \(error.localizedDescription)"
        }
    }

    func saveSessionJSON() {
        guard let frameRate = processor.modelFrameHz else { return }
        do {
            try FileManager.default.createDirectory(at: outputDirectory, withIntermediateDirectories: true)
            let outputURL = outputDirectory.appendingPathComponent(nextOutputStem()).appendingPathExtension("json")
            let payload: [String: Any] = [
                "variant": selectedVariant.rawValue,
                "coreml_model": modelPath,
                "metadata": metadataPath,
                "variant_stem": currentDescriptor?.variant.stem ?? "",
                "source": sourceText,
                "duration_seconds": Double(committedProbabilities.rows) / frameRate,
                "preview_duration_seconds": Double(previewProbabilities.rows) / frameRate,
                "frame_hz": frameRate,
                "display_order": displayOrder,
                "speaker_labels": speakerLabelsForDisplay(),
                "block_seconds": blockSeconds,
                "refresh_seconds": refreshSeconds,
                "threshold": threshold,
                "median": medianWidth,
                "streaming_latency_seconds": processor.streamingLatencySeconds ?? 0,
            ]
            let data = try JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: outputURL)
            statusText = "Saved session JSON: \(outputURL.lastPathComponent)"
        } catch {
            statusText = "Failed to save session JSON: \(error.localizedDescription)"
        }
    }

    private func enqueueChunk(_ chunk: [Float]) {
        processingQueue.async { [weak self] in
            guard let self, self.processor.isAvailable else { return }
            do {
                self.processor.addAudio(chunk)
                self.totalSamplesReceived += chunk.count
                self.pendingAudioCount += chunk.count

                // Respect refresh rate: only process when enough audio has accumulated
                let sampleRate = self.processor.targetSampleRate ?? 1
                let minimumDelta = max(1, Int(round(self.refreshSeconds * Double(max(sampleRate, 1)))))
                guard self.pendingAudioCount >= minimumDelta else {
                    DispatchQueue.main.async {
                        self.updateBufferText(receivedSamples: self.totalSamplesReceived)
                    }
                    return
                }
                self.pendingAudioCount = 0

                guard let result = try self.processor.process() else {
                    DispatchQueue.main.async {
                        self.updateBufferText(receivedSamples: self.totalSamplesReceived)
                    }
                    return
                }

                let numSpeakers = self.processor.numSpeakers ?? 0
                let totalEmitted = self.processor.numFramesProcessed
                DispatchQueue.main.async {
                    self.mergeChunkResult(result.chunkResult, numSpeakers: numSpeakers, totalEmittedFrames: totalEmitted)
                    self.updateBufferText(receivedSamples: self.totalSamplesReceived)
                }
            } catch {
                DispatchQueue.main.async {
                    self.statusText = "Inference failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func mergeChunkResult(_ result: DiarizerChunkResult, numSpeakers: Int, totalEmittedFrames: Int) {
        guard numSpeakers > 0 else { return }

        // Convert committed predictions to LSEENDMatrix
        let committedMatrix = LSEENDMatrix(
            validatingRows: result.finalizedFrameCount,
            columns: numSpeakers,
            values: result.finalizedPredictions
        )
        committedProbabilities = merge(
            existing: committedProbabilities,
            patch: committedMatrix,
            startFrame: result.startFrame
        )

        // Convert tentative predictions to LSEENDMatrix
        previewProbabilities = LSEENDMatrix(
            validatingRows: result.tentativeFrameCount,
            columns: numSpeakers,
            values: result.tentativePredictions
        )
        previewStartFrame = result.tentativeStartFrame

        ensureTrackState(trackCount: numSpeakers)
        inferenceText = "Inference: \(totalEmittedFrames) committed + \(result.tentativeFrameCount) preview frames"
        statusText = "Inference updated."
    }

    /// Sync heatmap from the full timeline (used after finalize)
    private func syncHeatmapFromTimeline(_ timeline: DiarizerTimeline, numSpeakers: Int) {
        guard numSpeakers > 0 else { return }
        committedProbabilities = LSEENDMatrix(
            validatingRows: timeline.numFrames,
            columns: numSpeakers,
            values: timeline.finalizedPredictions
        )
        ensureTrackState(trackCount: numSpeakers)
    }

    private func resetTimelineUI(clearStatus: Bool, receivedSamples: Int) {
        committedProbabilities = .empty(columns: 0)
        previewProbabilities = .empty(columns: 0)
        previewStartFrame = 0
        speakerLabels = []
        displayOrder = []
        inferenceText = "Inference: idle"
        updateBufferText(receivedSamples: receivedSamples)
        if clearStatus {
            statusText = "Timeline reset."
        }
    }

    private func merge(existing: LSEENDMatrix, patch: LSEENDMatrix, startFrame: Int) -> LSEENDMatrix {
        guard patch.rows > 0 else { return existing }
        let endFrame = startFrame + patch.rows
        let rows = max(existing.rows, endFrame)
        let columns = max(existing.columns, patch.columns)
        var output = LSEENDMatrix.zeros(rows: rows, columns: columns)
        for rowIndex in 0..<existing.rows {
            for columnIndex in 0..<existing.columns {
                output[rowIndex, columnIndex] = existing[rowIndex, columnIndex]
            }
        }
        for rowIndex in 0..<patch.rows {
            for columnIndex in 0..<patch.columns {
                output[startFrame + rowIndex, columnIndex] = patch[rowIndex, columnIndex]
            }
        }
        return output
    }

    private func ensureTrackState(trackCount: Int) {
        guard trackCount > 0 else { return }
        if speakerLabels.count != trackCount {
            let oldLabels = speakerLabels
            speakerLabels = (0..<trackCount).map { index in
                index < oldLabels.count ? oldLabels[index] : "Speaker \(index + 1)"
            }
        }
        if displayOrder.count != trackCount || Set(displayOrder) != Set(0..<trackCount) {
            displayOrder = Array(0..<trackCount)
        }
    }

    private func combinedProbabilities() -> LSEENDMatrix {
        let rows = max(committedProbabilities.rows, previewStartFrame + previewProbabilities.rows)
        let columns = max(committedProbabilities.columns, previewProbabilities.columns)
        guard rows > 0, columns > 0 else {
            return .empty(columns: 0)
        }
        var combined = LSEENDMatrix.zeros(rows: rows, columns: columns)
        for rowIndex in 0..<committedProbabilities.rows {
            for columnIndex in 0..<committedProbabilities.columns {
                combined[rowIndex, columnIndex] = committedProbabilities[rowIndex, columnIndex]
            }
        }
        for rowIndex in 0..<previewProbabilities.rows {
            for columnIndex in 0..<previewProbabilities.columns {
                combined[previewStartFrame + rowIndex, columnIndex] = previewProbabilities[rowIndex, columnIndex]
            }
        }
        return combined
    }

    private func reordered(matrix: LSEENDMatrix) -> LSEENDMatrix {
        guard matrix.rows > 0, matrix.columns > 0, !displayOrder.isEmpty else {
            return matrix
        }
        var output = LSEENDMatrix.zeros(rows: matrix.rows, columns: displayOrder.count)
        for rowIndex in 0..<matrix.rows {
            for (destinationColumn, sourceColumn) in displayOrder.enumerated() {
                output[rowIndex, destinationColumn] = matrix[rowIndex, sourceColumn]
            }
        }
        return output
    }

    private func currentBinary() -> LSEENDMatrix {
        let probabilities = combinedProbabilities()
        guard probabilities.rows > 0 else { return probabilities }
        return LSEENDEvaluation.medianFilter(
            binary: LSEENDEvaluation.threshold(probabilities: probabilities, value: Float(threshold)),
            width: medianWidth
        )
    }

    private func speakerLabelsForDisplay() -> [String] {
        guard !displayOrder.isEmpty else { return speakerLabels }
        return displayOrder.map { index in
            let label = speakerLabels[index].trimmingCharacters(in: .whitespacesAndNewlines)
            return label.isEmpty ? "Speaker \(index + 1)" : label
        }
    }

    private func updateBufferText(receivedSamples: Int) {
        let sampleRate = processor.targetSampleRate ?? 1
        let frameRate = processor.modelFrameHz ?? 10
        let receivedSeconds = Double(receivedSamples) / Double(max(sampleRate, 1))
        let committedSeconds = Double(committedProbabilities.rows) / frameRate
        let previewSeconds = Double(max(committedProbabilities.rows, previewStartFrame + previewProbabilities.rows)) / frameRate
        bufferText = String(format: "Buffered: %.1f s received, %.1f s committed, %.1f s incl preview", receivedSeconds, committedSeconds, previewSeconds)
    }

    private func previewRangeInDisplayedMatrix(totalRows: Int, startFrame: Int) -> (start: Double?, end: Double?) {
        guard let frameRate = processor.modelFrameHz, previewProbabilities.rows > 0 else {
            return (nil, nil)
        }
        let previewStart = max(previewStartFrame, startFrame)
        let previewEnd = min(totalRows, previewStartFrame + previewProbabilities.rows)
        guard previewEnd > previewStart else {
            return (nil, nil)
        }
        return (
            Double(previewStart) / frameRate,
            Double(previewEnd) / frameRate
        )
    }

    private func nextOutputStem() -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        return "mic_session_\(formatter.string(from: Date()))"
    }

    private func descriptorFromCurrentSelection() -> LSEENDModelDescriptor {
        LSEENDModelDescriptor(
            variant: selectedVariant,
            modelURL: URL(fileURLWithPath: modelPath),
            metadataURL: URL(fileURLWithPath: metadataPath)
        )
    }

    private func chooseFile(allowedExtensions: [String]) -> URL? {
        let panel = NSOpenPanel()
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = allowedExtensions.contains("mlmodelc")
        panel.canChooseFiles = true
        panel.allowedContentTypes = []
        panel.allowedFileTypes = allowedExtensions
        return panel.runModal() == .OK ? panel.url : nil
    }
}
