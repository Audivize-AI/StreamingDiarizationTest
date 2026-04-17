//
//  DiarizerViewModel.swift
//  LS-EEND-Test
//

import AVFoundation
import Combine
import CoreML
import Foundation
import SwiftUI

@MainActor
final class DiarizerViewModel: ObservableObject {

    enum LoadState: Equatable {
        case idle
        case loading(progress: Double)
        case ready
        case failed(String)
    }

    @Published var loadState: LoadState = .idle
    @Published var variant: LSEENDVariant = ModelNames.LSEEND.defaultVariant
    @Published var stepSize: LSEENDStepSize = ModelNames.LSEEND.defaultStep

    @Published var isProcessing: Bool = false
    @Published var progress: Double = 0
    @Published var statusMessage: String = ""

    // Snapshots used by the UI. Copied out of the DiarizerTimeline on the
    // main actor after every process() tick so SwiftUI can diff safely.
    @Published var segments: [DiarizerSegment] = []
    @Published var speakers: [Int: String?] = [:]
    @Published var finalizedPredictions: [Float] = []
    @Published var tentativePredictions: [Float] = []
    @Published var numSpeakerChannels: Int = 0
    @Published var frameDurationSeconds: Float = 0.1
    @Published var finalizedFrameCount: Int = 0
    @Published var tentativeFrameCount: Int = 0

    private var diarizer: LSEENDDiarizer?
    private let queue = DispatchQueue(label: "lseend.diarizer.serial", qos: .userInitiated)
    private var micEngine: AVAudioEngine?
    private var micTapFormat: AVAudioFormat?
    private var micSourceRate: Double = 0
    private lazy var mainActorSink: MainActorSink = MainActorSink(owner: self)

    init() {
        Task { await reload() }
    }

    // MARK: - Model lifecycle

    func reload(variant: LSEENDVariant? = nil, stepSize: LSEENDStepSize? = nil) async {
        if let v = variant { self.variant = v }
        if let s = stepSize { self.stepSize = s }

        print("[DiarizerVM] reload variant=\(self.variant) step=\(self.stepSize.suffix)")
        await stopMicrophone()
        loadState = .loading(progress: 0)
        statusMessage = "Loading \(self.variant.name) @ \(self.stepSize.suffix)…"

        do {
            let d: LSEENDDiarizer
            let v = self.variant, s = self.stepSize
            // Prefer local coreml/out/ packages — HF uploads are stale for
            // non-dih3 variants. HF load triggers on local-miss only.
            let localURL = Self.localOutURL(variant: v, stepSize: s)
            print("[DiarizerVM] local package path: \(localURL?.path ?? "<none>") exists=\(Self.localOutExists(variant: v, stepSize: s))")
            if Self.localOutExists(variant: v, stepSize: s) {
                d = try await Self.loadFromLocalOut(variant: v, stepSize: s)
                print("[DiarizerVM] loaded locally")
            } else {
                d = try await LSEENDDiarizer.loadFromHuggingFace(
                    variant: v,
                    stepSize: s,
                    computeUnits: .cpuOnly,
                    progressHandler: { [weak self] p in
                        let frac = p.fractionCompleted
                        Task { @MainActor [weak self] in
                            self?.loadState = .loading(progress: frac)
                        }
                    }
                )
            }
            self.diarizer = d
            self.numSpeakerChannels = d.numSpeakers ?? 0
            self.frameDurationSeconds = d.timeline.config.frameDurationSeconds
            self.loadState = .ready
            self.statusMessage = "Ready."
            clearSnapshot()
        } catch {
            loadState = .failed("\(error)")
            statusMessage = "Load failed: \(error.localizedDescription)"
        }
    }

    private func clearSnapshot() {
        segments = []
        speakers = [:]
        finalizedPredictions = []
        tentativePredictions = []
        finalizedFrameCount = 0
        tentativeFrameCount = 0
    }

    // MARK: - File processing

    func processFile(_ url: URL) async {
        guard let diarizer = diarizer else {
            statusMessage = "Model not ready yet."
            return
        }
        isProcessing = true
        progress = 0
        statusMessage = "Processing \(url.lastPathComponent)…"
        clearSnapshot()

        let capturedDiarizer = diarizer
        let didStartAccess = url.startAccessingSecurityScopedResource()
        let sink = self.mainActorSink
        print("[DiarizerVM] processFile start url=\(url.path) scoped=\(didStartAccess)")
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                defer {
                    if didStartAccess { url.stopAccessingSecurityScopedResource() }
                    cont.resume()
                    print("[DiarizerVM] processFile queue defer complete")
                }
                do {
                    print("[DiarizerVM] calling processComplete…")
                    let timeline = try capturedDiarizer.processComplete(
                        audioFileURL: url,
                        keepingEnrolledSpeakers: nil,
                        finalizeOnCompletion: true,
                        progressCallback: { done, total, _ in
                            let p = total > 0 ? Double(done) / Double(total) : 0
                            print("[DiarizerVM] progress \(done)/\(total) (\(Int(p*100))%)")
                            sink.postProgress(p)
                        }
                    )
                    print("[DiarizerVM] processComplete OK frames=\(timeline.numFinalizedFrames) speakers=\(timeline.speakers.count)")
                    sink.postDone()
                } catch {
                    print("[DiarizerVM] processComplete FAILED: \(error)")
                    sink.postFail(error.localizedDescription)
                }
            }
        }
    }

    // MARK: - Microphone

    func startMicrophone() async throws {
        print("[DiarizerVM] startMicrophone entry")
        guard let diarizer = diarizer else {
            print("[DiarizerVM] startMicrophone: diarizer is nil")
            return
        }
        guard micEngine == nil else {
            print("[DiarizerVM] startMicrophone: engine already running")
            return
        }

        let granted = await AVCaptureDevice.requestAccess(for: .audio)
        print("[DiarizerVM] mic permission granted=\(granted)")
        guard granted else {
            statusMessage = "Microphone permission denied."
            return
        }

        let capturedDiarizer = diarizer
        // Serialize the reset onto the same queue used for process() so a
        // still-running process() can't race with reset() mid-call.
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                capturedDiarizer.reset()
                cont.resume()
            }
        }
        clearSnapshot()

        let engine = AVAudioEngine()
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0 else {
            statusMessage = "Mic unavailable: zero sample rate."
            return
        }
        micSourceRate = format.sampleRate
        micTapFormat = format

        let capturedRate = format.sampleRate
        let capturedQueue = queue
        // Swift 6's runtime isolation checks crash when a MainActor-isolated
        // class is referenced (even weakly) from an AVAudioEngine tap's
        // real-time thread. Route UI updates through a nonisolated dispatch
        // sink that doesn't know about MainActor at all.
        let sink = self.mainActorSink
        print("[DiarizerVM] installTap nativeFormat=\(input.inputFormat(forBus: 0)) tapFormat=nil")
        Self.installMicTap(
            on: input,
            diarizer: capturedDiarizer,
            queue: capturedQueue,
            sampleRate: capturedRate,
            sink: sink
        )

        engine.prepare()
        do {
            try engine.start()
        } catch {
            print("[DiarizerVM] engine.start() threw: \(error)")
            statusMessage = "Engine start failed: \(error.localizedDescription)"
            return
        }
        self.micEngine = engine
        self.statusMessage = "Microphone active (\(Int(format.sampleRate)) Hz)."
        print("[DiarizerVM] engine started sampleRate=\(format.sampleRate)")
    }

    func stopMicrophone() async {
        guard let engine = micEngine else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        micEngine = nil

        if let diarizer = diarizer {
            let captured = diarizer
            let sink = self.mainActorSink
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                queue.async {
                    _ = try? captured.finalize()
                    sink.postSnapshot()
                    sink.postStatus("Microphone stopped.")
                    cont.resume()
                }
            }
        }
    }

    // MARK: - Enrollment

    func enroll(name: String, from url: URL) async {
        guard let diarizer = diarizer else { return }
        statusMessage = "Enrolling \(name)…"
        let captured = diarizer
        let didStartAccess = url.startAccessingSecurityScopedResource()
        let sink = self.mainActorSink
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                defer {
                    if didStartAccess { url.stopAccessingSecurityScopedResource() }
                    cont.resume()
                }
                do {
                    let samples = AVAudioFile_safeRead(url: url)
                    let speaker = try captured.enrollSpeaker(
                        withAudio: samples,
                        sourceSampleRate: nil,
                        named: name,
                        overwritingAssignedSpeakerName: true
                    )
                    let desc = speaker?.description ?? name
                    sink.postSnapshot()
                    sink.postStatus("Enrolled \(desc).")
                } catch {
                    sink.postStatus("Enrollment failed: \(error.localizedDescription)")
                }
            }
        }
    }

    // MARK: - Mic tap (nonisolated helper)

    /// nonisolated `static` so the installed tap closure inherits no actor
    /// isolation from the caller. Without this, Swift 6 infers the tap
    /// closure as MainActor-isolated (because startMicrophone is @MainActor)
    /// and the real-time audio thread trips `_swift_task_checkIsolatedSwift`
    /// at every invocation → SIGTRAP.
    nonisolated static func installMicTap(
        on input: AVAudioInputNode,
        diarizer: LSEENDDiarizer,
        queue: DispatchQueue,
        sampleRate: Double,
        sink: MainActorSink
    ) {
        let counter = TapCounter()
        input.installTap(onBus: 0, bufferSize: 4096, format: nil) { buffer, _ in
            let n = counter.inc()
            let frameCount = Int(buffer.frameLength)
            if n <= 3 {
                print("[DiarizerVM][tap#\(n)] frames=\(frameCount) format=\(buffer.format)")
            }
            guard frameCount > 0 else { return }
            let samples: [Float]
            if let floatChannels = buffer.floatChannelData {
                samples = Array(UnsafeBufferPointer(start: floatChannels[0], count: frameCount))
            } else if let int16Channels = buffer.int16ChannelData {
                let raw = UnsafeBufferPointer(start: int16Channels[0], count: frameCount)
                samples = raw.map { Float($0) / 32768.0 }
            } else {
                if n <= 3 { print("[DiarizerVM][tap] unsupported sample format") }
                return
            }
            queue.async {
                do {
                    _ = try diarizer.process(samples: samples, sourceSampleRate: sampleRate)
                    sink.postSnapshot()
                } catch {
                    print("[DiarizerVM][tap process] error \(error)")
                    sink.postStatus("Mic error: \(error)")
                }
            }
        }
    }

    // MARK: - Local-model fallback

    /// Development-only local package dir. Resolved at runtime from:
    ///   1. `LSEEND_LOCAL_MODELS_DIR` environment variable
    ///   2. User-selected bookmark persisted under
    ///      `localModelsBookmark` in `UserDefaults`
    /// Returns nil when neither is set — HF path is used instead.
    /// No absolute user paths are baked into the binary or entitlements.
    private static func localOutDir() -> URL? {
        if let envPath = ProcessInfo.processInfo.environment["LSEEND_LOCAL_MODELS_DIR"],
           !envPath.isEmpty {
            return URL(fileURLWithPath: envPath, isDirectory: true)
        }
        let defaults = UserDefaults.standard
        if let bookmark = defaults.data(forKey: "localModelsBookmark") {
            var stale = false
            if let url = try? URL(
                resolvingBookmarkData: bookmark,
                options: [.withSecurityScope],
                relativeTo: nil,
                bookmarkDataIsStale: &stale
            ) {
                return url
            }
        }
        return nil
    }

    private static func localOutURL(
        variant: LSEENDVariant, stepSize: LSEENDStepSize
    ) -> URL? {
        guard let dir = localOutDir() else { return nil }
        // Build path without appendingPathComponent — on macOS 26 that
        // auto-stats and appends a trailing "/" for directories, which
        // MLModel.compileModel rejects with "Input stream is not valid".
        let pkgName = "\(variant.name)_\(stepSize.suffix).mlpackage"
        return URL(
            fileURLWithPath: dir.path + "/" + pkgName,
            isDirectory: false
        )
    }

    private static func localOutExists(
        variant: LSEENDVariant, stepSize: LSEENDStepSize
    ) -> Bool {
        guard let url = localOutURL(variant: variant, stepSize: stepSize) else {
            return false
        }
        return FileManager.default.fileExists(atPath: url.path)
    }

    private static func loadFromLocalOut(
        variant: LSEENDVariant, stepSize: LSEENDStepSize
    ) async throws -> LSEENDDiarizer {
        guard let pkgURL = localOutURL(variant: variant, stepSize: stepSize) else {
            throw NSError(
                domain: "LSEEND.LocalFallback", code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "No local models dir configured. Set LSEEND_LOCAL_MODELS_DIR or pick a folder."]
            )
        }
        let needsScope = pkgURL.startAccessingSecurityScopedResource()
        defer { if needsScope { pkgURL.stopAccessingSecurityScopedResource() } }
        let compiled = try await MLModel.compileModel(at: pkgURL)
        let model = try LSEENDModel(modelURL: compiled, computeUnits: .cpuOnly)
        return try LSEENDDiarizer(model: model)
    }

    /// Persist a user-selected models directory as a security-scoped
    /// bookmark. Called from UI after a folder picker.
    func setLocalModelsDirectory(_ url: URL) {
        do {
            let data = try url.bookmarkData(
                options: [.withSecurityScope],
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
            UserDefaults.standard.set(data, forKey: "localModelsBookmark")
            statusMessage = "Local models: \(url.path)"
        } catch {
            statusMessage = "Bookmark failed: \(error.localizedDescription)"
        }
    }

    func renameSpeaker(slot: Int, to name: String) {
        guard let diarizer = diarizer else { return }
        _ = diarizer.timeline.upsertSpeaker(named: name.isEmpty ? nil : name, atIndex: slot)
        snapshotTimeline()
    }

    func reset() {
        guard let diarizer = diarizer else { return }
        // Serialize onto the same queue that owns inference.
        let captured = diarizer
        queue.async { captured.reset() }
        clearSnapshot()
    }

    // MARK: - Snapshot

    /// Public so MainActorSink can drive it from the dispatch-main hop.
    /// Must only be called on the main thread.
    func refreshSnapshotFromMainSink() {
        snapshotTimeline()
    }

    private func snapshotTimeline() {
        guard let d = diarizer else { return }
        let tl = d.timeline
        self.finalizedPredictions = tl.finalizedPredictions
        self.tentativePredictions = tl.tentativePredictions
        self.finalizedFrameCount = tl.numFinalizedFrames
        self.tentativeFrameCount = tl.numTentativeFrames
        self.frameDurationSeconds = tl.config.frameDurationSeconds
        self.numSpeakerChannels = tl.speakerCapacity
        self.speakers = Dictionary(uniqueKeysWithValues: tl.speakers.map { ($0.key, $0.value.name) })

        var allSegs: [DiarizerSegment] = []
        for (_, sp) in tl.speakers {
            allSegs.append(contentsOf: sp.finalizedSegments)
            allSegs.append(contentsOf: sp.tentativeSegments)
        }
        allSegs.sort()
        self.segments = allSegs
    }
}

/// Helper: load an audio file into `[Float]` at the target sample rate via
/// `AudioConverter`. Kept out-of-line so the ViewModel stays readable.
private func AVAudioFile_safeRead(url: URL) -> [Float] {
    let converter = AudioConverter(sampleRate: 8000)
    return (try? converter.resampleAudioFile(url)) ?? []
}

/// Fully nonisolated sink that receives update requests from real-time /
/// background threads and bounces them to the MainActor without ever
/// capturing a MainActor-isolated reference in a `@Sendable` boundary.
///
/// Swift 6's runtime isolation check crashes when a `Task { @MainActor in }`
/// is created while the enclosing context captures a MainActor-isolated
/// `self` (even weakly). The tap closure on AVAudioEngine runs on a
/// real-time thread, so any self-weak capture there triggers the crash.
/// This sink solves that by holding an `@unchecked Sendable` weak owner and
/// posting via `DispatchQueue.main.async`, which uses a dispatch hop — not
/// an actor hop — and bypasses the runtime isolation check.
final class MainActorSink: @unchecked Sendable {
    private weak var owner: DiarizerViewModel?
    init(owner: DiarizerViewModel) { self.owner = owner }

    func postSnapshot() {
        DispatchQueue.main.async { [weak self] in
            self?.owner?.objectWillChange.send()
            self?.owner?.refreshSnapshotFromMainSink()
        }
    }

    func postStatus(_ msg: String) {
        DispatchQueue.main.async { [weak self] in
            self?.owner?.statusMessage = msg
        }
    }

    func postProgress(_ p: Double) {
        DispatchQueue.main.async { [weak self] in
            self?.owner?.progress = p
            self?.owner?.refreshSnapshotFromMainSink()
        }
    }

    func postDone() {
        DispatchQueue.main.async { [weak self] in
            self?.owner?.refreshSnapshotFromMainSink()
            self?.owner?.isProcessing = false
            self?.owner?.progress = 1
            self?.owner?.statusMessage = "Done."
        }
    }

    func postFail(_ msg: String) {
        DispatchQueue.main.async { [weak self] in
            self?.owner?.isProcessing = false
            self?.owner?.statusMessage = "Failed: \(msg)"
        }
    }
}

/// Thread-safe counter for real-time callbacks to increment without mutating
/// a captured Int (which tap closures can't do).
final class TapCounter: @unchecked Sendable {
    private var n: Int = 0
    private let lock = NSLock()
    func inc() -> Int {
        lock.lock(); defer { lock.unlock() }
        n += 1
        return n
    }
}
