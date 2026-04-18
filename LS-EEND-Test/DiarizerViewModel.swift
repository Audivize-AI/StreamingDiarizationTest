//
//  DiarizerViewModel.swift
//  LS-EEND-Test
//

import AVFoundation
import Combine
import CoreML
import Foundation
import SwiftUI

/// File-scope logger so nonisolated contexts (the mic tap closure, the
/// serial inference queue) can log without a MainActor hop. `AppLogger` is
/// `Sendable`, so a plain `let` at file scope is safe.
private let log = AppLogger(category: "DiarizerViewModel")

@MainActor
final class DiarizerViewModel: ObservableObject {

    enum LoadState: Equatable {
        case idle
        case loading(progress: Double)
        case ready
        case failed(String)
    }

    /// Events posted from off-main contexts (AVAudioEngine tap thread, the
    /// serial inference queue) and drained on MainActor via an `AsyncStream`.
    ///
    /// Replaces the previous `MainActorSink` + `DispatchQueue.main.async`
    /// bridge. The tap closure never spawns a `Task`; it only calls
    /// `continuation.yield(...)` which is `@Sendable` + nonisolated and so
    /// doesn't trip Swift 6's `_swift_task_checkIsolatedSwift` on the
    /// real-time audio thread. The consumer `Task` is created from this
    /// `@MainActor init`, so it inherits MainActor isolation already — no
    /// `Task { @MainActor in }` indirection from the RT thread.
    private enum UIEvent: Sendable {
        case snapshot
        case progress(Double)
        case status(String)
        case done
        case failed(String)
    }

    @Published var loadState: LoadState = .idle
    @Published var variant: LSEENDVariant = ModelNames.LSEEND.defaultVariant
    @Published var stepSize: LSEENDStepSize = ModelNames.LSEEND.defaultStep

    @Published var isProcessing: Bool = false
    @Published var progress: Double = 0
    @Published var statusMessage: String = ""

    // Snapshots copied out of the DiarizerTimeline on the main actor after
    // every process() tick so SwiftUI can diff safely.
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

    // Continuations and tasks must be reachable from `deinit`, which runs
    // on whatever thread releases the last reference — not guaranteed to
    // be MainActor. `nonisolated let` on the continuation keeps Swift 6
    // strict concurrency happy (it's Sendable). `eventTask` is assigned
    // once inside `init` after all other properties are initialized, so it
    // needs `nonisolated(unsafe) var` — Task is Sendable but the compiler
    // can't prove definite-assignment before self escapes into the task's
    // `[weak self]` capture without the `var` indirection.
    private nonisolated let eventContinuation: AsyncStream<UIEvent>.Continuation
    private nonisolated(unsafe) var eventTask: Task<Void, Never>?

    init(skipAutoLoad: Bool = false) {
        let (stream, cont) = AsyncStream<UIEvent>.makeStream(bufferingPolicy: .unbounded)
        self.eventContinuation = cont
        self.eventTask = nil   // definite-assignment — replaced immediately
        // Task inherits MainActor isolation from this @MainActor init — so
        // the consumer loop runs on the main actor without re-hopping.
        self.eventTask = Task { [weak self] in
            for await event in stream {
                guard let self else { break }
                self.apply(event)
            }
        }
        if !skipAutoLoad {
            Task { await self.reload() }
        }
    }

    deinit {
        eventContinuation.finish()
        eventTask?.cancel()
    }

    private func apply(_ event: UIEvent) {
        switch event {
        case .snapshot:
            snapshotTimeline()
        case .progress(let p):
            progress = p
            snapshotTimeline()
        case .status(let msg):
            statusMessage = msg
        case .done:
            snapshotTimeline()
            isProcessing = false
            progress = 1
            statusMessage = "Done."
        case .failed(let msg):
            isProcessing = false
            statusMessage = "Failed: \(msg)"
        }
    }

    // MARK: - Model lifecycle

    func reload(variant: LSEENDVariant? = nil, stepSize: LSEENDStepSize? = nil) async {
        if let v = variant { self.variant = v }
        if let s = stepSize { self.stepSize = s }

        log.debug("reload variant=\(self.variant) step=\(self.stepSize.description)")
        await stopMicrophone()
        loadState = .loading(progress: 0)
        statusMessage = "Loading \(self.variant.name) @ \(self.stepSize.description)…"

        do {
            let d: LSEENDDiarizer
            let v = self.variant, s = self.stepSize
            if Self.localOutExists(variant: v, stepSize: s) {
                log.debug("using local package for \(v.name)/\(s.description)")
                d = try await Self.loadFromLocalOut(variant: v, stepSize: s)
            } else {
                log.debug("local miss — HF fallback for \(v.name)/\(s.description)")
                let model = try await LSEENDModel.loadFromHuggingFace(
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
                d = try LSEENDDiarizer(model: model)
            }
            self.diarizer = d
            self.numSpeakerChannels = d.numSpeakers ?? 0
            self.frameDurationSeconds = d.timeline.config.frameDurationSeconds
            self.loadState = .ready
            self.statusMessage = "Ready."
            clearSnapshot()
        } catch {
            log.error("reload failed: \(error.localizedDescription)")
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
        let events = self.eventContinuation
        log.debug("processFile start file=\(url.lastPathComponent) scoped=\(didStartAccess)")

        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                defer {
                    if didStartAccess { url.stopAccessingSecurityScopedResource() }
                    cont.resume()
                }
                do {
                    _ = try capturedDiarizer.processComplete(
                        audioFileURL: url,
                        keepingEnrolledSpeakers: nil,
                        finalizeOnCompletion: true,
                        progressCallback: { done, total, _ in
                            let p = total > 0 ? Double(done) / Double(total) : 0
                            events.yield(.progress(p))
                        }
                    )
                    events.yield(.done)
                } catch {
                    log.error("processComplete failed: \(error.localizedDescription)")
                    events.yield(.failed(error.localizedDescription))
                }
            }
        }
    }

    // MARK: - Microphone

    func startMicrophone() async throws {
        guard let diarizer = diarizer else {
            throw NSError(
                domain: "LSEEND.Mic", code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Model not ready."]
            )
        }
        guard micEngine == nil else { return }

        let granted = await AVCaptureDevice.requestAccess(for: .audio)
        guard granted else {
            statusMessage = "Microphone permission denied."
            throw NSError(
                domain: "LSEEND.Mic", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Microphone permission denied."]
            )
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
            throw NSError(
                domain: "LSEEND.Mic", code: 3,
                userInfo: [NSLocalizedDescriptionKey: "Microphone unavailable."]
            )
        }
        micSourceRate = format.sampleRate
        micTapFormat = format

        Self.installMicTap(
            on: input,
            diarizer: capturedDiarizer,
            queue: queue,
            sampleRate: format.sampleRate,
            events: eventContinuation
        )

        engine.prepare()
        do {
            try engine.start()
        } catch {
            log.error("engine.start() threw: \(error.localizedDescription)")
            statusMessage = "Engine start failed: \(error.localizedDescription)"
            input.removeTap(onBus: 0)
            throw error
        }
        self.micEngine = engine
        self.statusMessage = "Microphone active (\(Int(format.sampleRate)) Hz)."
        log.debug("engine started sampleRate=\(format.sampleRate)")
    }

    /// `removeTap` must run before the engine is released — otherwise a tap
    /// fired on the RT thread could still reach the continuation after
    /// `deinit` called `finish()` on it.
    func stopMicrophone() async {
        guard let engine = micEngine else { return }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        micEngine = nil

        if let diarizer = diarizer {
            let captured = diarizer
            let events = self.eventContinuation
            await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
                queue.async {
                    _ = try? captured.finalize()
                    events.yield(.snapshot)
                    events.yield(.status("Microphone stopped."))
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
        let events = self.eventContinuation
        await withCheckedContinuation { (cont: CheckedContinuation<Void, Never>) in
            queue.async {
                defer {
                    if didStartAccess { url.stopAccessingSecurityScopedResource() }
                    cont.resume()
                }
                let samples: [Float]
                do {
                    samples = try AudioConverter(sampleRate: 8000)
                        .resampleAudioFile(url)
                } catch {
                    events.yield(.status(
                        "Enrollment failed: audio unreadable — \(error.localizedDescription)"
                    ))
                    return
                }
                do {
                    let speaker = try captured.enrollSpeaker(
                        withAudio: samples,
                        sourceSampleRate: nil,
                        named: name,
                        overwritingAssignedSpeakerName: true
                    )
                    let desc = speaker?.description ?? name
                    events.yield(.snapshot)
                    events.yield(.status("Enrolled \(desc)."))
                } catch {
                    events.yield(.status(
                        "Enrollment failed: \(error.localizedDescription)"
                    ))
                }
            }
        }
    }

    // MARK: - Mic tap (nonisolated helper)

    /// `nonisolated static` so the installed tap closure inherits no actor
    /// isolation. The closure only calls `continuation.yield(...)` — which is
    /// `@Sendable` and nonisolated — so the RT thread never triggers Swift
    /// 6's runtime isolation check. No `Task { ... }` is created from here.
    private nonisolated static func installMicTap(
        on input: AVAudioInputNode,
        diarizer: LSEENDDiarizer,
        queue: DispatchQueue,
        sampleRate: Double,
        events: AsyncStream<UIEvent>.Continuation
    ) {
        let counter = TapCounter()
        input.installTap(onBus: 0, bufferSize: 4096, format: nil) { buffer, _ in
            let n = counter.inc()
            let frameCount = Int(buffer.frameLength)
            guard frameCount > 0, buffer.format.channelCount > 0 else { return }
            let samples: [Float]
            if let floatChannels = buffer.floatChannelData {
                samples = Array(UnsafeBufferPointer(start: floatChannels[0], count: frameCount))
            } else if let int16Channels = buffer.int16ChannelData {
                let raw = UnsafeBufferPointer(start: int16Channels[0], count: frameCount)
                samples = raw.map { Float($0) / 32768.0 }
            } else {
                if n <= 3 { log.warning("tap: unsupported sample format") }
                return
            }
            queue.async {
                do {
                    _ = try diarizer.process(samples: samples, sourceSampleRate: sampleRate)
                    events.yield(.snapshot)
                } catch {
                    // `\(error)` exposes enum-case associated values (e.g.
                    // `.invalidInputSize("mel features … 14490 != 13455")`);
                    // `localizedDescription` only shows the generic NSError
                    // string because LSEENDError's LocalizedError impl is
                    // incomplete.
                    log.error("tap process error: \(error)")
                    events.yield(.status("Mic error: \(error)"))
                }
            }
        }
    }

    // MARK: - Local-model fallback

    /// Development-only local package dir. Resolved at runtime from:
    ///   1. `LSEEND_LOCAL_MODELS_DIR` environment variable
    ///   2. User-selected bookmark persisted under
    ///      `localModelsBookmark` in `UserDefaults`
    /// Returns nil when neither is set/valid — HF path is used instead.
    /// No absolute user paths are baked into the binary or entitlements.
    private static func localOutDir() -> URL? {
        if let envPath = ProcessInfo.processInfo.environment["LSEEND_LOCAL_MODELS_DIR"],
           !envPath.isEmpty {
            // Env var is user-controlled (Xcode scheme / shell). Reject
            // anything that isn't an absolute, canonical, existing directory
            // before handing it to `URL` or touching disk.
            guard envPath.hasPrefix("/") else {
                log.warning("LSEEND_LOCAL_MODELS_DIR rejected: path is not absolute")
                return nil
            }
            let resolved = URL(fileURLWithPath: envPath, isDirectory: true)
                .standardizedFileURL
            guard isDirectory(resolved) else {
                log.warning("LSEEND_LOCAL_MODELS_DIR rejected: not a directory")
                return nil
            }
            return resolved
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
                if stale {
                    defaults.removeObject(forKey: "localModelsBookmark")
                    log.notice("stale models bookmark dropped — re-select folder")
                    return nil
                }
                return url
            }
        }
        return nil
    }

    private static func isDirectory(_ url: URL) -> Bool {
        (try? url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
    }

    private static func localOutURL(
        variant: LSEENDVariant, stepSize: LSEENDStepSize
    ) -> URL? {
        guard let dir = localOutDir() else { return nil }
        // Build path without appendingPathComponent — on macOS 26 that
        // auto-stats and appends a trailing "/" for directories, which
        // MLModel.compileModel rejects with "Input stream is not valid".
        let pkgName = "\(variant.name)_\(stepSize.description).mlpackage"
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
        guard let dir = localOutDir() else {
            throw NSError(
                domain: "LSEEND.LocalFallback", code: 1,
                userInfo: [NSLocalizedDescriptionKey:
                    "No local models dir configured. Set LSEEND_LOCAL_MODELS_DIR or pick a folder."]
            )
        }
        // Scope the *directory* bookmark (not the package) so access stays
        // open across compile + load and is released when we return.
        let needsScope = dir.startAccessingSecurityScopedResource()
        defer { if needsScope { dir.stopAccessingSecurityScopedResource() } }

        let pkgName = "\(variant.name)_\(stepSize.description).mlpackage"
        let pkgURL = URL(fileURLWithPath: dir.path + "/" + pkgName, isDirectory: false)
        guard FileManager.default.fileExists(atPath: pkgURL.path) else {
            throw NSError(
                domain: "LSEEND.LocalFallback", code: 2,
                userInfo: [NSLocalizedDescriptionKey: "Missing local package: \(pkgName)"]
            )
        }
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
            // Display folder name only — don't leak full path to UI/logs.
            statusMessage = "Local models: \(url.lastPathComponent)"
        } catch {
            statusMessage = "Bookmark failed: \(error.localizedDescription)"
        }
    }

    func renameSpeaker(slot: Int, to name: String) {
        guard let diarizer = diarizer else { return }
        _ = diarizer.timeline.upsertSpeaker(
            named: name.isEmpty ? nil : name, atIndex: slot
        )
        snapshotTimeline()
    }

    func reset() {
        guard let diarizer = diarizer else {
            clearSnapshot()
            return
        }
        let captured = diarizer
        queue.async { captured.reset() }
        clearSnapshot()
    }

    // MARK: - Snapshot

    private func snapshotTimeline() {
        guard let d = diarizer else { return }
        let tl = d.timeline
        self.finalizedPredictions = tl.finalizedPredictions
        self.tentativePredictions = tl.tentativePredictions
        self.finalizedFrameCount = tl.numFinalizedFrames
        self.tentativeFrameCount = tl.numTentativeFrames
        self.frameDurationSeconds = tl.config.frameDurationSeconds
        self.numSpeakerChannels = tl.speakerCapacity
        self.speakers = Dictionary(
            uniqueKeysWithValues: tl.speakers.map { ($0.key, $0.value.name) }
        )

        var allSegs: [DiarizerSegment] = []
        for (_, sp) in tl.speakers {
            allSegs.append(contentsOf: sp.finalizedSegments)
            allSegs.append(contentsOf: sp.tentativeSegments)
        }
        allSegs.sort()
        self.segments = allSegs
    }

    // MARK: - Test hooks

    #if DEBUG
    /// Test-only: inject events into the UI pipeline and observe the
    /// MainActor mutation via `await Task.yield()`. Keeps the stream
    /// private to the ViewModel.
    func _testInjectProgress(_ p: Double) {
        eventContinuation.yield(.progress(p))
    }
    func _testInjectStatus(_ msg: String) {
        eventContinuation.yield(.status(msg))
    }
    #endif
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
