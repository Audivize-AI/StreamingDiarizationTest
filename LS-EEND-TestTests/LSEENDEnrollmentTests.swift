//
//  LSEENDEnrollmentTests.swift
//  LS-EEND-TestTests
//
//  Exercises `LSEENDDiarizer.enrollSpeaker` on the 6-speaker WAV set.
//  Tests skip unless `TEST_RUNNER_LSEEND_ENROLL_DIR` points at the folder
//  `LS-EEND/6 Speakers/` (containing `Speaker 1M.wav` … `Speaker 6F.wav`).
//

import XCTest
@testable import LS_EEND_Test
import AVFoundation
import CoreML

final class LSEENDEnrollmentTests: XCTestCase {

    private static let targetSampleRate: Double = 8000

    private static let speakerFilenames: [String] = [
        "Speaker 1M.wav",
        "Speaker 2F.wav",
        "Speaker 3F.wav",
        "Speaker 4M.wav",
        "Speaker 5F.wav",
        "Speaker 6F.wav",
    ]

    /// Folder containing the six speaker WAVs. xcodebuild forwards the
    /// `TEST_RUNNER_` prefix; inside the test runner we read the unprefixed
    /// `LSEEND_ENROLL_DIR`.
    private static var enrollDir: URL? {
        guard let p = ProcessInfo.processInfo.environment["LSEEND_ENROLL_DIR"],
              !p.isEmpty
        else { return nil }
        return URL(fileURLWithPath: p, isDirectory: true)
    }

    private func skipIfNoEnrollDir() throws -> URL {
        guard let dir = Self.enrollDir else {
            throw XCTSkip("Set TEST_RUNNER_LSEEND_ENROLL_DIR to the `6 Speakers` folder.")
        }
        for name in Self.speakerFilenames {
            let url = dir.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: url.path) else {
                throw XCTSkip("Missing \(name) in \(dir.path)")
            }
        }
        return dir
    }

    private func loadSpeakers(from dir: URL) throws -> [(name: String, samples: [Float])] {
        let converter = AudioConverter(sampleRate: Self.targetSampleRate)
        return try Self.speakerFilenames.map { filename in
            let url = dir.appendingPathComponent(filename)
            let samples = try converter.resampleAudioFile(url)
            // `Speaker 3F.wav` → `Speaker 3F`.
            let name = (filename as NSString).deletingPathExtension
            return (name: name, samples: samples)
        }
    }

    private static func loadDiarizerFromHF(
        stepSize: LSEENDStepSize = .step300ms
    ) async throws -> LSEENDDiarizer {
        let model = try await LSEENDModel.loadFromHuggingFace(
            variant: .dihard3,
            stepSize: stepSize,
            computeUnits: .cpuOnly
        )
        return try LSEENDDiarizer(model: model)
    }

    /// Sum per-slot frame-weighted activity from a timeline update.
    /// Mirrors `enrollSpeaker`'s own scoring — picking the argmax should
    /// reveal which slot is being driven by the audio just processed.
    private func activityPerSlot(_ update: DiarizerTimelineUpdate) -> [Int: Float] {
        var acc: [Int: Float] = [:]
        for seg in update.finalizedSegments {
            acc[seg.speakerIndex, default: 0] += seg.activity * Float(seg.length)
        }
        return acc
    }

    // MARK: - Tests

    /// At least 4 of 6 speakers should enroll successfully on a fresh
    /// diarizer — enrollment relies on the model actually firing a slot
    /// above threshold within the first clip, which isn't guaranteed for
    /// every voice. 4/6 is a loose floor, not a tight target.
    func testAtLeastFourOfSixEnrollSuccessfully() async throws {
        let dir = try skipIfNoEnrollDir()
        let speakers = try loadSpeakers(from: dir)

        let diarizer = try await Self.loadDiarizerFromHF()

        var enrolled: [(name: String, slot: Int)] = []
        for spk in speakers {
            let result = try diarizer.enrollSpeaker(
                withAudio: spk.samples,
                sourceSampleRate: nil,
                named: spk.name,
                overwritingAssignedSpeakerName: false
            )
            if let result, result.name == spk.name {
                enrolled.append((name: spk.name, slot: result.index))
            }
        }
        let diag = "enrolled \(enrolled.count)/6: \(enrolled.map { "\($0.name)@\($0.slot)" })"
        print(diag)
        XCTAssertGreaterThanOrEqual(enrolled.count, 4, diag)

        // Slots must be distinct.
        let slots = Set(enrolled.map(\.slot))
        XCTAssertEqual(slots.count, enrolled.count,
                       "enrolled slots collided: \(enrolled) — \(diag)")

        // Names on the live timeline must match what was enrolled.
        for (name, slot) in enrolled {
            XCTAssertEqual(
                diarizer.timeline.speakers[slot]?.name, name,
                "slot \(slot) lost name after enroll — \(diag)"
            )
        }
    }

    /// For each speaker that successfully enrolls, feeding their own audio
    /// through `addAudio` + `process` must drive their enrolled slot harder
    /// than any other slot. Each speaker gets a **fresh diarizer** so
    /// cross-speaker session drift can't pollute the test — reuses the
    /// same `LSEENDModel` instance (HF download is cached, and the model
    /// is stateless; only the session carries state).
    func testEnrolledSpeakersReidentifyViaAddAudioAndProcess() async throws {
        let dir = try skipIfNoEnrollDir()
        let speakers = try loadSpeakers(from: dir)

        let model = try await LSEENDModel.loadFromHuggingFace(
            variant: .dihard3, stepSize: .step300ms, computeUnits: .cpuOnly
        )

        var enrolledCount = 0
        var correct = 0
        var misidentified: [(name: String, expected: Int, got: Int, score: Float)] = []

        for spk in speakers {
            let diarizer = try LSEENDDiarizer(model: model)
            guard let enrolled = try diarizer.enrollSpeaker(
                withAudio: spk.samples,
                sourceSampleRate: nil,
                named: spk.name,
                overwritingAssignedSpeakerName: false
            ), enrolled.name == spk.name else {
                continue
            }
            enrolledCount += 1

            try diarizer.addAudio(spk.samples, sourceSampleRate: nil)
            // Use `finalize()` so the session flushes its trailing silence
            // before the last chunks are drained — with `process()` alone,
            // short clips can leave real audio stuck in the queue behind
            // the silence padded in by `enrollSpeaker`'s finalize.
            let update = try diarizer.finalize() ?? (try diarizer.process())
            guard let update else {
                XCTFail("finalize()/process() returned nil for \(spk.name) — no frames drained")
                continue
            }
            // Aggregate activity across both finalized segments at the
            // chunk boundary AND whatever the whole-timeline finalize
            // committed — `finalize` returns the final incremental update,
            // which may be thin; fall back to the full timeline if empty.
            let segments = update.finalizedSegments.isEmpty
                ? diarizer.timeline.speakers.values.flatMap(\.finalizedSegments)
                : update.finalizedSegments
            var activity: [Int: Float] = [:]
            for seg in segments {
                activity[seg.speakerIndex, default: 0] += seg.activity * Float(seg.length)
            }
            guard let (topSlot, topScore) = activity.max(by: { $0.value < $1.value }) else {
                XCTFail("no segments for \(spk.name)")
                continue
            }
            if topSlot == enrolled.index {
                correct += 1
            } else {
                misidentified.append(
                    (name: spk.name, expected: enrolled.index, got: topSlot, score: topScore)
                )
            }
        }

        try XCTSkipIf(enrolledCount == 0, "no speakers enrolled")
        let diag = "reidentified \(correct)/\(enrolledCount). Misses: \(misidentified)"
        print(diag)
        XCTAssertEqual(
            correct, enrolledCount,
            "enrolled speakers failed to reidentify on a fresh session — \(diag)"
        )
    }

    /// After successful enrollment, `processComplete(..., keepingEnrolledSpeakers: nil)`
    /// on a fresh audio clip should keep the enrolled speaker names around
    /// — `keep` defaults to `!timeline.hasSegments`, which is true after
    /// the post-enrollment `timeline.reset(keepingSpeakers: true)`.
    func testProcessCompleteKeepsEnrolledSpeakersAfterEnrollmentOnly() async throws {
        let dir = try skipIfNoEnrollDir()
        let speakers = try loadSpeakers(from: dir)

        let diarizer = try await Self.loadDiarizerFromHF()

        // Enroll the first two speakers.
        var enrolled: [(name: String, slot: Int)] = []
        for spk in speakers.prefix(2) {
            if let r = try diarizer.enrollSpeaker(
                withAudio: spk.samples,
                sourceSampleRate: nil,
                named: spk.name,
                overwritingAssignedSpeakerName: false
            ) {
                enrolled.append((name: spk.name, slot: r.index))
            }
        }
        try XCTSkipIf(enrolled.isEmpty, "no speakers enrolled; can't exercise keep-path")

        // Run processComplete on a *different* speaker's audio with
        // default keepingEnrolledSpeakers — enrolled names must persist.
        let freshAudio = speakers.last!.samples
        _ = try diarizer.processComplete(
            freshAudio,
            sourceSampleRate: nil,
            keepingEnrolledSpeakers: nil,
            finalizeOnCompletion: true,
            progressCallback: nil
        )

        for (name, slot) in enrolled {
            XCTAssertEqual(
                diarizer.timeline.speakers[slot]?.name, name,
                "enrolled name '\(name)' at slot \(slot) was dropped by processComplete"
            )
        }
    }

    /// Harder variant: enroll 1M, 2F, 3F, 4M **on a single diarizer**,
    /// then concatenate their audio (with short silence gaps) and run the
    /// whole stream through `addAudio + finalize` once. Each speaker's
    /// time window in the final timeline must be dominated by their
    /// enrolled slot. Allows up to 10% confusion across all frames scored
    /// inside the per-speaker windows.
    ///
    /// Skipped if fewer than 4 of (1M, 2F, 3F, 4M) enroll on the shared
    /// session — the test needs all four slots filled to be meaningful.
    func testFourSpeakersEnrollAndReidentifyOnSingleSession() async throws {
        let dir = try skipIfNoEnrollDir()
        let speakers = try loadSpeakers(from: dir)
        let targets = Array(speakers.prefix(4))  // 1M, 2F, 3F, 4M
        XCTAssertEqual(targets.count, 4)

        let diarizer = try await Self.loadDiarizerFromHF()

        var enrolled: [(name: String, slot: Int, samples: [Float])] = []
        for spk in targets {
            guard let result = try diarizer.enrollSpeaker(
                withAudio: spk.samples,
                sourceSampleRate: nil,
                named: spk.name,
                overwritingAssignedSpeakerName: false
            ), result.name == spk.name else {
                continue
            }
            enrolled.append((name: spk.name, slot: result.index, samples: spk.samples))
        }
        try XCTSkipIf(
            enrolled.count < 4,
            "only \(enrolled.count)/4 of (1M,2F,3F,4M) enrolled on shared session"
        )
        let slots = Set(enrolled.map(\.slot))
        XCTAssertEqual(slots.count, 4, "enrolled slots collided: \(enrolled)")

        // Build a concatenated stream: 1M ‖ 0.5s gap ‖ 2F ‖ gap ‖ 3F ‖
        // gap ‖ 4M. Enrollment primes the KV cache with a silence-flushed
        // finalize, so the CNN warmup is effectively absorbed by the time
        // we start playback — no leading silence needed.
        let sampleRate = Int(Self.targetSampleRate)
        let silenceSamples = sampleRate / 2  // 0.5s inter-speaker gap
        let frameSamples = Int(Self.targetSampleRate / (diarizer.modelFrameHz ?? 100))
        var concat: [Float] = []
        var windows: [(name: String, slot: Int, startFrame: Int, endFrame: Int)] = []
        for (name, slot, samples) in enrolled {
            let startSample = concat.count
            concat.append(contentsOf: samples)
            let endSample = concat.count
            concat.append(contentsOf: repeatElement(0, count: silenceSamples))
            windows.append((
                name: name, slot: slot,
                startFrame: startSample / frameSamples,
                endFrame: endSample / frameSamples
            ))
        }

        try diarizer.addAudio(concat, sourceSampleRate: nil)
        _ = try diarizer.finalize()

        // Ground truth: per-window slot assignment, shifted forward by
        // the CNN right-context lookahead (`convDelay = 9` for dih3).
        // The diarizer's predictions arrive offset by that many frames
        // relative to the input-audio timeline, so comparing directly
        // would systematically penalize every window's leading edge.
        let totalFrames = (concat.count + frameSamples - 1) / frameSamples
        let rightContextShift = 9
        var groundTruth: [Int?] = Array(repeating: nil, count: totalFrames)
        for win in windows {
            let lo = max(0, win.startFrame + rightContextShift)
            let hi = min(totalFrames, win.endFrame + rightContextShift)
            for f in lo..<hi { groundTruth[f] = win.slot }
        }

        var predictedSlot: [Int?] = Array(repeating: nil, count: totalFrames)
        var predictedScore: [Float] = Array(repeating: 0, count: totalFrames)
        for seg in diarizer.timeline.speakers.values.flatMap(\.finalizedSegments) {
            let lo = max(0, seg.startFrame), hi = min(totalFrames, seg.endFrame)
            for f in lo..<hi where seg.activity > predictedScore[f] {
                predictedScore[f] = seg.activity
                predictedSlot[f] = seg.speakerIndex
            }
        }

        // Standard DER decomposition: confusion rate is measured over
        // frames where BOTH reference speech exists (inside a window) AND
        // the diarizer assigned a speaker. Frames the diarizer missed are
        // "missed speech", not confusion — they're not what this test is
        // asserting on.
        var scoredFrames = 0
        var wrongFrames = 0
        var missedFrames = 0
        var perWindow: [(name: String, wrong: Int, missed: Int, total: Int)] = []
        for win in windows {
            let lo = max(0, win.startFrame + rightContextShift)
            let hi = min(totalFrames, win.endFrame + rightContextShift)
            var winWrong = 0, winMissed = 0
            for f in lo..<hi {
                guard let pred = predictedSlot[f] else { winMissed += 1; continue }
                scoredFrames += 1
                if pred != win.slot { wrongFrames += 1; winWrong += 1 }
            }
            missedFrames += winMissed
            perWindow.append((name: win.name, wrong: winWrong, missed: winMissed, total: hi - lo))
        }

        let confusionRate = scoredFrames > 0
            ? Double(wrongFrames) / Double(scoredFrames)
            : 0
        let diag = "speaker confusion \(wrongFrames)/\(scoredFrames) "
            + "(\(String(format: "%.1f", confusionRate * 100))%); "
            + "missed speech frames: \(missedFrames). "
            + "Per-window: \(perWindow)"
        print(diag)
        XCTAssertLessThanOrEqual(
            confusionRate, 0.10,
            "speaker confusion exceeds 10% — \(diag)"
        )
        // Per-window missed-speech budget — 50%. Short windows and natural
        // pauses inside a speaker's clip can leave sparse diarizer output,
        // and this test is about identity, not coverage. The aggregate
        // confusion assertion above is the real signal.
        for w in perWindow where w.total > 0 {
            let missRate = Double(w.missed) / Double(w.total)
            XCTAssertLessThanOrEqual(
                missRate, 0.50,
                "\(w.name) missed speech \(w.missed)/\(w.total) "
                + "(\(String(format: "%.1f", missRate * 100))%) exceeds 50% — \(diag)"
            )
        }
    }

    /// `overwritingAssignedSpeakerName: false` must refuse to rename an
    /// existing named slot. `true` must allow it. Exercises the
    /// `requireNewSpeaker = isNamed && !overwriteAssignedSpeakerName` gate.
    func testOverwriteGateHonored() async throws {
        let dir = try skipIfNoEnrollDir()
        let speakers = try loadSpeakers(from: dir)

        let diarizer = try await Self.loadDiarizerFromHF()

        // Seed: enroll speaker 1 as "Original".
        guard let seed = try diarizer.enrollSpeaker(
            withAudio: speakers[0].samples,
            sourceSampleRate: nil,
            named: "Original",
            overwritingAssignedSpeakerName: false
        ) else {
            throw XCTSkip("seed enrollment failed; cannot exercise overwrite gate")
        }
        XCTAssertEqual(seed.name, "Original")
        let seedSlot = seed.index

        // Attempt overwrite with a name but overwrite=false. If the
        // diarizer matches the same slot, the guard should reject it.
        // If the diarizer happens to open a fresh slot instead, that's
        // also legal — what matters is that the original slot's name is
        // NOT silently changed.
        _ = try diarizer.enrollSpeaker(
            withAudio: speakers[0].samples,
            sourceSampleRate: nil,
            named: "Attempted",
            overwritingAssignedSpeakerName: false
        )
        XCTAssertEqual(
            diarizer.timeline.speakers[seedSlot]?.name, "Original",
            "overwrite=false leaked a rename onto slot \(seedSlot)"
        )

        // Now with overwrite=true — the same audio should be allowed to
        // rename slot (whatever the matched slot is).
        let renamed = try diarizer.enrollSpeaker(
            withAudio: speakers[0].samples,
            sourceSampleRate: nil,
            named: "Renamed",
            overwritingAssignedSpeakerName: true
        )
        try XCTSkipIf(renamed == nil, "overwrite=true match missed; model-dependent")
        XCTAssertEqual(renamed?.name, "Renamed")
    }
}
