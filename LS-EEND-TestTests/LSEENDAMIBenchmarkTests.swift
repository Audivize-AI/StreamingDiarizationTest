//
//  LSEENDAMIBenchmarkTests.swift
//  LS-EEND-TestTests
//
//  AMI SDM (single-distant-microphone) benchmark for the Swift LS-EEND
//  pipeline. Sweeps T ∈ 1…5 against the `ami` mlpackage variant and
//  reports per-meeting + aggregate DER (collar=0) and RTFx. Not a
//  regression gate — the only hard assertions are on "did we actually
//  run something" (meetings > 0, frames > 0). The target number is the
//  paper's 20.97% AMI SDM DER; if the aggregate lands within ~1pp for
//  some T, the Swift port is behaving correctly end-to-end.
//
//  Env vars (forward via `TEST_RUNNER_*` under xcodebuild):
//    - LSEEND_LOCAL_MODELS_DIR  dir containing `ls_eend_ami_{100..500}ms.mlpackage`
//    - LSEEND_AMI_SDM_DIR       dir containing `*.Mix-Headset.wav`
//    - LSEEND_AMI_WORDS_DIR     dir containing `{meeting}.{A|B|C|D}.words.xml`
//                               (paper-style word-level reference — Kaldi-derived)
//    - LSEEND_AMI_SEGS_DIR      (optional) fallback for legacy segments.xml
//                               reference if words dir unset; kept for debugging
//

import XCTest
import CoreML
@testable import LS_EEND_Test

private func fmt(_ x: Double, _ digits: Int) -> String {
    String(format: "%.\(digits)f", x)
}

final class LSEENDAMIBenchmarkTests: XCTestCase {

    private static func envDir(_ key: String) -> URL? {
        guard let p = ProcessInfo.processInfo.environment[key], !p.isEmpty,
              p.hasPrefix("/") else { return nil }
        return URL(fileURLWithPath: p, isDirectory: true)
    }

    /// Reference-annotation source. `words` (default, paper/Kaldi-
    /// style) uses `{meeting}.{A..D}.words.xml` merged with a 0.5 s
    /// gap. `segs` uses the looser transcriber `segments.xml` turns.
    private enum RefSource { case words, segs }

    private func skipIfMissing() throws -> (
        models: URL, sdm: URL, refDir: URL, refSource: RefSource
    ) {
        guard let m = Self.envDir("LSEEND_LOCAL_MODELS_DIR"),
              let s = Self.envDir("LSEEND_AMI_SDM_DIR")
        else {
            throw XCTSkip(
                "Set LSEEND_LOCAL_MODELS_DIR + LSEEND_AMI_SDM_DIR + "
                + "LSEEND_AMI_WORDS_DIR (paper reference) via TEST_RUNNER_*."
            )
        }
        if let w = Self.envDir("LSEEND_AMI_WORDS_DIR") {
            return (m, s, w, .words)
        }
        if let r = Self.envDir("LSEEND_AMI_SEGS_DIR") {
            return (m, s, r, .segs)
        }
        throw XCTSkip(
            "Set LSEEND_AMI_WORDS_DIR (preferred — Kaldi/paper reference) "
            + "or LSEEND_AMI_SEGS_DIR (legacy transcriber turns)."
        )
    }

    // MARK: - Meeting discovery

    /// Kaldi AMI "full-corpus-ASR" eval split — 16 meetings. Must match
    /// the list the paper reports 20.97% DER on. See
    /// https://github.com/kaldi-asr/kaldi/blob/master/egs/ami/s5/local/split_train_dev_eval.pl
    private static let amiEvalSplit: Set<String> = [
        "EN2002a", "EN2002b", "EN2002c", "EN2002d",
        "ES2004a", "ES2004b", "ES2004c", "ES2004d",
        "IS1009a", "IS1009b", "IS1009c", "IS1009d",
        "TS3003a", "TS3003b", "TS3003c", "TS3003d",
    ]

    /// Meetings under `sdmDir` that have both a `.Mix-Headset.wav` and
    /// all four A/B/C/D reference XMLs in `refDir`, restricted to the
    /// eval split. Ref suffix depends on `refSource`.
    private func listMeetings(
        sdmDir: URL, refDir: URL, refSource: RefSource
    ) throws -> [String] {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(
            at: sdmDir, includingPropertiesForKeys: nil)
        let suffix = ".Mix-Headset.wav"
        let refSuffix = refSource == .words ? "words.xml" : "segments.xml"
        var meetings: [String] = []
        for url in contents where url.lastPathComponent.hasSuffix(suffix) {
            let m = String(url.lastPathComponent.dropLast(suffix.count))
            guard Self.amiEvalSplit.contains(m) else { continue }
            let ok = ["A", "B", "C", "D"].allSatisfy { ch in
                fm.fileExists(atPath: refDir.appendingPathComponent(
                    "\(m).\(ch).\(refSuffix)", isDirectory: false).path)
            }
            if ok { meetings.append(m) }
        }
        return meetings.sorted()
    }

    // MARK: - Local model load

    private func loadLocalDiarizer(
        modelsDir: URL, stepSize: LSEENDStepSize
    ) async throws -> LSEENDDiarizer {
        // Must avoid appendingPathComponent on mlpackage URLs —
        // LS-EEND-Test CLAUDE.md gotcha #1: the directory-trailing-slash
        // breaks MLModel.compileModel on macOS 26.
        let pkgName = "ls_eend_ami_\(stepSize.description).mlpackage"
        let candidates = [
            "\(modelsDir.path)/\(pkgName)",
            "\(modelsDir.path)/optimized/ami/\(stepSize.description)/\(pkgName)",
            "\(modelsDir.path)/ami/\(stepSize.description)/\(pkgName)",
            "\(modelsDir.path)/\(stepSize.description)/\(pkgName)",
        ]
        guard let hit = candidates.first(where: {
            FileManager.default.fileExists(atPath: $0)
        }) else {
            throw XCTSkip(
                "Missing mlpackage \(pkgName) — searched: "
                + candidates.joined(separator: ", "))
        }
        let pkgURL = URL(fileURLWithPath: hit, isDirectory: false)
        let compiled = try await MLModel.compileModel(at: pkgURL)
        let model = try LSEENDModel(modelURL: compiled, computeUnits: .cpuOnly)
        return try LSEENDDiarizer(model: model)
    }

    // MARK: - probs → segments

    /// Threshold → per-speaker binary mask → run-length → segments.
    /// Paper's AMI eval uses threshold=0.5 and **no** median filter
    /// (README line 35: "for AMI, DIHARD2 and DIHARD3 data, no collar
    /// tolerance and no median filtering are used" — we keep the no-
    /// median part; collar is imposed by the paper's 500 ms collar in
    /// metrics.py).
    private func probsToBinary(
        probs: [Float], numFrames: Int, numSpeakers: Int,
        threshold: Float = 0.5
    ) -> [Bool] {
        var out = [Bool](repeating: false, count: numFrames * numSpeakers)
        for f in 0..<numFrames {
            for sp in 0..<numSpeakers {
                out[f * numSpeakers + sp] = probs[f * numSpeakers + sp] > threshold
            }
        }
        return out
    }

    /// Run-length encode the per-speaker binary mask into
    /// `[SpeakerSegment]` with speaker label = slot index.
    private func binaryToSegments(
        _ binary: [Bool], numFrames: Int, numSpeakers: Int,
        frameStep: Double
    ) -> [SpeakerSegment] {
        var out: [SpeakerSegment] = []
        for sp in 0..<numSpeakers {
            var runStart = -1
            for f in 0..<numFrames {
                if binary[f * numSpeakers + sp] {
                    if runStart < 0 { runStart = f }
                } else if runStart >= 0 {
                    out.append(SpeakerSegment(
                        speaker: String(sp),
                        start: Double(runStart) * frameStep,
                        end: Double(f) * frameStep))
                    runStart = -1
                }
            }
            if runStart >= 0 {
                out.append(SpeakerSegment(
                    speaker: String(sp),
                    start: Double(runStart) * frameStep,
                    end: Double(numFrames) * frameStep))
            }
        }
        return out
    }

    // MARK: - The benchmark

    func testAMISDMBenchmark() async throws {
        let (modelsDir, sdmDir, refDir, refSource) = try skipIfMissing()
        let meetings = try listMeetings(
            sdmDir: sdmDir, refDir: refDir, refSource: refSource)
        print("Reference source: \(refSource == .words ? "words (paper)" : "segments (legacy)")")
        XCTAssertGreaterThan(meetings.count, 0,
            "No AMI SDM meetings with all four A/B/C/D segment XMLs found.")
        print("LSEENDAMIBenchmarkTests: \(meetings.count) meetings.")

        let converter = AudioConverter(sampleRate: 8000)
        var csv = "T_ms,meeting,audio_s,run_s,rtfx,der,confusion,fa,miss\n"
        var summary = ""

        // Optional `LSEEND_AMI_STEPS=3,5` restricts the sweep; default is
        // all five. Useful for iterating on a single T without paying the
        // full 5x audio-duration cost.
        let stepsToRun: [LSEENDStepSize] = {
            guard let raw = ProcessInfo.processInfo.environment["LSEEND_AMI_STEPS"],
                  !raw.isEmpty else { return LSEENDStepSize.allCases }
            let vals = raw.split(separator: ",").compactMap { Int($0) }
            return LSEENDStepSize.allCases.filter { vals.contains($0.rawValue) }
        }()
        for step in stepsToRun {
            let diarizer = try await loadLocalDiarizer(
                modelsDir: modelsDir, stepSize: step)
            let maxSpk = diarizer.numSpeakers ?? 0
            let frameS = Double(diarizer.modelFrameHz.map { 1.0 / $0 } ?? 0.1)

            var rows: [(String, Double, Double, DERResult)] = []
            for m in meetings {
                let wavURL = sdmDir.appendingPathComponent(
                    "\(m).Mix-Headset.wav", isDirectory: false)
                let audio = try converter.resampleAudioFile(wavURL)
                let audioDur = Double(audio.count) / 8000.0

                let t0 = ContinuousClock.now
                let timeline = try diarizer.processComplete(
                    audioFileURL: wavURL,
                    keepingEnrolledSpeakers: false,
                    finalizeOnCompletion: true,
                    progressCallback: nil
                )
                let elapsed = (ContinuousClock.now - t0).components
                let runDur = Double(elapsed.seconds)
                    + Double(elapsed.attoseconds) / 1e18
                let nFrames = timeline.numFinalizedFrames
                XCTAssertGreaterThan(nFrames, 0,
                    "[\(step.description) \(m)] 0 finalized frames")

                let binary = probsToBinary(
                    probs: timeline.finalizedPredictions,
                    numFrames: nFrames, numSpeakers: maxSpk)
                let hyp = binaryToSegments(
                    binary, numFrames: nFrames, numSpeakers: maxSpk,
                    frameStep: frameS)
                let ref: [SpeakerSegment] = {
                    switch refSource {
                    case .words:
                        return AMIWordsParser.parse(meeting: m, wordsDir: refDir)
                    case .segs:
                        return AMISegmentsParser.parse(meeting: m, segsDir: refDir)
                    }
                }()
                // Paper's AMI eval: **no collar**, no median filter, no
                // oracle SAD (README: "for AMI, DIHARD2 and DIHARD3
                // data, no collar tolerance and no median filtering are
                // used"). This is the 20.97% online number in Table VII.
                let d = DiarizationDER.compute(
                    ref: ref, hyp: hyp, frameStep: 0.01, collar: 0)
                rows.append((m, audioDur, runDur, d))
                let rtfx = runDur > 0 ? audioDur / runDur : 0
                let stepMs = step.rawValue * 100
                csv += "\(stepMs),\(m),"
                    + fmt(audioDur, 3) + "," + fmt(runDur, 3) + ","
                    + fmt(rtfx, 2) + "," + fmt(d.der, 5) + ","
                    + fmt(d.confusion, 5) + "," + fmt(d.falseAlarm, 5) + ","
                    + fmt(d.miss, 5) + "\n"
                print("[T=\(stepMs)ms] \(m)  DER=\(fmt(d.der * 100, 2))%  "
                    + "conf=\(fmt(d.confusion, 2))  FA=\(fmt(d.falseAlarm, 2))  "
                    + "miss=\(fmt(d.miss, 2))  RTFx=\(fmt(rtfx, 1))x  "
                    + "(audio \(fmt(audioDur, 1))s / run \(fmt(runDur, 1))s)")
            }

            let derMean = rows.map { $0.3.der }.reduce(0, +) / Double(rows.count)
            let totalAudio = rows.map { $0.1 }.reduce(0, +)
            let totalRun = rows.map { $0.2 }.reduce(0, +)
            let rtfx = totalRun > 0 ? totalAudio / totalRun : 0
            let line = "== T=\(step.rawValue * 100)ms  n=\(rows.count)  "
                + "DER=\(fmt(derMean * 100, 2))%  RTFx=\(fmt(rtfx, 1))x  "
                + "(audio=\(fmt(totalAudio, 0))s run=\(fmt(totalRun, 0))s) =="
            print(line)
            summary += line + "\n"
        }

        // Persist CSV + summary for later inspection.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("ami_sdm_bench.csv", isDirectory: false)
        try csv.data(using: .utf8)?.write(to: tmp)
        print("wrote CSV → \(tmp.path)")

        let att = XCTAttachment(string: summary + "\n\n" + csv)
        att.name = "ami_sdm_bench"
        att.lifetime = .keepAlways
        add(att)
    }
}
