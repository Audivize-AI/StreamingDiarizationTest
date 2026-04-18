//
//  LSEENDParityTests.swift
//  LS-EEND-TestTests
//
//  Per-stage parity against Python fixtures dumped by
//  `LS-EEND/coreml/dump_parity_fixtures.py`.
//

import XCTest
@testable import LS_EEND_Test
import Accelerate
import CoreML

final class LSEENDParityTests: XCTestCase {

    /// Python fixture directory. Resolved at test time from the
    /// `LSEEND_PARITY_DIR` environment variable — set in the Xcode test
    /// scheme for local parity runs. Tests skip when unset.
    private static var fixturesDir: URL? {
        guard let p = ProcessInfo.processInfo.environment["LSEEND_PARITY_DIR"],
              !p.isEmpty
        else { return nil }
        return URL(fileURLWithPath: p, isDirectory: true)
    }

    /// Local mlpackage path — same pattern. Set `LSEEND_LOCAL_MODELS_DIR`.
    private static func modelPackageURL(_ filename: String) -> URL? {
        guard let p = ProcessInfo.processInfo.environment["LSEEND_LOCAL_MODELS_DIR"],
              !p.isEmpty
        else { return nil }
        return URL(fileURLWithPath: p + "/" + filename, isDirectory: false)
    }

    private func skipIfNoFixtures() throws {
        guard let dir = Self.fixturesDir else {
            throw XCTSkip("Set LSEEND_PARITY_DIR in the test scheme.")
        }
        let manifest = dir.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifest.path) else {
            throw XCTSkip("Run `python coreml/dump_parity_fixtures.py` first.")
        }
    }

    private func readFloats(_ name: String) throws -> [Float] {
        guard let dir = Self.fixturesDir else {
            throw XCTSkip("LSEEND_PARITY_DIR unset.")
        }
        let url = dir.appendingPathComponent(name)
        let data = try Data(contentsOf: url)
        let count = data.count / MemoryLayout<Float>.stride
        return data.withUnsafeBytes { raw in
            let base = raw.bindMemory(to: Float.self).baseAddress!
            return Array(UnsafeBufferPointer(start: base, count: count))
        }
    }

    // MARK: - Stage 3: mel-spectrogram parity

    func testLogMel23Parity() throws {
        try skipIfNoFixtures()
        let audio = try readFloats("audio_8k.f32")
        let refMel = try readFloats("logmel23.f32")  // [T * 23]

        let mel = AudioMelSpectrogram(
            sampleRate: 8000,
            nMels: 23,
            nFFT: 256,
            hopLength: 80,
            winLength: 200,
            preemph: 0,
            padTo: 0,
            logFloor: 1e-10,
            logFloorMode: .clamped,
            windowPeriodic: true
        )
        // librosa frame count: 1 + audio.count/hop (uses n_fft in formula,
        // not win_length).
        let expectedFrames = 1 + audio.count / 80
        var (swiftMel, _, _) = mel.computeFlatTransposed(
            audio: audio,
            lastAudioSample: 0,
            paddingMode: .center,
            expectedFrameCount: expectedFrames
        )
        // ln → log10
        var scale: Float = 1.0 / Float(log(10.0))
        swiftMel.withUnsafeMutableBufferPointer { buf in
            vDSP_vsmul(buf.baseAddress!, 1, &scale, buf.baseAddress!, 1,
                       vDSP_Length(buf.count))
        }

        XCTAssertEqual(swiftMel.count, refMel.count,
                       "mel length mismatch: swift=\(swiftMel.count) python=\(refMel.count)")
        let n = min(swiftMel.count, refMel.count)
        var maxAbs: Float = 0
        var argmax = 0
        for i in 0..<n {
            let d = abs(swiftMel[i] - refMel[i])
            if d > maxAbs { maxAbs = d; argmax = i }
        }
        let frame = argmax / 23
        let bin = argmax % 23
        let msg = "logmel23 max|Δ|=\(maxAbs) at frame=\(frame) bin=\(bin) " +
                  "swift=\(swiftMel[argmax]) python=\(refMel[argmax]) " +
                  "early(frame 0 bin 0): swift=\(swiftMel[0]) python=\(refMel[0])"
        print(msg)
        XCTAssertLessThan(maxAbs, 5e-5, "log10-mel parity exceeded 5e-5 — \(msg)")
    }

    // MARK: - Stage 4: subsampled+context feature parity

    func testFeat345Parity() throws {
        try skipIfNoFixtures()
        let audio = try readFloats("audio_8k.f32")
        let refFeat = try readFloats("feat345.f32")

        guard let modelURL = Self.modelPackageURL("ls_eend_dih3_300ms.mlpackage"),
              FileManager.default.fileExists(atPath: modelURL.path)
        else {
            throw XCTSkip("Set LSEEND_LOCAL_MODELS_DIR to a dir containing ls_eend_dih3_300ms.mlpackage.")
        }
        let compiled = try MLModel.compileModel(at: modelURL)
        let model = try LSEENDModel(modelURL: compiled, computeUnits: .cpuOnly)
        let diarizer = try LSEENDDiarizer(model: model)

        let swiftFeat = try diarizer.debugExtractFeatures(audio, sourceSampleRate: nil)
        let sFrames = swiftFeat.count / 345
        let rFrames = refFeat.count / 345

        let n = min(sFrames, rFrames) * 345
        var maxAbs: Float = 0
        var argmax: Int = 0
        for i in 0..<n {
            let d = abs(swiftFeat[i] - refFeat[i])
            if d > maxAbs { maxAbs = d; argmax = i }
        }
        let frame = argmax / 345, bin = argmax % 345
        // Compute per-frame deltas so we can see if the mismatch is
        // per-frame noise or a frame-shift.
        var perFrameMax = [Float](repeating: 0, count: min(sFrames, rFrames))
        for f in 0..<perFrameMax.count {
            var m: Float = 0
            for j in 0..<345 {
                m = max(m, abs(swiftFeat[f*345 + j] - refFeat[f*345 + j]))
            }
            perFrameMax[f] = m
        }
        let firstBigFrame = perFrameMax.firstIndex(where: { $0 > 0.01 }) ?? -1
        // Centre bin of the stacked window: context-frame 7, mel-bin 11 →
        // flat index 7 * 23 + 11 = 172. This is the "real" center frame,
        // not left-padding.
        let centerBin = 7 * 23 + 11
        func frameStr(_ buf: [Float], _ f: Int) -> String {
            let row = buf[f*345 ..< f*345 + 345]
            return "[center]=\(String(format: "%+.4f", row[row.startIndex + centerBin])) " +
                   "bin100=\(String(format: "%+.4f", row[row.startIndex + 100])) " +
                   "bin200=\(String(format: "%+.4f", row[row.startIndex + 200]))"
        }
        let dFrame = max(firstBigFrame, 0)
        let diag = """
            feat345: swift=\(sFrames) frames, python=\(rFrames) frames
            max|Δ|=\(maxAbs) at frame=\(frame) bin=\(bin)
              swift[\(argmax)]=\(swiftFeat[argmax]) python[\(argmax)]=\(refFeat[argmax])
            first frame w/ Δ>0.01: \(firstBigFrame)
            frame \(dFrame) (first-diff):
              swift:  \(frameStr(swiftFeat, dFrame))
              python: \(frameStr(refFeat, dFrame))
            frame 50 (mid):
              swift:  \(frameStr(swiftFeat, min(50, sFrames-1)))
              python: \(frameStr(refFeat, min(50, rFrames-1)))
            perFrameMax[0..<10]=\(perFrameMax.prefix(10).map { String(format: "%.3f", $0) })
            """
        XCTAssertLessThan(maxAbs, 1e-3, diag)
    }

    // MARK: - Stage 5: .flac file path (AVAudioFile + AudioConverter)

    /// Drives the full Swift pipeline from the raw `.flac` file (uses
    /// AVAudioConverter, not the Python-resampled fixture). Can't be
    /// byte-compared to Python (different resampler kernels) but must
    /// produce a sane timeline: non-empty, finite duration, at least
    /// one finalized segment for the 3-speaker DIHARD III sample.
    func testEndToEndFromFlac() throws {
        guard let flacPath = ProcessInfo.processInfo.environment["LSEEND_PARITY_FLAC"],
              FileManager.default.fileExists(atPath: flacPath)
        else {
            throw XCTSkip("Set LSEEND_PARITY_FLAC to a .flac audio file path.")
        }
        let flacURL = URL(fileURLWithPath: flacPath)
        guard let modelURL = Self.modelPackageURL("ls_eend_dih3_300ms.mlpackage"),
              FileManager.default.fileExists(atPath: modelURL.path)
        else {
            throw XCTSkip("Set LSEEND_LOCAL_MODELS_DIR to a dir containing ls_eend_dih3_300ms.mlpackage.")
        }
        let compiled = try MLModel.compileModel(at: modelURL)
        let model = try LSEENDModel(modelURL: compiled, computeUnits: .cpuOnly)
        let diarizer = try LSEENDDiarizer(model: model)

        var lastProgress = (done: 0, total: 0, chunks: 0)
        let timeline = try diarizer.processComplete(
            audioFileURL: flacURL,
            keepingEnrolledSpeakers: false,
            finalizeOnCompletion: true,
            progressCallback: { done, total, chunks in
                lastProgress = (done, total, chunks)
            }
        )

        // File is 49.3s @ 16kHz → resampled to 8kHz → ~394k samples → ~493 frames
        // of 100ms model output.
        XCTAssertGreaterThan(timeline.numFinalizedFrames, 400,
            "expected ~493 frames, got \(timeline.numFinalizedFrames)")
        XCTAssertLessThan(timeline.numFinalizedFrames, 600)
        let duration = timeline.finalizedDuration
        XCTAssertGreaterThan(duration, 40)
        XCTAssertLessThan(duration, 60)

        let totalSegs = timeline.speakers.values
            .reduce(0) { $0 + $1.finalizedSegments.count }
        XCTAssertGreaterThan(totalSegs, 0, "no segments detected")

        // Python reference only fires 2 slots >0.5 on this sample; AVAudio
        // resampler ≠ librosa can push marginal slots below threshold. So
        // require only ≥1 active slot here — the Python-resampled parity
        // test independently confirms the diarizer logic matches.
        let activeSlots = timeline.speakers.values.filter { !$0.finalizedSegments.isEmpty }.count
        XCTAssertGreaterThanOrEqual(activeSlots, 1,
            "expected >=1 speaker, got \(activeSlots)")

        let diag = "flac path: frames=\(timeline.numFinalizedFrames) "
            + "duration=\(String(format: "%.2f", duration))s "
            + "segments=\(totalSegs) activeSlots=\(activeSlots) "
            + "progress=\(lastProgress)"
        print(diag)
    }

    /// Regression guard for the `convDelay` warmup-trim in
    /// `LSEENDDiarizer.drainAndUpdate`: running `processComplete` twice on
    /// the same diarizer must produce the same frame count both times. If
    /// the trim were ever applied to an already-warm KV cache, the second
    /// run would drop real frames and shrink. Pins the second-run count to
    /// within ±2 frames of the first.
    func testProcessCompleteIsIdempotentOnFrameCount() throws {
        guard let flacPath = ProcessInfo.processInfo.environment["LSEEND_PARITY_FLAC"],
              FileManager.default.fileExists(atPath: flacPath)
        else {
            throw XCTSkip("Set LSEEND_PARITY_FLAC to a .flac audio file path.")
        }
        guard let modelURL = Self.modelPackageURL("ls_eend_dih3_300ms.mlpackage"),
              FileManager.default.fileExists(atPath: modelURL.path)
        else {
            throw XCTSkip("Set LSEEND_LOCAL_MODELS_DIR to a dir containing ls_eend_dih3_300ms.mlpackage.")
        }
        let flacURL = URL(fileURLWithPath: flacPath)
        let compiled = try MLModel.compileModel(at: modelURL)
        let model = try LSEENDModel(modelURL: compiled, computeUnits: .cpuOnly)
        let diarizer = try LSEENDDiarizer(model: model)

        let t1 = try diarizer.processComplete(
            audioFileURL: flacURL,
            keepingEnrolledSpeakers: false,
            finalizeOnCompletion: true,
            progressCallback: nil
        )
        let n1 = t1.numFinalizedFrames
        XCTAssertGreaterThan(n1, 400, "first run produced implausibly few frames")

        let t2 = try diarizer.processComplete(
            audioFileURL: flacURL,
            keepingEnrolledSpeakers: false,
            finalizeOnCompletion: true,
            progressCallback: nil
        )
        let n2 = t2.numFinalizedFrames

        // Second run may differ by one or two frames due to trailing-silence
        // rounding, but must not shrink by `convDelay` (=9) — that would
        // indicate the warmup trim is being applied twice.
        XCTAssertLessThanOrEqual(
            abs(n1 - n2), 2,
            "second processComplete diverged by more than 2 frames "
            + "(\(n1) → \(n2)) — warmup trim likely applied to already-warm state"
        )
    }

    // MARK: - Stage 6: end-to-end probs parity (T=3, uses bundled model)

    func testEndToEndProbsParityT3() throws {
        try skipIfNoFixtures()

        guard let modelURL = Self.modelPackageURL("ls_eend_dih3_300ms.mlpackage"),
              FileManager.default.fileExists(atPath: modelURL.path)
        else {
            throw XCTSkip("Set LSEEND_LOCAL_MODELS_DIR to a dir containing ls_eend_dih3_300ms.mlpackage.")
        }
        // Compile mlpackage to mlmodelc lazily.
        let compiled = try MLModel.compileModel(at: modelURL)
        let model = try LSEENDModel(modelURL: compiled, computeUnits: .cpuOnly)
        let diarizer = try LSEENDDiarizer(model: model)

        let audio = try readFloats("audio_8k.f32")
        let refProbs = try readFloats("probs_dih3_T3.f32")

        _ = try diarizer.processComplete(
            audio,
            sourceSampleRate: nil,
            keepingEnrolledSpeakers: false,
            finalizeOnCompletion: true,
            progressCallback: nil
        )

        let swiftProbs = diarizer.timeline.finalizedPredictions
        let maxSpk = diarizer.numSpeakers ?? 10
        let swiftFrames = swiftProbs.count / maxSpk
        let refFrames = refProbs.count / maxSpk
        print("probs T=3 swift=\(swiftFrames) python=\(refFrames) spk=\(maxSpk)")

        let n = min(swiftFrames, refFrames) * maxSpk
        var maxAbs: Float = 0
        var argmax = 0
        for i in 0..<n {
            let d = abs(swiftProbs[i] - refProbs[i])
            if d > maxAbs { maxAbs = d; argmax = i }
        }
        let frame = argmax / maxSpk, spk = argmax % maxSpk
        var diag = "probs T=3 max|Δ|=\(maxAbs) at frame=\(frame) spk=\(spk)\n"
        // Probe actual model output shape with a single prediction call.
        // WIP: `LSEENDState(from:)` signature changed in Benjamin's refactor
        // — probe block disabled until the new ctor is available.
        // do {
        //     let zeros = [Float](repeating: 0, count: 3 * 345)
        //     let mask = [Float](repeating: 1, count: 3)
        //     var s = try LSEENDState(from: model.metadata)
        //     let out = try model.predict(state: &s, features: zeros, frameMask: mask)
        //     diag += "probe: probs element count = \(out.count) (expect T*maxSpk=\(3*maxSpk))\n"
        // } catch {
        //     diag += "probe failed: \(error)\n"
        // }
        diag += "swift=\(swiftProbs[argmax]) python=\(refProbs[argmax])\n"
        diag += "Swift frame 0: \(Array(swiftProbs[0..<maxSpk]))\n"
        diag += "Python frame 0: \(Array(refProbs[0..<maxSpk]))\n"
        diag += "Swift frame 10: \(Array(swiftProbs[10*maxSpk..<11*maxSpk]))\n"
        diag += "Python frame 10: \(Array(refProbs[10*maxSpk..<11*maxSpk]))\n"
        diag += "Swift frame 100: \(Array(swiftProbs[100*maxSpk..<101*maxSpk]))\n"
        diag += "Python frame 100: \(Array(refProbs[100*maxSpk..<101*maxSpk]))\n"
        diag += "swiftFrames=\(swiftFrames) pythonFrames=\(refFrames)\n"

        // Binarized counts per speaker (>0.5) — what actually drives segments.
        var swiftAbove = [Int](repeating: 0, count: maxSpk)
        var pyAbove = [Int](repeating: 0, count: maxSpk)
        for f in 0..<min(swiftFrames, refFrames) {
            for s in 0..<maxSpk {
                if swiftProbs[f*maxSpk + s] > 0.5 { swiftAbove[s] += 1 }
                if refProbs[f*maxSpk + s] > 0.5 { pyAbove[s] += 1 }
            }
        }
        diag += "binarized>0.5 swift=\(swiftAbove) python=\(pyAbove)\n"

        // Tight: per-slot count must match within a small slack. Python is
        // [385, 179, 0, ..., 0] on this sample.
        for s in 0..<maxSpk {
            XCTAssertEqual(swiftAbove[s], pyAbove[s], accuracy: 5,
                "spk \(s) binarized count mismatch — \(diag)")
        }
        XCTAssertLessThan(maxAbs, 0.25, "probs parity exceeded 0.25 — \(diag)")
    }
}
