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
        print("feat345: swift=\(sFrames) python=\(rFrames)")

        let n = min(sFrames, rFrames) * 345
        var maxAbs: Float = 0
        var argmax: Int = 0
        for i in 0..<n {
            let d = abs(swiftFeat[i] - refFeat[i])
            if d > maxAbs { maxAbs = d; argmax = i }
        }
        let frame = argmax / 345
        let bin = argmax % 345
        print("feat345 max|Δ|=\(maxAbs) at frame=\(frame) bin=\(bin)")
        print("  swift=\(swiftFeat[argmax]) python=\(refFeat[argmax])")
        XCTAssertLessThan(maxAbs, 1e-3)
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
        do {
            let zeros = [Float](repeating: 0, count: 3 * 345)
            let mask = [Float](repeating: 1, count: 3)
            var s = try LSEENDState(from: model.metadata)
            let out = try model.predict(state: &s, features: zeros, frameMask: mask)
            diag += "probe: probs element count = \(out.count) (expect T*maxSpk=\(3*maxSpk))\n"
        } catch {
            diag += "probe failed: \(error)\n"
        }
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
