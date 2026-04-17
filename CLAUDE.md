# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

SwiftUI macOS app demonstrating a CPU-optimized Swift port of the **LS-EEND** (Long-form Streaming End-to-End Neural Diarization) CoreML pipeline. The Python reference lives in the parent repo at `../LS-EEND/` — this project consumes the mlpackages emitted by `LS-EEND/coreml/convert_pipeline.py`.

Target platform: macOS 26+, arm64. Swift 6 strict concurrency, sandboxed.

## Commands

```bash
# Build
xcodebuild -project LS-EEND-Test.xcodeproj -scheme LS-EEND-Test -destination 'platform=macOS' build

# Run all unit tests (excludes UI tests on purpose — do not run UI tests)
xcodebuild test -project LS-EEND-Test.xcodeproj -scheme LS-EEND-Test \
  -destination 'platform=macOS' -only-testing:LS-EEND-TestTests

# Run a single test
xcodebuild test -project LS-EEND-Test.xcodeproj -scheme LS-EEND-Test \
  -destination 'platform=macOS' \
  -only-testing:LS-EEND-TestTests/LSEENDParityTests/testEndToEndProbsParityT3

# Parity tests require three env vars forwarded to the test runner via the
# TEST_RUNNER_ prefix (xcodebuild does not forward normal shell env vars).
# Point them at your local checkout:
TEST_RUNNER_LSEEND_PARITY_DIR=/path/to/LS-EEND/coreml/parity \
TEST_RUNNER_LSEEND_LOCAL_MODELS_DIR=/path/to/LS-EEND/coreml/out \
TEST_RUNNER_LSEEND_PARITY_FLAC=/path/to/LDC2022S14.flac \
xcodebuild test … -only-testing:LS-EEND-TestTests/LSEENDParityTests

# Dump Python parity fixtures (run before parity tests)
cd ../LS-EEND && conda activate NeMo && python coreml/dump_parity_fixtures.py
```

Parity fixtures land under `../LS-EEND/coreml/parity/`. `LSEENDParityTests.swift` reads them through the `LSEEND_PARITY_DIR` env var (set in the Xcode scheme for local parity runs) — no absolute paths baked in.

## Architecture

### Diarizer pipeline (the thing this repo validates)

Audio (16/48 kHz) → **`AudioConverter`** → 8 kHz mono → **`AudioMelSpectrogram`** (FP32 STFT + mel + ln) → **log10 scale + CMN** (in `LSEENDDiarizer.extractNewFeatures`) → **subsample ×10 + ±7 context** → 345-dim feature frames → **`LSEENDModel.predict`** (CoreML, T-frame batch, FP32 state I/O) → `probs [1, T, maxSpeakers]` → **`DiarizerTimeline`** → segments.

All hot-path state is preallocated. MLMultiArray state tensors are ref-swapped when CoreML outputs contiguous strides, stride-aware-copied otherwise.

### Key files

- `FluidAudio/Diarizer/DiarizerProtocol.swift` — the `Diarizer` protocol the LS-EEND (and any future diarizer) implements.
- `FluidAudio/Diarizer/DiarizerTimeline.swift` — speaker/segment accumulation with onset/offset thresholding. Stores flat `[Float]` predictions per-slot.
- `FluidAudio/Diarizer/LS-EEND/LSEENDDiarizer.swift` — streaming orchestrator (audio ring, CMN, feature windowing, T-block driving, finalize with silence flush).
- `FluidAudio/Diarizer/LS-EEND/LSEENDModels.swift` — CoreML wrapper. Preallocates `melBuffer`, `validBuffer`, `probsBuffer`; consumes state by ref-swap + fallback stride-aware copy.
- `FluidAudio/Diarizer/LS-EEND/LSEENDTypes.swift` — `LSEENDMetadata` (decoded from `userDefinedMetadata["config"]`) + `LSEENDState` (FP32 MLMultiArrays).
- `FluidAudio/AudioUtils/AudioMelSpectrogram.swift` — shared vDSP-backed mel. LS-EEND uses it with `preemph=0`, `windowPeriodic=true`, `logFloorMode=.clamped`.
- `FluidAudio/MultiArrayUtils/ANEMemoryUtils.swift` — `strideAwareCopy(from:to:)` is load-bearing. CoreML returns tile-padded strides on the innermost axis; flat memcpy silently reads garbage.
- `DiarizerViewModel.swift` — `@MainActor` ObservableObject. File/mic/enroll flows. Mic tap is installed via a `nonisolated static` helper to avoid Swift 6 isolation-check crashes on the audio RT thread. UI updates on background queues route through `MainActorSink` (dispatch-based, never `Task { @MainActor }`).

### Parity contract vs Python

Numerical parity is the ship-readiness criterion. Tests under `LS-EEND-TestTests/LSEENDParityTests.swift` enforce:

| Stage | Threshold | Mechanism |
|---|---|---|
| `logmel23` | max\|Δ\| < 5e-5 | `AudioMelSpectrogram` + log10 scale vs librosa FP32 |
| `feat345` | max\|Δ\| < 1e-3 | CMN + subsample + context concat vs Python diarizer tap |
| `probs` T=3 | max\|Δ\| < 0.25 | end-to-end CoreML on `audio_8k.f32` fixture |
| binarized >0.5 per-slot | within ±5 frames | Python reference `[385, 179, 0..0]` on LDC2022S14 |

The Python fixtures (`dump_parity_fixtures.py`) explicitly force FP32 everywhere (`stft dtype=np.complex64`, `filters.mel dtype=np.float32`, `pad_mode="constant"`) to isolate real bugs from FP32/FP64 drift.

## Known gotchas (encoded in code; do not regress)

1. **`URL.appendingPathComponent(String)`** stats the filesystem and appends a trailing `/` for directories. `MLModel.compileModel(at:)` rejects those URLs with "Input stream is not valid". Always build mlpackage URLs with `URL(fileURLWithPath:isDirectory: false)`.
2. **CoreML output MLMultiArrays use ANE tile-padded strides** (inner-axis physical stride may be 16 even when shape says 10). `probs` is copied through `probsBuffer` via `ANEMemoryUtils.strideAwareCopy` for this reason. A flat `memcpy(dst, probs.dataPointer, count*4)` reads padding bytes and shifts outputs.
3. **`AVAudioEngine` tap closures run on a real-time thread.** Capturing a `@MainActor`-isolated `self` (even weakly) in the tap trips Swift 6's `_swift_task_checkIsolatedSwift` → `dispatch_assert_queue_fail`. Install the tap from a `nonisolated static` helper and post UI updates via `MainActorSink` (plain `DispatchQueue.main.async`, not `Task { @MainActor }`).
4. **`NSMicrophoneUsageDescription`** is set as `INFOPLIST_KEY_NSMicrophoneUsageDescription` in `project.pbxproj`. Without it the app gets SIGKILL'd on first mic access despite the `audio-input` entitlement.
5. **`fileImporter` returns security-scoped URLs.** Access evaporates when the URL crosses threads. Wrap background work in `startAccessingSecurityScopedResource()` / `stopAccessing…`.
6. **`fileImporter`'s `isPresented` setter** is called *before* the completion handler. Clearing state in the setter makes the completion see `nil` and skip dispatch. Leave the setter empty; clear at the top of the completion handler.
7. **`DiarizerTimeline.updateSegments`** previously had a `return` inside `for speakerIndex in 0..<speakerCapacity` — aborted after slot 0 so only one speaker ever committed segments. Must be `continue`. `DiarizerTimelineTests.testSegmentsCommitAcrossAllSlots` guards this.
8. **Python feat/mel pipeline uses zero-padded STFT (`pad_mode="constant"`)**, not librosa's default `reflect`. Training used `reflect` — the diarizer runtime mismatch is documented in `LS-EEND/coreml/CLAUDE.md`. Parity fixtures must match the runtime, not training.
9. **LS-EEND has `max_speakers` (output) and `max_nspks` (decoder attractors) that differ** (e.g. dih3 = 10 / 12). `LSEENDState.init(from:)` uses `maxNspks` for `dec_kv`; the timeline uses `maxSpeakers` for output channels. Don't conflate.

## Configuration

- **Local model dir** (overrides HuggingFace download): set `LSEEND_LOCAL_MODELS_DIR` env var in the Xcode scheme, or call `DiarizerViewModel.setLocalModelsDirectory(_:)` from UI to persist a security-scoped bookmark. No absolute user paths are baked into the binary or entitlements.
- **Sandbox entitlements** live in `LS-EEND-Test/LS-EEND-Test.entitlements`: `app-sandbox`, `audio-input`, `user-selected.read-write`, `network.{client,server}`. No temp-exception-absolute-path — dev-tree access goes through the env var (no sandbox check) or user-selected bookmark.

## Running UI tests

**Don't.** `xcodebuild test` without `-only-testing:LS-EEND-TestTests` pulls in UI tests that require entitlements/permissions the CI shell won't grant. Always scope to `LS-EEND-TestTests`.

## Diagnostic commands (learned this session)

```bash
# Latest xcresult path
ls -td ~/Library/Developer/Xcode/DerivedData/LS-EEND-Test-*/Logs/Test/*.xcresult | head -1

# Test detail JSON (failure messages embedded under testRuns)
xcrun xcresulttool get test-results test-details \
  --test-id "LSEENDParityTests/testEndToEndProbsParityT3()" \
  --path <xcresult> --format json

# App entitlements / Info.plist (built product)
codesign -d --entitlements - <path>/LS-EEND-Test.app
plutil -p <path>/LS-EEND-Test.app/Contents/Info.plist

# Read mlpackage config JSON without loading CoreML
strings <pkg>.mlpackage/Data/com.apple.CoreML/model.mlmodel | grep -oE '\{"model_name[^}]*\}'
```

**XCTest stdout does not surface in xcresult.** Put diagnostic strings into the `XCTAssert…` message arg so they show up as "Failure Message" in the test-details JSON. `print()` output is swallowed.

## Python env

```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate NeMo
```

For parity fixtures: FP32 at every cast (`dtype=np.complex64`, `dtype=np.float32`, `.astype(np.float32)`) — librosa/numpy default to FP64, which drifts vs Swift's vDSP by 1e-3 instead of 3e-5. See `dump_parity_fixtures.py` for the reference pipeline.

## User directives (strict)

- **Never bake absolute user paths (`/Users/<name>/…`) into files that ship to git.** Applies to `CLAUDE.md`, Swift tests, entitlements, `project.pbxproj`. Use `TEST_RUNNER_*` env vars or security-scoped bookmarks. Sweep before commit: `grep -rn "/Users/" LS-EEND-Test | grep -v DerivedData`.
- **Don't auto-simplify the `MainActorSink` pattern** back to `Task { @MainActor }` — the isolation-check crash rationale is in the file comments. A future refactor that "just uses async/await" will crash on the audio RT thread.
- **`appendingPathComponent(_:)` without `isDirectory:`** silently stats. Always pass the flag explicitly.
