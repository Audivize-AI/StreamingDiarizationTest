# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
# Build
xcodebuild -scheme LS-EENDTest -configuration Debug build

# Run all tests (requires model artifacts under LS-EEND workspace root)
xcodebuild test -scheme LS-EENDTest -configuration Debug

# Run a single test
xcodebuild test -scheme LS-EENDTest -configuration Debug \
  -only-testing:LS-EENDTestTests/LSEENDRuntimeTests/testOfflineDIHARD3ParityAndDER
```

The test suite compiles a standalone CLI probe (`Tools/LSEENDRuntimeProbe.swift`) from Swift source on first run. It locates `swiftc` via Xcode toolchain and writes the binary to `artifacts/bin/lseend_runtime_probe`. Tests compare Swift inference output against golden NumPy arrays and RTTM reference files stored under `artifacts/`.

The workspace root is resolved by walking up from `LSEENDSupport.swift` to find the `LS-EEND` directory, or via the `LSEEND_WORKSPACE_ROOT` environment variable.

## Architecture

This is a macOS SwiftUI application for real-time speaker diarization using CoreML. The app target is `LS-EENDTest` (module name `LS_EENDTest`).

### Inference pipeline (LS-EEND)

The LS-EEND (Linear Streaming End-to-End Neural Diarization) pipeline is stateful and frame-by-frame:

1. **Feature extraction** (`LSEENDFeatureExtraction.swift`): Mel spectrogram → log-mel cumulative mean normalization → splice-and-subsample context windowing. Two extractors exist: `LSEENDOfflineFeatureExtractor` (batch) and `LSEENDStreamingFeatureExtractor` (incremental with audio buffer management).

2. **Stateful model inference** (`LSEENDInference.swift`): `LSEENDInferenceEngine` wraps a CoreML model with 6-tensor RNN state (enc/dec KV caches, scales, conv cache, top buffer). Each `predictStep` feeds one feature frame and returns logits + updated state. The engine caches compiled models and shared resources via `LSEENDInferenceSharedResourcesStore` (keyed by model path + compute units).

3. **Streaming session** (`LSEENDStreamingSession`): Manages incremental audio → feature → inference loop. `pushAudio()` returns committed frames + a preview tail (zero-padded decode of pending frames). `finalize()` flushes remaining frames. `snapshot()` assembles the full result.

4. **Evaluation** (`LSEENDEvaluation.swift`): RTTM parsing/writing, threshold + median filter, DER computation with collar masking and Hungarian-style speaker assignment.

### Model configuration

- Model variants are defined in `LSEENDModelVariant` (AMI, CALLHOME, DIHARD II/III)
- Each variant maps to a `LSEENDModelDescriptor` with URLs for `.mlpackage`, metadata JSON, checkpoint, and config
- `LSEENDModelMetadata` (decoded from JSON) describes all model dimensions, state shapes, and audio parameters
- `LSEENDFeatureConfig` derives computed properties (FFT size, mel count, context, subsampling) from metadata

### Sortformer (stub)

The Sortformer diarizer runtime has been removed from this app target. Only `SortformerConfig.swift` remains as a compatibility shim for `ModelNames` registry lookups. The full Sortformer runtime lives in a sibling project.

### Shared utilities (`FluidAudio/Shared/`)

- `NeMoMelSpectrogram`: NeMo-compatible mel spectrogram (vDSP FFT, mel filterbanks)
- `AudioStream`: Real-time chunking/windowing buffer for microphone input
- `AudioConverter`: Sample rate conversion via AVAudioConverter
- `AppLogger`: OSLog wrapper that mirrors to stderr in DEBUG builds (subsystem `com.fluidinference`)
- `ANEMemoryOptimizer`/`ANEMemoryUtils`: Apple Neural Engine memory alignment utilities

### UI layer

- `ContentView.swift`: HSplitView with controls (left) and dual heatmap visualization (right)
- `LSEENDDemoViewModel`: State machine managing model loading, audio capture (mic or file simulation), inference scheduling, and display state. All inference runs on a serial `DispatchQueue`; UI updates dispatch to main.

## Key patterns

- **Shared resource caching**: `LSEENDInferenceSharedResourcesStore` and `LSEENDMelSpectrogramStore` are singletons with NSLock-guarded caches, keyed by model path or feature config. Multiple engine instances sharing the same model path reuse the same compiled MLModel.
- **State deep-copying**: `LSEENDModelState.copy()` and `cloneMultiArray` create independent copies of MLMultiArray tensors for preview/branch inference without mutating the committed state.
- **Full vs real outputs**: The model outputs `fullOutputDim` columns (including 2 boundary tracks). `cropRealTracks` strips the first and last columns to yield `realOutputDim` speaker probabilities.
- **Threading**: NSLock for mutable shared state, DispatchQueue for async processing. Not Actor-based.
- **Probe-based testing**: Tests compile `LSEENDRuntimeProbe.swift` + source files into a CLI binary and run it as a subprocess to isolate inference from the test process.
