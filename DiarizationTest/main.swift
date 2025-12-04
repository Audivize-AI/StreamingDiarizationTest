import AVFoundation
import FluidAudio
import Dispatch

// Keep a strong reference to the diarizer so it isn't deallocated
var globalDiarizer: RealTimeDiarizer?

// MARK: - CLI entry point

Task.detached {
    do {
        let diarizer = try await RealTimeDiarizer()
        try diarizer.startCapture()
        globalDiarizer = diarizer
        
    } catch {
        print("Fatal error starting diarizer: \(error)")
    }
}

// Keep the process alive so AVAudioEngine and async tasks can run
dispatchMain()
