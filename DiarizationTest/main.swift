import AVFoundation
import FluidAudio
import Dispatch

struct Segment {
    var start: Float
    var end: Float
    let speaker: String
    var string: String { "Speaker \(speaker): \(start)s - \(end)s" }
    
    init(from segment: TimedSpeakerSegment) {
        self.start = segment.startTimeSeconds
        self.end = segment.endTimeSeconds
        self.speaker = segment.speakerId
    }
    
    func isPartOf(_ other: Segment) -> Bool {
        guard self.speaker == other.speaker else { return false }
        return (self.start < other.end) && (other.start < self.end)
    }
    
    mutating func merge(with other: Segment) {
        self.start = min(self.start, other.start)
        self.end = max(self.end, other.end)
    }
}

class RealTimeDiarizer {
    private let audioEngine = AVAudioEngine()
    private let diarizer: DiarizerManager
    private let sampleRate: Double = 16000
    private var streamPosition: Double = 0
    // Audio converter for format conversion
    private let converter = AudioConverter()
    private let stream: AudioStream
    private var segments: [Segment] = []
    
    init() async throws {
        let models = try await DiarizerModels.downloadIfNeeded()
        diarizer = DiarizerManager()  // Default config
        diarizer.initialize(models: models)
        stream = try AudioStream.init(chunkDuration: 10, chunkSkip: 0.25, streamStartTime: 0, chunkingStrategy: .useMostRecent)
        
        stream.bind { chunk, timestamp in
            Task {
                do {
                    print("Running diarization...")
                    let result = try self.diarizer.performCompleteDiarization(chunk, atTime: timestamp)
                    await self.handleResults(result)
                } catch {
                    print("Diarization error: \(error)")
                }
            }
        }
    }

    func startCapture() throws {
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)

        // Install tap to capture audio
        inputNode.installTap(onBus: 0, bufferSize: 64000, format: recordingFormat) { [weak self] buffer, _ in
            guard let self = self else { return }

            // Convert to 16kHz mono Float array using AudioConverter (streaming)
            
            if let samples = try? self.converter.resampleBuffer(buffer) {
                try? stream.write(from: samples)
            } else {
                
            }
        }

        audioEngine.prepare()
        try audioEngine.start()
    }

    @MainActor
    private func handleResults(_ result: DiarizationResult) {
        for segment in result.segments {
            let newSegment = Segment(from: segment)
            if self.segments.last?.isPartOf(newSegment) == true {
                self.segments[segments.count-1].merge(with: newSegment)
            } else {
                self.segments.append(newSegment)
            }
        }
        
        for segment in segments {
            print(segment.string)
        }
    }

    private func convertBuffer(_ buffer: AVAudioPCMBuffer) -> [Float] {
        // Use FluidAudio.AudioConverter in streaming mode
        // Returns 16kHz mono Float array; swallow conversion errors in sample code
        return (try? converter.resampleBuffer(buffer)) ?? []
    }
}

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
