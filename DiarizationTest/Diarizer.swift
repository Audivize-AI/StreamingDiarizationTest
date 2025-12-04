//
//  Diarizer.swift
//  DiarizationTest
//
//  Created by Benjamin Lee on 12/4/25.
//

import Foundation
import FluidAudio
import AVFoundation


class RealTimeDiarizer {
    private let audioEngine = AVAudioEngine()
    private let diarizer: DiarizerManager
    private let sampleRate: Double = 16000
    private var streamPosition: Double = 0
    // Audio converter for format conversion
    private let converter = AudioConverter()
    private let stream: AudioStream
    private var history: DiarizationHistory = .init()
    
    init() async throws {
        let models = try await DiarizerModels.downloadIfNeeded()
        let config = DiarizerConfig(
            clusteringThreshold: 0.7,
            minSpeechDuration: 0.7,
            minEmbeddingUpdateDuration: 1.5,
            minSilenceGap: 0.5,
            chunkDuration: Float(Config.chunkDuration),
            chunkOverlap: Float(Config.chunkSkip)
        )
        diarizer = DiarizerManager(config: config)  // Default config
        diarizer.initialize(models: models)
        stream = try AudioStream.init(chunkDuration: Config.chunkDuration,
                                      chunkSkip: Config.chunkSkip,
                                      streamStartTime: 0,
                                      chunkingStrategy: .useMostRecent)
        
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
            history.upsert(segment: segment)
        }
        
        print("---------- DIARIZATION HISTORY -----------")
        let orderedHistory = history.getHistory()
        for segment in orderedHistory {
            print(segment.string)
        }
    }

    private func convertBuffer(_ buffer: AVAudioPCMBuffer) -> [Float] {
        // Use FluidAudio.AudioConverter in streaming mode
        // Returns 16kHz mono Float array; swallow conversion errors in sample code
        return (try? converter.resampleBuffer(buffer)) ?? []
    }
}
