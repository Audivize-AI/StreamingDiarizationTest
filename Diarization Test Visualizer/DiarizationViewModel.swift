//
//  DiarizationViewModel.swift
//  Diarization Test Visualizer
//
//  Created by Benjamin Lee on 12/6/25.
//

import Foundation
import SwiftUI
import Combine

@MainActor
final class DiarizationViewModel: NSObject, ObservableObject {
    @Published var segments: [DiarizationTimelineSegment] = SampleDiarizationData.exampleSegments
    @Published var status: String = "Idle"
    @Published var isListening: Bool = false
    @Published var errorMessage: String?

    private var diarizer: RealTimeDiarizer?
    private var isStarting = false

    func start() {
        guard diarizer == nil, !isStarting else { return }
        isStarting = true
        status = "Starting…"
        errorMessage = nil

        Task {
            do {
                let liveDiarizer = try await RealTimeDiarizer()
                liveDiarizer.delegate = self
                try liveDiarizer.startCapture()
                diarizer = liveDiarizer
                isListening = true
                status = "Listening for speech…"
            } catch {
                errorMessage = error.localizedDescription
                status = "Failed to start"
            }
            isStarting = false
        }
    }

    func stop() {
        diarizer?.stopCapture()
        diarizer = nil
        isListening = false
        status = "Stopped"
    }
}

@MainActor
extension DiarizationViewModel: RealTimeDiarizerDelegate {
    func diarizationDidUpdate(history: DiarizationHistory) {
        segments = history.speakers.values
            .flatMap { track in
                track.segments.map { segment in
                    DiarizationTimelineSegment(
                        speakerId: segment.speakerId,
                        start: segment.start,
                        end: segment.end,
                        quality: segment.embedding.quality
                    )
                }
            }
            .sorted { $0.start < $1.start }

        let speakerCount = history.speakers.count
        let segmentCount = history.segmentCount
        status = "Speakers: \(speakerCount), Segments: \(segmentCount)"
    }
}
