//
//  TimelineModels.swift
//  Diarization Test Visualizer
//
//  Created by Benjamin Lee on 12/5/25.
//

import Foundation
import SwiftUI

struct DiarizationTimelineSegment: Identifiable {
    let id = UUID()
    let speakerId: String
    let start: Double
    let end: Double
    let quality: Float

    var duration: Double { end - start }
}

struct SpeakerColorMap {
    private let palette: [Color] = [
        Color(red: 0.87, green: 0.36, blue: 0.26),
        Color(red: 0.19, green: 0.50, blue: 0.82),
        Color(red: 0.11, green: 0.70, blue: 0.55),
        Color(red: 0.58, green: 0.35, blue: 0.86),
        Color(red: 0.95, green: 0.73, blue: 0.28),
        Color(red: 0.21, green: 0.64, blue: 0.73)
    ]

    private(set) var orderedSpeakers: [String]
    private var mapping: [String: Color] = [:]

    init(speakers: [String]) {
        var seen = Set<String>()
        var unique: [String] = []
        for id in speakers where !seen.contains(id) {
            seen.insert(id)
            unique.append(id)
        }
        orderedSpeakers = unique

        for (index, speaker) in orderedSpeakers.enumerated() {
            mapping[speaker] = color(for: index)
        }
    }

    func color(for speaker: String) -> Color {
        mapping[speaker] ?? Color.gray
    }

    private func color(for index: Int) -> Color {
        if index < palette.count { return palette[index] }
        let hue = Double(index % 12) / 12.0
        return Color(hue: hue, saturation: 0.65, brightness: 0.92)
    }
}

enum SampleDiarizationData {
    static let exampleSegments: [DiarizationTimelineSegment] = [
        .init(speakerId: "A", start: 0.0, end: 3.5, quality: 1.0),
        .init(speakerId: "B", start: 3.6, end: 6.2, quality: 1.0),
        .init(speakerId: "A", start: 6.4, end: 11.0, quality: 1.0),
        .init(speakerId: "C", start: 11.1, end: 13.4, quality: 1.0),
        .init(speakerId: "B", start: 13.6, end: 17.0, quality: 1.0),
        .init(speakerId: "C", start: 17.2, end: 22.0, quality: 1.0),
        .init(speakerId: "A", start: 22.1, end: 26.0, quality: 1.0),
        .init(speakerId: "B", start: 26.3, end: 30.0, quality: 1.0),
        .init(speakerId: "A", start: 30.2, end: 33.5, quality: 1.0),
        .init(speakerId: "D", start: 33.8, end: 38.0, quality: 1.0),
        .init(speakerId: "C", start: 38.3, end: 41.0, quality: 1.0),
        .init(speakerId: "D", start: 41.1, end: 46.5, quality: 1.0),
        .init(speakerId: "B", start: 46.8, end: 49.0, quality: 1.0),
        .init(speakerId: "C", start: 49.2, end: 55.0, quality: 1.0)
    ]
}
