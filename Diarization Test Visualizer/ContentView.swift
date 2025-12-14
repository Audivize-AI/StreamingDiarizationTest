//
//  ContentView.swift
//  Diarization Test Visualizer
//
//  Created by Benjamin Lee on 12/5/25.
//

import SwiftUI

struct AutoScrollKey: Hashable {
    let lastId: UUID?
    let width: CGFloat
}

struct ContentView: View {
    @StateObject private var viewModel = DiarizationViewModel()
    private let pixelsPerSecondBase: CGFloat = 70
    private let minScale: CGFloat = 0.4
    private let maxScale: CGFloat = 2.5
    @State private var timelineScale: CGFloat = 1.0
    @State private var lastMagnification: CGFloat = 1.0
    private let labelWidth: CGFloat = 110
    @State private var scrollOffset: CGFloat = 0
    @State private var viewportWidth: CGFloat = 0
    @State private var shouldFollowLatest: Bool = true

    var body: some View {
        let colorMap = SpeakerColorMap(speakers: viewModel.segments.map { $0.speakerId })
        let pixelsPerSecond = pixelsPerSecondBase * timelineScale
        let totalDuration = viewModel.segments.map(\.end).max() ?? 0
        let totalWidth = max(CGFloat(totalDuration) * pixelsPerSecond, 480)
        let autoScrollToken = AutoScrollKey(lastId: viewModel.segments.last?.id, width: totalWidth)

        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    header
                    controlRow
                    if let error = viewModel.errorMessage {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                    }
                    TimelineLegend(colorMap: colorMap)
                    if viewModel.segments.isEmpty {
                        placeholder
                    } else {
                        DiarizationTimeline(
                            segments: viewModel.segments,
                            colorMap: colorMap,
                            pixelsPerSecond: pixelsPerSecond,
                            labelWidth: labelWidth,
                            totalWidth: totalWidth,
                            autoScrollToken: autoScrollToken,
                            onScrollOffsetChange: { offset in
                                scrollOffset = offset
                                shouldFollowLatest = true
                            },
                            onViewportWidthChange: { width in
                                viewportWidth = width
                            },
                            shouldAutoScroll: shouldFollowLatest
                        )
                        .gesture(magnificationGesture)
                    }
                }
                .padding()
            }
            .navigationTitle("Diarization History")
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Test Diarization Timeline")
                .font(.largeTitle.weight(.semibold))
            Text("Scroll horizontally to inspect every segment. Each speaker keeps a consistent color, and labels show both the speaker ID and timestamps.")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
    }

    private var controlRow: some View {
        HStack(spacing: 12) {
            Label(viewModel.status, systemImage: viewModel.isListening ? "waveform" : "pause")
                .font(.subheadline)
                .foregroundColor(.secondary)

            Spacer()

            Button {
                viewModel.isListening ? viewModel.stop() : viewModel.start()
            } label: {
                Label(viewModel.isListening ? "Stop Capture" : "Start Capture",
                      systemImage: viewModel.isListening ? "stop.fill" : "play.fill")
                    .font(.subheadline.weight(.semibold))
            }
            .buttonStyle(.borderedProminent)
        }
    }

    private var placeholder: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("No segments yet")
                .font(.headline)
            Text("Start capture to stream diarization results. Segments will appear here as they are identified.")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding()
        .background(RoundedRectangle(cornerRadius: 12).fill(Color.gray.opacity(0.1)))
    }

    private var magnificationGesture: some Gesture {
        MagnificationGesture()
            .onChanged { value in
                let delta = value / lastMagnification
                let newScale = timelineScale * delta
                timelineScale = min(max(newScale, minScale), maxScale)
                lastMagnification = value
                scrollOffset *= delta
            }
            .onEnded { _ in
                lastMagnification = 1.0
            }
    }
}

struct DiarizationTimeline: View {
    let segments: [DiarizationTimelineSegment]
    let colorMap: SpeakerColorMap
    let pixelsPerSecond: CGFloat
    let labelWidth: CGFloat
    let totalWidth: CGFloat
    let autoScrollToken: AutoScrollKey
    let onScrollOffsetChange: (CGFloat) -> Void
    let onViewportWidthChange: (CGFloat) -> Void
    let shouldAutoScroll: Bool

    private let endAnchorId = "timeline-end-anchor"

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView(.horizontal, showsIndicators: true) {
                timelineContent
            }
            .coordinateSpace(name: "timelineScroll")
            .onPreferenceChange(TimelineScrollOffsetKey.self) { onScrollOffsetChange($0) }
            .onPreferenceChange(TimelineViewportWidthKey.self) { onViewportWidthChange($0) }
            .onChange(of: autoScrollToken) {
                triggerAutoScroll(proxy)
            }
            .onChange(of: segments.count) {
                triggerAutoScroll(proxy)
            }
            .onAppear {
                triggerAutoScroll(proxy)
            }
            .background(
                GeometryReader { geo in
                    Color.clear
                        .preference(key: TimelineViewportWidthKey.self, value: geo.size.width)
                }
            )
        }
    }

    private func triggerAutoScroll(_ proxy: ScrollViewProxy) {
        if shouldAutoScroll {
            DispatchQueue.main.async {
                if shouldAutoScroll {
                    withAnimation(.easeOut) {
                        proxy.scrollTo(endAnchorId, anchor: .trailing)
                    }
                }
            }
        }
    }

    private var timelineContent: some View {
        VStack(alignment: .leading, spacing: 8) {
            timeAxis

            ForEach(colorMap.orderedSpeakers, id: \.self) { speaker in
                HStack(alignment: .center, spacing: 10) {
                    Text("Speaker \(speaker)")
                        .font(.caption2.weight(.semibold))
                        .foregroundColor(.secondary)
                        .frame(width: labelWidth, alignment: .leading)

                            TimelineRow(
                                segments: segments.filter { $0.speakerId == speaker },
                                totalWidth: totalWidth,
                                pixelsPerSecond: pixelsPerSecond,
                                color: colorMap.color(for: speaker)
                            )
                        }
                        .frame(height: 30)
                    }
                }
                .padding(10)
                .background(RoundedRectangle(cornerRadius: 12).fill(Color.gray.opacity(0.12)))
                .frame(width: totalWidth + labelWidth + 24, alignment: .leading)
                .background(
                    GeometryReader { geo in
                        Color.clear
                            .preference(key: TimelineScrollOffsetKey.self,
                                        value: -geo.frame(in: .named("timelineScroll")).minX)
                    }
                )
                .overlay(alignment: .trailing) {
                    Color.clear
                        .frame(width: 1, height: 1)
                        .id(endAnchorId)
                }
            }

    private var timeAxis: some View {
        HStack(alignment: .top, spacing: 10) {
            Text("Time (s)")
                .offset(x: 0, y: 16)
                .font(.caption2.weight(.semibold))
                .foregroundColor(.secondary)
                .frame(width: labelWidth, alignment: .leading)

            TimeAxis(
                totalDuration: totalWidth / pixelsPerSecond,
                totalWidth: totalWidth,
                pixelsPerSecond: pixelsPerSecond
            )
        }
        .frame(height: 32)
    }
}

struct TimeAxis: View {
    let totalDuration: Double
    let totalWidth: CGFloat
    let pixelsPerSecond: CGFloat

    private var tickStep: Double {
        let desiredPixelSpacing: Double = 120
        let approxSeconds = desiredPixelSpacing / Double(pixelsPerSecond)
        let candidates: [Double] = [0.5, 1, 2, 5, 10, 15, 30, 60, 120]
        return candidates.first(where: { $0 >= approxSeconds }) ?? max(1, approxSeconds)
    }

    private var ticks: [Double] {
        guard totalDuration > 0 else { return [0] }
        return stride(from: 0.0, through: totalDuration, by: tickStep).map { $0 }
    }

    var body: some View {
        ZStack(alignment: .topLeading) {
            Rectangle()
                .fill(Color.gray.opacity(0.25))
                .frame(width: totalWidth, height: 1)
                .offset(y: 19)

            ForEach(ticks, id: \.self) { tick in
                let x = CGFloat(tick) * pixelsPerSecond

                Rectangle()
                    .fill(Color.gray.opacity(0.35))
                    .frame(width: 1, height: 30)
                    .offset(x: x)

                Text("\(tick, specifier: "%.1f")")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .offset(x: x + 4, y: 32)
            }
        }
        .frame(width: totalWidth, height: 50, alignment: .topLeading)
    }
}

struct TimelineRow: View {
    let segments: [DiarizationTimelineSegment]
    let totalWidth: CGFloat
    let pixelsPerSecond: CGFloat
    let color: Color

    var body: some View {
        ZStack(alignment: .leading) {
            RoundedRectangle(cornerRadius: 10)
                .fill(Color.gray.opacity(0.1))
                .frame(width: totalWidth, height: 28)

            ForEach(segments) { segment in
                let barWidth = max(12, CGFloat(segment.duration) * pixelsPerSecond)
                let x = CGFloat(segment.start) * pixelsPerSecond

                RoundedRectangle(cornerRadius: 8)
                    .fill(color.opacity(0.9))
                    .frame(width: barWidth, height: 22)
                    .overlay(
                        VStack(alignment: .leading, spacing: 2) {
                            Text("\(segment.start, specifier: "%.1f")s – \(segment.end, specifier: "%.1f")s (\(100 * segment.quality, specifier: "%.2f")%)")
                                .font(.caption2.weight(.semibold))
                        }
                        .foregroundColor(.white)
                        .padding(.horizontal, 6),
                        alignment: .leading
                    )
                    .offset(x: x)
            }
        }
        .frame(width: totalWidth, height: 28, alignment: .leading)
    }
}

struct TimelineLegend: View {
    let colorMap: SpeakerColorMap

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Speakers")
                .font(.headline)
            HStack(spacing: 12) {
                ForEach(colorMap.orderedSpeakers, id: \.self) { speaker in
                    HStack(spacing: 6) {
                        Circle()
                            .fill(colorMap.color(for: speaker))
                            .frame(width: 14, height: 14)
                        Text("ID \(speaker)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .background(
                        Capsule()
                            .fill(Color.gray.opacity(0.15))
                    )
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
