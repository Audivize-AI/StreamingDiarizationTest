import SwiftUI
#if os(macOS)
import AppKit
#endif

/// Segment-only timeline driven by variable speaker identities.
struct SpeakerIdentityTimelineView: View {
    let timeline: SortformerTimeline?
    let isRecording: Bool
    let updateTrigger: Int
    let onPlaySegment: ((Float, Float) -> Void)?

    @State private var isFollowingLive = true

    private let rowHeight: CGFloat = 24
    private let labelWidth: CGFloat = 78
    private let visibleFrames = max(
        globalConfig.fifoLen + globalConfig.chunkLen + globalConfig.chunkRightContext,
        globalConfig.spkcacheLen
    )

    private var finalizedSegments: [SpeakerSegment] {
        timeline?.finalizedSpeakerSegments ?? []
    }

    private var tentativeSegments: [SpeakerSegment] {
        timeline?.tentativeSpeakerSegments ?? []
    }

    private var speakerIDs: [Int] {
        Array(Set((finalizedSegments + tentativeSegments).map(\.speakerId))).sorted()
    }

    private var rowCount: Int {
        max(1, speakerIDs.count)
    }

    private var rowHeightTotal: CGFloat {
        CGFloat(rowCount) * rowHeight
    }

    private var horizontalScrollbarHeight: CGFloat {
#if os(macOS)
        switch NSScroller.preferredScrollerStyle {
        case .legacy:
            return NSScroller.scrollerWidth(for: .regular, scrollerStyle: .legacy)
        case .overlay:
            return 0
        @unknown default:
            return 0
        }
#else
        return 0
#endif
    }

    private var totalFrames: Int {
        guard let timeline else { return 0 }
        return timeline.cursorFrame + timeline.numTentative
    }

    var body: some View {
        let scrollViewportHeight = rowHeightTotal + horizontalScrollbarHeight

        VStack(alignment: .leading, spacing: 6) {
            Text(headerText)
                .font(.caption)
                .foregroundStyle(.secondary)

            HStack(alignment: .top, spacing: 4) {
                VStack(alignment: .trailing, spacing: 0) {
                    ForEach(0..<rowCount, id: \.self) { row in
                        if row < speakerIDs.count {
                            Text("ID \(speakerIDs[row])")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        } else {
                            Text("No data")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                }
                .frame(width: labelWidth, height: rowHeightTotal, alignment: .topTrailing)

                GeometryReader { geometry in
                    let viewportWidth = geometry.size.width
                    let cellWidth = viewportWidth / CGFloat(max(visibleFrames, 1))
                    let framesToDraw = max(totalFrames, visibleFrames)
                    let contentWidth = max(CGFloat(framesToDraw) * cellWidth, viewportWidth)

                        ScrollViewReader { proxy in
                            ScrollView(.horizontal, showsIndicators: true) {
                                Canvas { context, size in
                                    drawRows(
                                        context: &context,
                                        size: size,
                                        cellWidth: cellWidth
                                    )
                                }
                            .contentShape(Rectangle())
                            .onTapGesture(count: 1) { location in
                                guard !isRecording, cellWidth > 0 else { return }
                                let row = Int(location.y / rowHeight)
                                guard row >= 0, row < speakerIDs.count else { return }
                                let frame = Int(location.x / cellWidth)
                                let speakerID = speakerIDs[row]
                                guard let segment = identitySegmentAt(frame: frame, speakerID: speakerID) else {
                                    return
                                }
                                onPlaySegment?(segment.startTime, segment.endTime)
                            }
                            .frame(width: contentWidth, height: rowHeightTotal)
                            .background(Color.black.opacity(0.30))
                            .cornerRadius(6)
                            .overlay(alignment: .bottomTrailing) {
                                Color.clear
                                    .frame(width: 1, height: 1)
                                    .id("identity-scroll-end")
                            }
                        }
                        .onChange(of: updateTrigger) { _, _ in
                            if isRecording && isFollowingLive {
                                proxy.scrollTo("identity-scroll-end", anchor: .trailing)
                            }
                        }
                        .onAppear {
                            proxy.scrollTo("identity-scroll-end", anchor: .trailing)
                            isFollowingLive = true
                        }
                    }
                }
                .frame(height: scrollViewportHeight)
            }
        }
        .padding(10)
        .background(Color(white: 0.05))
        .cornerRadius(10)
    }

    private var headerText: String {
        let finalizedCount = finalizedSegments.count
        let tentativeCount = tentativeSegments.count
        let speakerCount = speakerIDs.count
        return "Identity Timeline - Speakers: \(speakerCount) | Segments: \(finalizedCount) + \(tentativeCount)"
    }

    private func drawRows(
        context: inout GraphicsContext,
        size: CGSize,
        cellWidth: CGFloat
    ) {
        guard size.width > 0, size.height > 0 else { return }

        let rowIndexBySpeakerID = Dictionary(uniqueKeysWithValues: speakerIDs.enumerated().map { ($1, $0) })
        let rowDivider = Color.white.opacity(0.08)

        for row in 0...rowCount {
            let y = CGFloat(row) * rowHeight
            var path = Path()
            path.move(to: CGPoint(x: 0, y: y))
            path.addLine(to: CGPoint(x: size.width, y: y))
            context.stroke(path, with: .color(rowDivider), lineWidth: 0.8)
        }

        for segment in finalizedSegments {
            guard let row = rowIndexBySpeakerID[segment.speakerId] else { continue }
            drawIdentitySegment(
                segment,
                row: row,
                cellWidth: cellWidth,
                tentative: false,
                context: &context
            )
        }

        for segment in tentativeSegments {
            guard let row = rowIndexBySpeakerID[segment.speakerId] else { continue }
            drawIdentitySegment(
                segment,
                row: row,
                cellWidth: cellWidth,
                tentative: true,
                context: &context
            )
        }
    }

    private func identitySegmentAt(frame: Int, speakerID: Int) -> SpeakerSegment? {
        guard frame >= 0 else { return nil }

        if let finalized = finalizedSegments.first(where: {
            $0.speakerId == speakerID && frame >= $0.startFrame && frame < $0.endFrame
        }) {
            return finalized
        }

        return tentativeSegments.first(where: {
            $0.speakerId == speakerID && frame >= $0.startFrame && frame < $0.endFrame
        })
    }

    private func drawIdentitySegment(
        _ segment: SpeakerSegment,
        row: Int,
        cellWidth: CGFloat,
        tentative: Bool,
        context: inout GraphicsContext
    ) {
        let x = CGFloat(segment.startFrame) * cellWidth
        let width = max(1, CGFloat(segment.endFrame - segment.startFrame) * cellWidth)
        let y = CGFloat(row) * rowHeight + 2
        let rect = CGRect(x: x, y: y, width: width, height: rowHeight - 4)
        let color = speakerColor(for: segment.speakerId)
        let path = Path(roundedRect: rect, cornerRadius: 3)

        if tentative {
            context.fill(path, with: .color(color.opacity(0.24)))
            context.stroke(
                path,
                with: .color(color.opacity(0.9)),
                style: StrokeStyle(lineWidth: 1.2, dash: [3, 2])
            )
        } else {
            context.fill(path, with: .color(color.opacity(0.55)))
            context.stroke(path, with: .color(color.opacity(0.95)), lineWidth: 1.0)
        }
    }

    private func speakerColor(for speakerID: Int) -> Color {
        let hue = Double((speakerID * 53).quotientAndRemainder(dividingBy: 360).remainder) / 360.0
        return Color(hue: hue, saturation: 0.72, brightness: 0.93)
    }
}
