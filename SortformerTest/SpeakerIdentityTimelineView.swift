import SwiftUI
#if os(macOS)
import AppKit
#endif

/// Segment-only timeline driven by variable speaker identities.
struct SpeakerIdentityTimelineView: View {
    private struct SpeakerRow: Identifiable {
        let speakerID: Int
        let slot: Int?

        var id: Int { speakerID }
    }

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

    private var activeSlotBySpeakerID: [Int: Int] {
        guard let timeline else { return [:] }
        var result: [Int: Int] = [:]
        for (slot, profile) in timeline.activeSpeakers {
            result[profile.speakerId] = slot
        }
        return result
    }

    private var speakerRows: [SpeakerRow] {
        let segmentSpeakerIDs = Set((finalizedSegments + tentativeSegments).map(\.speakerId))
        let sortedActive = activeSlotBySpeakerID.sorted { $0.value < $1.value }
        let activeRows = sortedActive.map { SpeakerRow(speakerID: $0.key, slot: $0.value) }

        let activeIDs = Set(sortedActive.map(\.key))
        let inactiveRows = segmentSpeakerIDs
            .subtracting(activeIDs)
            .sorted()
            .map { SpeakerRow(speakerID: $0, slot: nil) }

        let rows = activeRows + inactiveRows
        if rows.isEmpty {
            return [SpeakerRow(speakerID: -1, slot: nil)]
        }
        return rows
    }

    private var rowCount: Int {
        speakerRows.count
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
        VStack(alignment: .leading, spacing: 6) {
            Text(headerText)
                .font(.caption)
                .foregroundStyle(.secondary)

            GeometryReader { geometry in
                let timelineViewportWidth = max(0, geometry.size.width - labelWidth - 4)
                let cellWidth = timelineViewportWidth / CGFloat(max(visibleFrames, 1))
                let framesToDraw = max(totalFrames, visibleFrames)
                let contentWidth = max(CGFloat(framesToDraw) * cellWidth, timelineViewportWidth)
                let timelineContentHeight = rowHeightTotal + horizontalScrollbarHeight

                ScrollView(.vertical, showsIndicators: true) {
                    HStack(alignment: .top, spacing: 4) {
                        VStack(alignment: .trailing, spacing: 0) {
                            ForEach(0..<rowCount, id: \.self) { row in
                                HStack(spacing: 6) {
                                    Circle()
                                        .fill(labelColor(for: row))
                                        .frame(width: 8, height: 8)
                                    Text(labelText(for: row))
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .trailing)
                                .padding(.trailing, 2)
                                .frame(height: rowHeight)
                            }
                        }
                        .frame(width: labelWidth, height: rowHeightTotal, alignment: .topTrailing)

                        ScrollViewReader { proxy in
                            ScrollView(.horizontal, showsIndicators: true) {
                                Canvas { context, size in
                                    drawRows(
                                        context: &context,
                                        size: size,
                                        cellWidth: cellWidth,
                                        slotBySpeakerID: activeSlotBySpeakerID
                                    )
                                }
                                .contentShape(Rectangle())
                                .onTapGesture(count: 1) { location in
                                    guard !isRecording, cellWidth > 0 else { return }
                                    let row = Int(location.y / rowHeight)
                                    guard row >= 0, row < speakerRows.count else { return }
                                    let speakerID = speakerRows[row].speakerID
                                    guard speakerID >= 0 else { return }
                                    let frame = Int(location.x / cellWidth)
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
                            .frame(width: timelineViewportWidth, height: timelineContentHeight, alignment: .topLeading)
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
                    .frame(height: timelineContentHeight, alignment: .topLeading)
                }
            }
        }
        .padding(10)
        .background(Color(white: 0.05))
        .cornerRadius(10)
    }

    private var headerText: String {
        let finalizedCount = finalizedSegments.count
        let tentativeCount = tentativeSegments.count
        let speakerCount = Set((finalizedSegments + tentativeSegments).map(\.speakerId)).count
        return "Identity Timeline - Speakers: \(speakerCount) | Segments: \(finalizedCount) + \(tentativeCount)"
    }

    private func labelText(for row: Int) -> String {
        guard row < speakerRows.count else { return "No data" }
        let rowInfo = speakerRows[row]
        guard rowInfo.speakerID >= 0 else { return "No data" }
        return "ID \(rowInfo.speakerID)"
    }

    private func labelColor(for row: Int) -> Color {
        guard row < speakerRows.count else { return .gray.opacity(0.45) }
        let rowInfo = speakerRows[row]
        guard rowInfo.speakerID >= 0 else { return .gray.opacity(0.45) }
        return speakerColor(for: rowInfo.speakerID, slotBySpeakerID: activeSlotBySpeakerID)
    }

    private func drawRows(
        context: inout GraphicsContext,
        size: CGSize,
        cellWidth: CGFloat,
        slotBySpeakerID: [Int: Int]
    ) {
        guard size.width > 0, size.height > 0 else { return }
        let validRows = speakerRows.enumerated().filter { $0.element.speakerID >= 0 }
        let rowIndexBySpeakerID = Dictionary(uniqueKeysWithValues: validRows.map { ($0.element.speakerID, $0.offset) })
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
                slotBySpeakerID: slotBySpeakerID,
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
                slotBySpeakerID: slotBySpeakerID,
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
        slotBySpeakerID: [Int: Int],
        context: inout GraphicsContext
    ) {
        let x = CGFloat(segment.startFrame) * cellWidth
        let width = max(1, CGFloat(segment.endFrame - segment.startFrame) * cellWidth)
        let y = CGFloat(row) * rowHeight + 2
        let rect = CGRect(x: x, y: y, width: width, height: rowHeight - 4)
        let color = speakerColor(for: segment.speakerId, slotBySpeakerID: slotBySpeakerID)
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

    private func speakerColor(for speakerID: Int, slotBySpeakerID: [Int: Int]) -> Color {
        if let slot = slotBySpeakerID[speakerID] {
            return SpeakerTimelinePalette.slotColor(for: slot)
        }
        return SpeakerTimelinePalette.slotColor(for: speakerID)
    }
}
