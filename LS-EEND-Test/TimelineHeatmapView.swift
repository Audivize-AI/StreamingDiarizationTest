//
//  TimelineHeatmapView.swift
//  LS-EEND-Test
//

import SwiftUI

struct TimelineHeatmapView: View {
    let finalized: [Float]
    let tentative: [Float]
    let finalizedFrameCount: Int
    let tentativeFrameCount: Int
    let numSpeakers: Int
    let frameDurationSeconds: Float
    let speakerNames: [Int: String?]
    let windowSeconds: Double

    /// Viridis colormap stops (purple → blue → teal → green → yellow).
    /// Canonical matplotlib viridis sampled at 9 points.
    private static let viridisStops: [(r: Double, g: Double, b: Double)] = [
        (0.267, 0.005, 0.329),  // 0.000 — dark purple
        (0.283, 0.141, 0.458),  // 0.125
        (0.254, 0.265, 0.530),  // 0.250
        (0.207, 0.372, 0.553),  // 0.375
        (0.164, 0.471, 0.558),  // 0.500 — teal
        (0.128, 0.567, 0.551),  // 0.625
        (0.135, 0.659, 0.518),  // 0.750 — green
        (0.478, 0.821, 0.318),  // 0.875
        (0.993, 0.906, 0.144),  // 1.000 — yellow
    ]

    private static func viridis(_ t: Float) -> Color {
        let clamped = Double(min(max(t, 0), 1))
        let scaled = clamped * Double(viridisStops.count - 1)
        let lo = Int(scaled.rounded(.down))
        let hi = min(lo + 1, viridisStops.count - 1)
        let f = scaled - Double(lo)
        let a = viridisStops[lo], b = viridisStops[hi]
        return Color(
            red: a.r + (b.r - a.r) * f,
            green: a.g + (b.g - a.g) * f,
            blue: a.b + (b.b - a.b) * f
        )
    }

    var body: some View {
        GeometryReader { geom in
            Canvas { ctx, size in
                guard numSpeakers > 0 else { return }
                let totalFrames = finalizedFrameCount + tentativeFrameCount
                guard totalFrames > 0 else {
                    // Still draw empty rows + labels so user sees the slot layout.
                    ctx.fill(Path(CGRect(origin: .zero, size: size)),
                             with: .color(Self.viridis(0)))
                    let rowH = size.height / CGFloat(numSpeakers)
                    for slot in 0..<numSpeakers {
                        let y = CGFloat(slot) * rowH
                        let divider = Path { p in
                            p.move(to: CGPoint(x: 0, y: y))
                            p.addLine(to: CGPoint(x: size.width, y: y))
                        }
                        ctx.stroke(divider, with: .color(.white.opacity(0.15)))
                        let label = speakerNames[slot].flatMap { $0 } ?? "Speaker \(slot)"
                        let text = Text(label).font(.caption2).foregroundColor(.white)
                        ctx.draw(text, at: CGPoint(x: 4, y: y + rowH/2), anchor: .leading)
                    }
                    return
                }

                let framesPerSecond = Float(1.0) / frameDurationSeconds
                let windowFrames = max(
                    1, Int(Float(windowSeconds) * framesPerSecond)
                )
                let startFrame = max(0, totalFrames - windowFrames)
                let visibleFrames = totalFrames - startFrame
                guard visibleFrames > 0 else { return }

                let w = size.width
                let h = size.height
                let rowH = h / CGFloat(numSpeakers)
                let colW = w / CGFloat(visibleFrames)

                // Viridis floor — fill background with viridis(0) so empty
                // slots still look like the "no activity" end of the colormap.
                ctx.fill(Path(CGRect(origin: .zero, size: size)),
                         with: .color(Self.viridis(0)))

                for slot in 0..<numSpeakers {
                    let y = CGFloat(slot) * rowH

                    for f in 0..<visibleFrames {
                        let globalFrame = startFrame + f
                        let isFinalized = globalFrame < finalizedFrameCount
                        let (arr, idx) = isFinalized
                            ? (finalized, globalFrame)
                            : (tentative, globalFrame - finalizedFrameCount)
                        let offset = idx * numSpeakers + slot
                        guard offset >= 0, offset < arr.count else { continue }
                        let p = arr[offset]
                        let rect = CGRect(
                            x: CGFloat(f) * colW,
                            y: y,
                            width: max(colW, 1),
                            height: rowH
                        )
                        let color = Self.viridis(p)
                        let drawColor = isFinalized
                            ? color
                            : color.opacity(0.7)
                        ctx.fill(Path(rect), with: .color(drawColor))
                    }

                    // Row divider + label
                    let divider = Path { p in
                        p.move(to: CGPoint(x: 0, y: y))
                        p.addLine(to: CGPoint(x: w, y: y))
                    }
                    ctx.stroke(divider, with: .color(.white.opacity(0.15)))

                    let label = speakerNames[slot].flatMap { $0 } ?? "Speaker \(slot)"
                    let text = Text(label)
                        .font(.caption2)
                        .foregroundColor(.white)
                    ctx.draw(text, at: CGPoint(x: 4, y: y + rowH/2), anchor: .leading)
                }

                // Tentative region marker
                if tentativeFrameCount > 0 {
                    let xStart = CGFloat(finalizedFrameCount - startFrame) * colW
                    let markerRect = CGRect(
                        x: xStart, y: 0, width: max(w - xStart, 0), height: h
                    )
                    ctx.stroke(
                        Path(markerRect),
                        with: .color(.gray.opacity(0.6)),
                        style: StrokeStyle(lineWidth: 1, dash: [2, 2])
                    )
                }
            }
            .drawingGroup()
            .overlay(alignment: .topTrailing) {
                let totalSec = Double(finalizedFrameCount + tentativeFrameCount)
                    * Double(frameDurationSeconds)
                Text(String(format: "%.1fs", totalSec))
                    .font(.caption2)
                    .padding(4)
                    .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 4))
                    .padding(4)
            }
            .frame(width: geom.size.width, height: geom.size.height)
        }
    }
}
