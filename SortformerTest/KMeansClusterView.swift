import SwiftUI

private enum KMeansPalette {
    static let panelTop = Color(red: 0.964, green: 0.974, blue: 0.987)
    static let panelBottom = Color(red: 0.897, green: 0.926, blue: 0.958)
    static let chartTop = Color(red: 0.985, green: 0.993, blue: 0.999)
    static let chartBottom = Color(red: 0.932, green: 0.960, blue: 0.985)
    static let border = Color(red: 0.35, green: 0.53, blue: 0.69)
    static let title = Color(red: 0.09, green: 0.20, blue: 0.31)
    static let subtitle = Color(red: 0.30, green: 0.41, blue: 0.52)
    static let grid = Color(red: 0.32, green: 0.46, blue: 0.60)
    static let finalizedCentroid = Color(red: 0.08, green: 0.59, blue: 0.36)
    static let tentativeCentroid = Color(red: 0.24, green: 0.44, blue: 0.82)
    static let outlier = Color(red: 0.82, green: 0.24, blue: 0.20)
    static let unknownSpeaker = Color(red: 0.43, green: 0.47, blue: 0.54)
    static let speakerSwatches: [Color] = [.red, .green, .blue, .orange]

    static func speakerColor(index: Int) -> Color {
        guard index >= 0 else { return unknownSpeaker }
        return speakerSwatches[index % speakerSwatches.count]
    }
}

struct KMeansClusterView: View {
    let model: KMeansClusterPlotModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header

            ZStack {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [KMeansPalette.chartTop, KMeansPalette.chartBottom],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .stroke(KMeansPalette.border.opacity(0.22), lineWidth: 1)
                    )

                if model.isEmpty {
                    emptyState
                } else {
                    GeometryReader { _ in
                        Canvas { context, size in
                            drawPlot(in: &context, size: size)
                        }
                    }
                    .padding(.horizontal, 6)
                    .padding(.vertical, 8)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            footer
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [KMeansPalette.panelTop, KMeansPalette.panelBottom],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(KMeansPalette.border.opacity(0.34), lineWidth: 1.1)
                )
                .shadow(color: Color.black.opacity(0.10), radius: 8, x: 0, y: 3)
        )
        .animation(.easeInOut(duration: 0.22), value: model.updatedAt)
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text("K-means")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundStyle(KMeansPalette.title)
                Text(model.isEmpty ? "Waiting for embeddings" : "Finalized + tentative centroid tracking")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(KMeansPalette.subtitle)
            }

            Spacer(minLength: 6)

            metricPill(label: "Points", value: "\(model.points.count)", tint: KMeansPalette.speakerColor(index: 0))
            metricPill(label: "Outliers", value: "\(model.outlierCount)", tint: KMeansPalette.outlier)
        }
    }

    private var footer: some View {
        HStack(spacing: 8) {
            HStack(spacing: 5) {
                Rectangle()
                    .fill(KMeansPalette.finalizedCentroid)
                    .frame(width: 9, height: 9)
                Text("Finalized centroid")
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .foregroundStyle(KMeansPalette.subtitle)
            }

            HStack(spacing: 5) {
                DiamondShape()
                    .fill(KMeansPalette.tentativeCentroid)
                    .frame(width: 9, height: 9)
                Text("Tentative centroid")
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .foregroundStyle(KMeansPalette.subtitle)
            }

            HStack(spacing: 5) {
                Circle()
                    .stroke(KMeansPalette.outlier, lineWidth: 1.3)
                    .frame(width: 9, height: 9)
                Text("Outlier")
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .foregroundStyle(KMeansPalette.subtitle)
            }

            Spacer(minLength: 4)

            Text("F: \(model.finalizedCentroids.count)  T: \(model.tentativeCentroids.count)")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(KMeansPalette.subtitle)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(Color.white.opacity(0.55))
                )
        }
    }

    private var emptyState: some View {
        VStack(spacing: 7) {
            Image(systemName: "chart.scatter")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(KMeansPalette.subtitle.opacity(0.8))
            Text("Cluster plot will appear as embeddings stream in")
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(KMeansPalette.subtitle.opacity(0.9))
        }
    }

    private func metricPill(label: String, value: String, tint: Color) -> some View {
        VStack(spacing: 1) {
            Text(label.uppercased())
                .font(.system(size: 8, weight: .bold, design: .monospaced))
                .foregroundStyle(tint.opacity(0.85))
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(KMeansPalette.title)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(Color.white.opacity(0.60))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .stroke(tint.opacity(0.28), lineWidth: 0.9)
        )
    }

    private func drawPlot(in context: inout GraphicsContext, size: CGSize) {
        let allCoordinates = model.points.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            + model.finalizedCentroids.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }
            + model.tentativeCentroids.map { CGPoint(x: CGFloat($0.x), y: CGFloat($0.y)) }

        guard !allCoordinates.isEmpty else {
            return
        }

        let margin: CGFloat = 22
        let plotRect = CGRect(
            x: margin,
            y: margin,
            width: max(10, size.width - 2 * margin),
            height: max(10, size.height - 2 * margin)
        )

        let minX = allCoordinates.map(\.x).min() ?? 0
        let maxX = allCoordinates.map(\.x).max() ?? 0
        let minY = allCoordinates.map(\.y).min() ?? 0
        let maxY = allCoordinates.map(\.y).max() ?? 0

        let xSpan = max(0.0001, maxX - minX)
        let ySpan = max(0.0001, maxY - minY)

        func toPlot(_ x: CGFloat, _ y: CGFloat) -> CGPoint {
            let normalizedX = (x - minX) / xSpan
            let normalizedY = (y - minY) / ySpan
            return CGPoint(
                x: plotRect.minX + normalizedX * plotRect.width,
                y: plotRect.maxY - normalizedY * plotRect.height
            )
        }

        for step in 0...4 {
            let ratio = CGFloat(step) / 4
            let x = plotRect.minX + ratio * plotRect.width
            let y = plotRect.minY + ratio * plotRect.height

            var vertical = Path()
            vertical.move(to: CGPoint(x: x, y: plotRect.minY))
            vertical.addLine(to: CGPoint(x: x, y: plotRect.maxY))
            context.stroke(vertical, with: .color(KMeansPalette.grid.opacity(step == 0 ? 0.28 : 0.14)), lineWidth: 0.7)

            var horizontal = Path()
            horizontal.move(to: CGPoint(x: plotRect.minX, y: y))
            horizontal.addLine(to: CGPoint(x: plotRect.maxX, y: y))
            context.stroke(horizontal, with: .color(KMeansPalette.grid.opacity(step == 0 ? 0.28 : 0.14)), lineWidth: 0.7)
        }

        let finalizedLookup = Dictionary(
            uniqueKeysWithValues: model.finalizedCentroids.map { ("\($0.slot)-\($0.clusterIndex)", $0) }
        )
        for centroid in model.tentativeCentroids {
            let key = "\(centroid.slot)-\(centroid.clusterIndex)"
            guard let anchor = finalizedLookup[key] else { continue }

            let start = toPlot(CGFloat(anchor.x), CGFloat(anchor.y))
            let end = toPlot(CGFloat(centroid.x), CGFloat(centroid.y))
            var path = Path()
            path.move(to: start)
            path.addLine(to: end)
            context.stroke(
                path,
                with: .color(KMeansPalette.border.opacity(0.48)),
                style: StrokeStyle(lineWidth: 1.0, dash: [3, 2])
            )
        }

        var boundaryCentroidsByKey: [String: KMeansCentroidModel] = Dictionary(
            uniqueKeysWithValues: model.finalizedCentroids.map { ("\($0.slot)-\($0.clusterIndex)", $0) }
        )
        for centroid in model.tentativeCentroids {
            boundaryCentroidsByKey["\(centroid.slot)-\(centroid.clusterIndex)"] = centroid
        }

        for centroid in boundaryCentroidsByKey.values {
            let center = toPlot(CGFloat(centroid.x), CGFloat(centroid.y))
            let clusterPoints = model.points.filter {
                $0.slot == centroid.slot && $0.clusterIndex == centroid.clusterIndex
            }

            let maxPointDistance = clusterPoints.reduce(CGFloat.zero) { currentMax, point in
                let p = toPlot(CGFloat(point.x), CGFloat(point.y))
                let dx = p.x - center.x
                let dy = p.y - center.y
                return max(currentMax, sqrt(dx * dx + dy * dy))
            }

            let minRadius: CGFloat = 9
            let radius: CGFloat = max(minRadius, maxPointDistance + 5)
            let circle = CGRect(
                x: center.x - radius,
                y: center.y - radius,
                width: radius * 2,
                height: radius * 2
            )
            let boundaryColor = KMeansPalette.speakerColor(index: centroid.slot).opacity(0.55)

            context.stroke(
                Path(ellipseIn: circle),
                with: .color(boundaryColor),
                style: StrokeStyle(lineWidth: 1.05, dash: [4, 3])
            )
        }

        for point in model.points {
            let center = toPlot(CGFloat(point.x), CGFloat(point.y))
            let radius: CGFloat = point.isTentative ? 2.4 : 3.0
            let rect = CGRect(x: center.x - radius, y: center.y - radius, width: radius * 2, height: radius * 2)
            let fillColor = KMeansPalette.speakerColor(index: point.slot).opacity(point.isTentative ? 0.45 : 0.84)
            context.fill(Path(ellipseIn: rect), with: .color(fillColor))

            if point.isOutlier {
                let outlierRect = rect.insetBy(dx: -2, dy: -2)
                context.stroke(Path(ellipseIn: outlierRect), with: .color(KMeansPalette.outlier), lineWidth: 1.25)
            }
        }

        for centroid in model.finalizedCentroids {
            let center = toPlot(CGFloat(centroid.x), CGFloat(centroid.y))
            let side: CGFloat = 9
            let rect = CGRect(x: center.x - side / 2, y: center.y - side / 2, width: side, height: side)
            context.fill(Path(rect), with: .color(KMeansPalette.finalizedCentroid))
            context.stroke(Path(rect), with: .color(.white.opacity(0.9)), lineWidth: 0.7)
        }

        for centroid in model.tentativeCentroids {
            let center = toPlot(CGFloat(centroid.x), CGFloat(centroid.y))
            let size: CGFloat = 10
            let diamond = DiamondPath(center: center, radius: size / 2)
            context.fill(diamond.path, with: .color(KMeansPalette.tentativeCentroid.opacity(0.95)))
            context.stroke(diamond.path, with: .color(.white.opacity(0.9)), lineWidth: 0.7)
        }
    }
}

private struct DiamondShape: Shape {
    func path(in rect: CGRect) -> Path {
        DiamondPath(center: CGPoint(x: rect.midX, y: rect.midY), radius: min(rect.width, rect.height) / 2).path
    }
}

private struct DiamondPath {
    let path: Path

    init(center: CGPoint, radius: CGFloat) {
        var path = Path()
        path.move(to: CGPoint(x: center.x, y: center.y - radius))
        path.addLine(to: CGPoint(x: center.x + radius, y: center.y))
        path.addLine(to: CGPoint(x: center.x, y: center.y + radius))
        path.addLine(to: CGPoint(x: center.x - radius, y: center.y))
        path.closeSubpath()
        self.path = path
    }
}
