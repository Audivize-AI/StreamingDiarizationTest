import SwiftUI

private enum DendrogramPalette {
    static let panelTop = Color(red: 0.965, green: 0.976, blue: 0.988)
    static let panelBottom = Color(red: 0.905, green: 0.932, blue: 0.958)
    static let chartTop = Color(red: 0.986, green: 0.992, blue: 0.997)
    static let chartBottom = Color(red: 0.938, green: 0.963, blue: 0.983)
    static let border = Color(red: 0.38, green: 0.54, blue: 0.70)
    static let title = Color(red: 0.09, green: 0.20, blue: 0.32)
    static let subtitle = Color(red: 0.30, green: 0.41, blue: 0.52)
    static let grid = Color(red: 0.29, green: 0.43, blue: 0.57)
    static let threshold = Color(red: 0.84, green: 0.23, blue: 0.16)
    static let thresholdText = Color(red: 0.50, green: 0.17, blue: 0.13)
    static let rootHalo = Color(red: 0.08, green: 0.64, blue: 0.77)
    static let unknownSpeaker = Color(red: 0.42, green: 0.46, blue: 0.54)
    static let speakerSwatches: [Color] = [
        .red, .green, .blue, .orange
    ]

    static func speakerColor(index: Int) -> Color {
        guard index >= 0 else { return unknownSpeaker }
        let swatchIndex = index % speakerSwatches.count
        return speakerSwatches[swatchIndex]
    }
}

struct AHCDendrogramView: View {
    let model: AHCDendrogramModel
    @State private var hoverLocation: CGPoint?

    private var speakerIndices: [Int] {
        let leafSpeakers = model.nodes
            .filter { $0.isLeaf && $0.speakerIndex >= 0 }
            .map(\AHCDendrogramNodeModel.speakerIndex)
        return Array(Set(leafSpeakers)).sorted()
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header

            ZStack {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [DendrogramPalette.chartTop, DendrogramPalette.chartBottom],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .stroke(DendrogramPalette.border.opacity(0.22), lineWidth: 1)
                    )

                if model.isEmpty {
                    emptyState
                } else {
                    GeometryReader { _ in
                        Canvas { context, size in
                            drawDendrogram(in: &context, size: size, hoverLocation: hoverLocation)
                        }
                        .contentShape(Rectangle())
#if os(macOS)
                        .onContinuousHover { phase in
                            switch phase {
                            case .active(let location):
                                hoverLocation = location
                            case .ended:
                                hoverLocation = nil
                            }
                        }
#else
                        .gesture(
                            DragGesture(minimumDistance: 0)
                                .onChanged { value in
                                    hoverLocation = value.location
                                }
                                .onEnded { _ in
                                    hoverLocation = nil
                                }
                        )
#endif
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
                        colors: [DendrogramPalette.panelTop, DendrogramPalette.panelBottom],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(DendrogramPalette.border.opacity(0.34), lineWidth: 1.1)
                )
                .shadow(color: Color.black.opacity(0.10), radius: 8, x: 0, y: 3)
        )
        .animation(.easeInOut(duration: 0.22), value: model.updatedAt)
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Hierarchy")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundStyle(DendrogramPalette.title)
                Text(model.isEmpty ? "Waiting for live merges" : "Live speaker-colored merge tree")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(DendrogramPalette.subtitle)
            }

            Spacer(minLength: 6)

            metricPill(label: "Leaves", value: "\(model.activeLeafCount)", tint: DendrogramPalette.speakerColor(index: 0))
            metricPill(label: "Nodes", value: "\(model.nodes.count)", tint: DendrogramPalette.speakerColor(index: 1))
        }
    }

    private var footer: some View {
        HStack(spacing: 8) {
            if speakerIndices.isEmpty {
                Text("Speaker colors appear once leaves are labeled")
                    .font(.system(size: 10, weight: .medium, design: .rounded))
                    .foregroundStyle(DendrogramPalette.subtitle)
            } else {
                ForEach(Array(speakerIndices.prefix(6)), id: \.self) { speakerIndex in
                    speakerLegendChip(speakerIndex: speakerIndex)
                }

                if speakerIndices.count > 6 {
                    Text("+\(speakerIndices.count - 6)")
                        .font(.system(size: 10, weight: .bold, design: .rounded))
                        .foregroundStyle(DendrogramPalette.subtitle)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(Color.white.opacity(0.52))
                        )
                }
            }

            Spacer(minLength: 4)

            mergeLegendItem()

            Text(String(format: "max %.3f", model.maxMergeDistance))
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(DendrogramPalette.subtitle)
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
            Image(systemName: "tree")
                .font(.system(size: 22, weight: .semibold))
                .foregroundStyle(DendrogramPalette.subtitle.opacity(0.8))
            Text("Dendrogram will appear as merges stream in")
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(DendrogramPalette.subtitle.opacity(0.9))
        }
    }

    private func metricPill(label: String, value: String, tint: Color) -> some View {
        VStack(spacing: 1) {
            Text(label.uppercased())
                .font(.system(size: 8, weight: .bold, design: .monospaced))
                .foregroundStyle(tint.opacity(0.85))
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(DendrogramPalette.title)
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

    private func speakerLegendChip(speakerIndex: Int) -> some View {
        HStack(spacing: 4) {
            Circle()
                .fill(DendrogramPalette.speakerColor(index: speakerIndex))
                .frame(width: 8, height: 8)
            Text("S\(speakerIndex)")
                .font(.system(size: 10, weight: .semibold, design: .monospaced))
                .foregroundStyle(DendrogramPalette.subtitle)
        }
        .padding(.horizontal, 7)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(Color.white.opacity(0.56))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .stroke(DendrogramPalette.speakerColor(index: speakerIndex).opacity(0.34), lineWidth: 0.9)
        )
    }

    private func mergeLegendItem() -> some View {
        HStack(spacing: 5) {
            Capsule()
                .stroke(DendrogramPalette.subtitle.opacity(0.8), style: StrokeStyle(lineWidth: 1.2, dash: [3, 2]))
                .frame(width: 14, height: 5)
            Text("Must-link")
                .font(.system(size: 10, weight: .medium, design: .rounded))
                .foregroundStyle(DendrogramPalette.subtitle)
        }
    }

    private func drawDendrogram(in context: inout GraphicsContext, size: CGSize, hoverLocation: CGPoint?) {
        guard let layout = DendrogramLayout(model: model, size: size) else {
            return
        }

        let plotRect = layout.plotRect
        let guideCount = 4

        for step in 0...guideCount {
            let ratio = CGFloat(step) / CGFloat(guideCount)
            let y = plotRect.maxY - ratio * plotRect.height

            var guidePath = Path()
            guidePath.move(to: CGPoint(x: plotRect.minX, y: y))
            guidePath.addLine(to: CGPoint(x: plotRect.maxX, y: y))
            context.stroke(
                guidePath,
                with: .color(DendrogramPalette.grid.opacity(step == 0 ? 0.58 : 0.22)),
                lineWidth: step == 0 ? 1.0 : 0.7
            )

            let distance = layout.maxDistance * Float(ratio)
            let tickLabel = Text(String(format: "%.2f", distance))
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundColor(DendrogramPalette.subtitle.opacity(0.95))
            context.draw(tickLabel, at: CGPoint(x: plotRect.minX - 6, y: y), anchor: .trailing)
        }

        let axisLabel = Text("distance")
            .font(.system(size: 8, weight: .bold, design: .monospaced))
            .foregroundColor(DendrogramPalette.subtitle.opacity(0.90))
        context.draw(axisLabel, at: CGPoint(x: plotRect.minX - 6, y: plotRect.minY - 6), anchor: .bottomTrailing)

        var regularPathsBySpeaker: [Int: Path] = [:]
        var mustLinkPathsBySpeaker: [Int: Path] = [:]
        var mergeDotsBySpeaker: [Int: Path] = [:]

        for branch in layout.branches {
            guard
                let parentPoint = layout.points[branch.parent],
                let leftPoint = layout.points[branch.left],
                let rightPoint = layout.points[branch.right]
            else {
                continue
            }

            let speakerIndex = layout.dominantSpeakerByNodeId[branch.parent] ?? -1
            if layout.mustLinkNodeIds.contains(branch.parent) {
                var path = mustLinkPathsBySpeaker[speakerIndex] ?? Path()
                appendBranch(to: &path, parentPoint: parentPoint, leftPoint: leftPoint, rightPoint: rightPoint)
                mustLinkPathsBySpeaker[speakerIndex] = path
            } else {
                var path = regularPathsBySpeaker[speakerIndex] ?? Path()
                appendBranch(to: &path, parentPoint: parentPoint, leftPoint: leftPoint, rightPoint: rightPoint)
                regularPathsBySpeaker[speakerIndex] = path
            }

            var dotPath = mergeDotsBySpeaker[speakerIndex] ?? Path()
            dotPath.addEllipse(in: CGRect(x: parentPoint.x - 2.0, y: parentPoint.y - 2.0, width: 4.0, height: 4.0))
            mergeDotsBySpeaker[speakerIndex] = dotPath
        }

        for speakerIndex in regularPathsBySpeaker.keys.sorted() {
            guard let path = regularPathsBySpeaker[speakerIndex] else { continue }
            context.stroke(
                path,
                with: .color(DendrogramPalette.speakerColor(index: speakerIndex).opacity(0.90)),
                style: StrokeStyle(lineWidth: 1.9, lineCap: .round, lineJoin: .round)
            )
        }

        for speakerIndex in mustLinkPathsBySpeaker.keys.sorted() {
            guard let path = mustLinkPathsBySpeaker[speakerIndex] else { continue }
            context.stroke(
                path,
                with: .color(DendrogramPalette.speakerColor(index: speakerIndex)),
                style: StrokeStyle(lineWidth: 2.2, lineCap: .round, lineJoin: .round, dash: [6, 4])
            )
        }

        for speakerIndex in mergeDotsBySpeaker.keys.sorted() {
            guard let dotPath = mergeDotsBySpeaker[speakerIndex] else { continue }
            context.fill(dotPath, with: .color(DendrogramPalette.speakerColor(index: speakerIndex).opacity(0.92)))
        }

        var leafDotsBySpeaker: [Int: Path] = [:]
        for nodeId in layout.leafNodeIds {
            guard let point = layout.points[nodeId] else { continue }
            let speakerIndex = layout.leafSpeakerByNodeId[nodeId] ?? -1
            var path = leafDotsBySpeaker[speakerIndex] ?? Path()
            path.addEllipse(in: CGRect(x: point.x - 1.9, y: point.y - 1.9, width: 3.8, height: 3.8))
            leafDotsBySpeaker[speakerIndex] = path
        }

        for speakerIndex in leafDotsBySpeaker.keys.sorted() {
            guard let path = leafDotsBySpeaker[speakerIndex] else { continue }
            context.fill(path, with: .color(DendrogramPalette.speakerColor(index: speakerIndex)))
        }

        if let rootPoint = layout.points[model.rootIndex] {
            let rootSpeakerIndex = layout.dominantSpeakerByNodeId[model.rootIndex] ?? -1
            let rootColor = DendrogramPalette.speakerColor(index: rootSpeakerIndex)
            let haloRect = CGRect(x: rootPoint.x - 6.0, y: rootPoint.y - 6.0, width: 12.0, height: 12.0)
            context.fill(Path(ellipseIn: haloRect), with: .color(rootColor.opacity(0.20)))
            let markerRect = CGRect(x: rootPoint.x - 2.7, y: rootPoint.y - 2.7, width: 5.4, height: 5.4)
            context.fill(Path(ellipseIn: markerRect), with: .color(rootColor))
        }

        if let hoverLocation {
            let locationIsWithinBounds = hoverLocation.x >= 0 && hoverLocation.y >= 0 && hoverLocation.x <= size.width && hoverLocation.y <= size.height
            if locationIsWithinBounds {
                let y = min(max(hoverLocation.y, plotRect.minY), plotRect.maxY)
                var thresholdPath = Path()
                thresholdPath.move(to: CGPoint(x: plotRect.minX, y: y))
                thresholdPath.addLine(to: CGPoint(x: plotRect.maxX, y: y))
                context.stroke(
                    thresholdPath,
                    with: .color(DendrogramPalette.threshold.opacity(0.92)),
                    style: StrokeStyle(lineWidth: 1.3, lineCap: .round, dash: [5, 4])
                )

                let threshold = layout.distance(atY: y)
                let thresholdLabel = Text(String(format: "linkage %.3f", threshold))
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(DendrogramPalette.thresholdText)
                let labelY = max(plotRect.minY + 12, min(y - 6, plotRect.maxY - 4))
                context.draw(thresholdLabel, at: CGPoint(x: plotRect.maxX - 4, y: labelY), anchor: .bottomTrailing)
            }
        }
    }

    private func appendBranch(to path: inout Path, parentPoint: CGPoint, leftPoint: CGPoint, rightPoint: CGPoint) {
        path.move(to: leftPoint)
        path.addLine(to: CGPoint(x: leftPoint.x, y: parentPoint.y))
        path.move(to: rightPoint)
        path.addLine(to: CGPoint(x: rightPoint.x, y: parentPoint.y))
        path.move(to: CGPoint(x: leftPoint.x, y: parentPoint.y))
        path.addLine(to: CGPoint(x: rightPoint.x, y: parentPoint.y))
    }
}

private struct DendrogramBranch {
    let parent: Int
    let left: Int
    let right: Int
}

private struct DendrogramLayout {
    let points: [Int: CGPoint]
    let branches: [DendrogramBranch]
    let mustLinkNodeIds: Set<Int>
    let maxDistance: Float
    let plotRect: CGRect
    let leafNodeIds: [Int]
    let dominantSpeakerByNodeId: [Int: Int]
    let leafSpeakerByNodeId: [Int: Int]

    init?(model: AHCDendrogramModel, size: CGSize) {
        guard !model.isEmpty else {
            return nil
        }

        let nodesById = model.nodesById
        guard nodesById[model.rootIndex] != nil else {
            return nil
        }

        var reachable = Set<Int>()
        var stack: [Int] = [model.rootIndex]

        while let nodeId = stack.popLast() {
            guard reachable.insert(nodeId).inserted, let node = nodesById[nodeId] else {
                continue
            }

            if node.leftChild >= 0 {
                stack.append(node.leftChild)
            }
            if node.rightChild >= 0 {
                stack.append(node.rightChild)
            }
        }

        if reachable.isEmpty {
            return nil
        }

        let maxInternalDistance = reachable.compactMap { id -> Float? in
            guard let node = nodesById[id], !node.isLeaf else { return nil }
            return max(0, node.mergeDistance)
        }.max() ?? 0
        let normalizedMaxDistance = max(maxInternalDistance, 1e-6)

        var logicalX: [Int: CGFloat] = [:]
        var normalizedY: [Int: CGFloat] = [:]
        var branches: [DendrogramBranch] = []
        var leafNodeIds: [Int] = []
        var leafCursor = 0

        func assignCoordinates(nodeId: Int, path: inout Set<Int>) -> CGFloat {
            guard
                !path.contains(nodeId),
                let node = nodesById[nodeId],
                reachable.contains(nodeId)
            else {
                let x = CGFloat(leafCursor)
                leafCursor += 1
                return x
            }

            path.insert(nodeId)
            defer { path.remove(nodeId) }

            let hasBothChildren = node.leftChild >= 0 && node.rightChild >= 0 && reachable.contains(node.leftChild) && reachable.contains(node.rightChild)

            if node.isLeaf || !hasBothChildren {
                let x = CGFloat(leafCursor)
                leafCursor += 1
                logicalX[nodeId] = x
                normalizedY[nodeId] = 0
                leafNodeIds.append(nodeId)
                return x
            }

            let leftX = assignCoordinates(nodeId: node.leftChild, path: &path)
            let rightX = assignCoordinates(nodeId: node.rightChild, path: &path)
            let x = (leftX + rightX) * 0.5
            logicalX[nodeId] = x
            normalizedY[nodeId] = CGFloat(max(0, node.mergeDistance) / normalizedMaxDistance)
            branches.append(DendrogramBranch(parent: nodeId, left: node.leftChild, right: node.rightChild))
            return x
        }

        var path: Set<Int> = []
        _ = assignCoordinates(nodeId: model.rootIndex, path: &path)

        let leafCount = max(leafCursor, 1)
        let leftPad: CGFloat = 42
        let rightPad: CGFloat = 16
        let topPad: CGFloat = 18
        let bottomPad: CGFloat = 20

        let drawWidth = max(1, size.width - leftPad - rightPad)
        let drawHeight = max(1, size.height - topPad - bottomPad)
        let plotRect = CGRect(x: leftPad, y: topPad, width: drawWidth, height: drawHeight)
        let baseY = plotRect.maxY

        var points: [Int: CGPoint] = [:]
        let xDenominator = max(CGFloat(leafCount - 1), 1)

        for (nodeId, xValue) in logicalX {
            guard let yValue = normalizedY[nodeId] else {
                continue
            }

            let x: CGFloat
            if leafCount == 1 {
                x = plotRect.minX + drawWidth * 0.5
            } else {
                x = plotRect.minX + (xValue / xDenominator) * drawWidth
            }

            let y = baseY - yValue * drawHeight
            points[nodeId] = CGPoint(x: x, y: y)
        }

        let uniqueLeafNodeIds = Array(Set(leafNodeIds))
        branches.sort { lhs, rhs in
            let leftDistance = nodesById[lhs.parent]?.mergeDistance ?? 0
            let rightDistance = nodesById[rhs.parent]?.mergeDistance ?? 0
            return leftDistance < rightDistance
        }

        var leafSpeakerByNodeId: [Int: Int] = [:]
        for nodeId in uniqueLeafNodeIds {
            guard let speakerIndex = nodesById[nodeId]?.speakerIndex, speakerIndex >= 0 else {
                continue
            }
            leafSpeakerByNodeId[nodeId] = speakerIndex
        }

        var speakerHistogramCache: [Int: [Int: Int]] = [:]

        func speakerHistogram(nodeId: Int, path: inout Set<Int>) -> [Int: Int] {
            if let cached = speakerHistogramCache[nodeId] {
                return cached
            }

            guard
                !path.contains(nodeId),
                let node = nodesById[nodeId],
                reachable.contains(nodeId)
            else {
                return [:]
            }

            path.insert(nodeId)
            defer { path.remove(nodeId) }

            let hasBothChildren = node.leftChild >= 0 && node.rightChild >= 0 && reachable.contains(node.leftChild) && reachable.contains(node.rightChild)

            var histogram: [Int: Int] = [:]
            if node.isLeaf || !hasBothChildren {
                if node.speakerIndex >= 0 {
                    histogram[node.speakerIndex, default: 0] += 1
                }
            } else {
                let leftHistogram = speakerHistogram(nodeId: node.leftChild, path: &path)
                let rightHistogram = speakerHistogram(nodeId: node.rightChild, path: &path)

                for (speakerIndex, count) in leftHistogram {
                    histogram[speakerIndex, default: 0] += count
                }
                for (speakerIndex, count) in rightHistogram {
                    histogram[speakerIndex, default: 0] += count
                }
            }

            speakerHistogramCache[nodeId] = histogram
            return histogram
        }

        func dominantSpeaker(from histogram: [Int: Int]) -> Int? {
            histogram.max { lhs, rhs in
                if lhs.value == rhs.value {
                    return lhs.key > rhs.key
                }
                return lhs.value < rhs.value
            }?.key
        }

        var dominantSpeakerByNodeId: [Int: Int] = [:]
        for nodeId in uniqueLeafNodeIds {
            if let speakerIndex = leafSpeakerByNodeId[nodeId] {
                dominantSpeakerByNodeId[nodeId] = speakerIndex
            }
        }

        for branch in branches {
            var trace: Set<Int> = []
            let histogram = speakerHistogram(nodeId: branch.parent, path: &trace)
            if let speakerIndex = dominantSpeaker(from: histogram) {
                dominantSpeakerByNodeId[branch.parent] = speakerIndex
            }
        }

        if dominantSpeakerByNodeId[model.rootIndex] == nil {
            var trace: Set<Int> = []
            let histogram = speakerHistogram(nodeId: model.rootIndex, path: &trace)
            if let speakerIndex = dominantSpeaker(from: histogram) {
                dominantSpeakerByNodeId[model.rootIndex] = speakerIndex
            }
        }

        self.points = points
        self.branches = branches
        self.mustLinkNodeIds = Set(model.nodes.filter(\AHCDendrogramNodeModel.mustLink).map(\AHCDendrogramNodeModel.id))
        self.maxDistance = maxInternalDistance
        self.plotRect = plotRect
        self.leafNodeIds = uniqueLeafNodeIds
        self.dominantSpeakerByNodeId = dominantSpeakerByNodeId
        self.leafSpeakerByNodeId = leafSpeakerByNodeId
    }

    func distance(atY y: CGFloat) -> Float {
        let clampedY = min(max(y, plotRect.minY), plotRect.maxY)
        let normalized = 1 - ((clampedY - plotRect.minY) / max(plotRect.height, 1))
        let ratio = max(0, min(1, normalized))
        return maxDistance * Float(ratio)
    }
}
