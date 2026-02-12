//
//  ClusterPlotView.swift
//  SortformerTest
//
//  Static UMAP plot of spectral clusters with interactive features.
//

import SwiftUI
import simd

/// Shape types for different speakers
public enum SpeakerShape: Int, CaseIterable {
    case circle = 0
    case square = 1
    case triangle = 2
    case diamond = 3
    
    /// Draw the shape at the given position
    func path(at center: CGPoint, radius: CGFloat) -> Path {
        switch self {
        case .circle:
            return Path(ellipseIn: CGRect(
                x: center.x - radius,
                y: center.y - radius,
                width: radius * 2,
                height: radius * 2
            ))
            
        case .square:
            let size = radius * 1.6  // Slightly larger to match circle visual weight
            return Path(CGRect(
                x: center.x - size / 2,
                y: center.y - size / 2,
                width: size,
                height: size
            ))
            
        case .triangle:
            var path = Path()
            let h = radius * 1.8
            path.move(to: CGPoint(x: center.x, y: center.y - h * 0.6))
            path.addLine(to: CGPoint(x: center.x - h * 0.5, y: center.y + h * 0.4))
            path.addLine(to: CGPoint(x: center.x + h * 0.5, y: center.y + h * 0.4))
            path.closeSubpath()
            return path
            
        case .diamond:
            var path = Path()
            let size = radius * 1.5
            path.move(to: CGPoint(x: center.x, y: center.y - size))
            path.addLine(to: CGPoint(x: center.x + size * 0.7, y: center.y))
            path.addLine(to: CGPoint(x: center.x, y: center.y + size))
            path.addLine(to: CGPoint(x: center.x - size * 0.7, y: center.y))
            path.closeSubpath()
            return path
        }
    }
}

/// Static cluster plot view showing UMAP visualization of spectral clusters
public struct ClusterPlotView: View {
    @ObservedObject var model: EmbeddingGraphModel
    
    /// Callback when a node is selected - provides startFrame, endFrame, speakerIndex
    var onSelectNode: ((Int, Int, Int) -> Void)?
    
    /// Callback to close the plot
    var onClose: (() -> Void)?
    
    // MARK: - State
    
    @State private var scale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    @State private var dragOffset: CGSize = .zero
    @State private var hoveredEdge: (edge: GraphEdge, position: CGPoint)? = nil
    @State private var isShiftPressed = false
    @State private var showingSettings = false
    
    // Node size
    private let baseNodeRadius: CGFloat = 7
    private let selectedNodeRadius: CGFloat = 11
    
    // Cluster colors (distinct, pleasing palette)
    private let clusterColors: [Color] = [
        Color(hue: 0.0, saturation: 0.75, brightness: 0.9),   // Red
        Color(hue: 0.35, saturation: 0.75, brightness: 0.85), // Green
        Color(hue: 0.6, saturation: 0.75, brightness: 0.9),   // Blue
        Color(hue: 0.08, saturation: 0.8, brightness: 0.95),  // Orange
        Color(hue: 0.8, saturation: 0.65, brightness: 0.85),  // Purple
        Color(hue: 0.5, saturation: 0.7, brightness: 0.85),   // Cyan
        Color(hue: 0.15, saturation: 0.8, brightness: 0.95),  // Yellow
        Color(hue: 0.9, saturation: 0.6, brightness: 0.9),    // Pink
        Color(hue: 0.45, saturation: 0.6, brightness: 0.8),   // Teal
        Color(hue: 0.7, saturation: 0.5, brightness: 0.75),   // Lavender
    ]
    
    public init(model: EmbeddingGraphModel, 
                onSelectNode: ((Int, Int, Int) -> Void)? = nil,
                onClose: (() -> Void)? = nil) {
        self.model = model
        self.onSelectNode = onSelectNode
        self.onClose = onClose
    }
    
    public var body: some View {
        VStack(spacing: 8) {
            // Header with title, stats, and close button
            HStack {
                Text("Spectral Cluster Plot")
                    .font(.headline)
                
                Spacer()
                
                let clusterCount = Set(model.nodes.compactMap { $0.clusterLabel }).count
                Text("\(model.nodes.count) embeddings • \(clusterCount) clusters")
                    .font(.caption)
                    .foregroundColor(.secondary)
                
                // Settings button
                Button(action: { showingSettings.toggle() }) {
                    Image(systemName: "gearshape.fill")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
                .help("Graph settings")
                .popover(isPresented: $showingSettings) {
                    VStack(alignment: .leading, spacing: 12) {
                        Toggle("Show All Edges", isOn: $model.config.showAllEdges)
                        
                        Divider()
                        
                        Text("UMAP Parameters").font(.headline)
                        
                        VStack(alignment: .leading) {
                            Text("Neighbors: \(model.config.umapNeighbors)")
                            Slider(value: Binding(get: { Double(model.config.umapNeighbors) }, set: { model.config.umapNeighbors = Int($0) }), in: 2...50, step: 1)
                        }
                        
                        VStack(alignment: .leading) {
                            Text("Min Dist: \(model.config.umapMinDist, specifier: "%.2f")")
                            Slider(value: $model.config.umapMinDist, in: 0.01...1.0)
                        }
                        
                        VStack(alignment: .leading) {
                            Text("Epochs: \(model.config.umapEpochs)")
                            Slider(value: Binding(get: { Double(model.config.umapEpochs) }, set: { model.config.umapEpochs = Int($0) }), in: 50...500, step: 10)
                        }
                        
                        VStack(alignment: .leading) {
                            Text("Neg Samples: \(model.config.numNeg)")
                            Slider(value: Binding(get: { Double(model.config.numNeg) }, set: { model.config.numNeg = Int($0) }), in: 1...20, step: 1)
                        }
                    }
                    .padding()
                    .frame(width: 300)
                }
                
                // Close button
                Button(action: { onClose?() }) {
                    Image(systemName: "xmark.circle.fill")
                        .font(.title3)
                        .foregroundColor(.secondary)
                }
                .buttonStyle(.plain)
                .help("Close cluster plot")
            }
            .padding(.horizontal, 8)
            
            // Main cluster canvas
            GeometryReader { geometry in
                let canvasSize = geometry.size
                
                ZStack {
                    // Background
                    Color(white: 0.08)
                    
                    // Canvas for edges and nodes
                    Canvas { context, size in
                        drawClusterPlot(context: context, size: size)
                    }
                    .gesture(dragGesture)
                    .gesture(magnifyGesture)
                    .onContinuousHover { phase in
                        handleHover(phase: phase, canvasSize: canvasSize)
                    }
                    .onTapGesture { location in
                        handleTap(location: location, canvasSize: canvasSize)
                    }
                    
                    // Tooltip overlay for nodes
                    if let hoveredIdx = model.hoveredNodeIndex, hoveredIdx < model.nodes.count, !isShiftPressed {
                        let node = model.nodes[hoveredIdx]
                        nodeTooltipView(for: node)
                            .position(nodeScreenPosition(node, in: canvasSize))
                            .offset(y: -40)
                    }
                    
                    // Tooltip overlay for edges (prioritized when shift is pressed)
                    if let edgeInfo = hoveredEdge {
                        edgeTooltipView(for: edgeInfo.edge)
                            .position(edgeInfo.position)
                            .offset(y: -25)
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .background(
                    // Track keyboard modifiers
                    KeyboardModifierReader { modifiers in
                        isShiftPressed = modifiers.contains(.shift)
                    }
                )
            }
            
            // Legend and controls
            VStack(spacing: 6) {
                // Cluster color legend
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 12) {
                        let uniqueClusters = Set(model.nodes.compactMap { $0.clusterLabel }).sorted()
                        ForEach(uniqueClusters, id: \.self) { cluster in
                            HStack(spacing: 4) {
                                Circle()
                                    .fill(clusterColors[cluster % clusterColors.count])
                                    .frame(width: 10, height: 10)
                                Text("Cluster \(cluster)")
                                    .font(.caption2)
                                    .foregroundColor(.secondary)
                            }
                        }
                    }
                    .padding(.horizontal, 4)
                }
                .frame(height: 20)
                
                // Speaker shape legend
                HStack(spacing: 12) {
                    Text("Speakers:")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    
                    ForEach(0..<4) { speakerIndex in
                        HStack(spacing: 4) {
                            speakerShapeIcon(for: speakerIndex)
                            Text("Spk \(speakerIndex)")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    Spacer()
                    
                    // Reset view button
                    Button(action: resetView) {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                    .help("Reset view")
                    
                    Text("Hold ⇧ for edges")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
            }
            .padding(.horizontal, 8)
        }
        .padding(8)
        .background(Color(white: 0.05))
        .cornerRadius(12)
    }
    
    // MARK: - Speaker Shape Icon View
    
    @ViewBuilder
    private func speakerShapeIcon(for speakerIndex: Int) -> some View {
        let shape = SpeakerShape(rawValue: speakerIndex % SpeakerShape.allCases.count) ?? .circle
        Canvas { context, size in
            let center = CGPoint(x: size.width / 2, y: size.height / 2)
            let path = shape.path(at: center, radius: 5)
            context.fill(path, with: .color(.gray))
        }
        .frame(width: 12, height: 12)
    }
    
    // MARK: - Drawing
    
    private func drawClusterPlot(context: GraphicsContext, size: CGSize) {
        let nodes = model.nodes
        guard !nodes.isEmpty else {
            // Draw empty state
            let text = Text("No cluster data")
                .font(.caption)
                .foregroundColor(.secondary)
            context.draw(context.resolve(text), at: CGPoint(x: size.width / 2, y: size.height / 2), anchor: .center)
            return
        }
        
        // Draw edges (thresheld by distance)
        drawEdges(context: context, size: size)
        
        // Draw nodes with shapes based on speaker and colors based on cluster
        drawNodes(context: context, size: size)
    }
    
    private func drawEdges(context: GraphicsContext, size: CGSize) {
        // Only draw edges for selected/hovered node, or if we have few nodes
        let edgesToDraw: [GraphEdge]
        
        if model.config.showAllEdges {
            edgesToDraw = model.thresholdedEdges
        } else if let selectedIdx = model.selectedNodeIndex ?? model.hoveredNodeIndex {
            edgesToDraw = model.edgesForNode(selectedIdx)
                .filter { $0.distance <= model.config.distanceThreshold }
        } else if model.nodes.count < 30 {
            // Show all edges for small graphs
            edgesToDraw = model.thresholdedEdges
        } else {
            return
        }
        
        for edge in edgesToDraw {
            guard edge.u < model.nodes.count && edge.v < model.nodes.count else { continue }
            
            let nodeU = model.nodes[edge.u]
            let nodeV = model.nodes[edge.v]
            
            let posU = nodeScreenPosition(nodeU, in: size)
            let posV = nodeScreenPosition(nodeV, in: size)
            
            var path = Path()
            path.move(to: posU)
            path.addLine(to: posV)
            
            // Edge visibility based on distance (closer = more visible)
            let normalizedDist = min(max(edge.distance / model.config.distanceThreshold, 0), 1.0)
            let opacity = Double(1.0 - normalizedDist * 0.7)
            
            // Color based on speaker index
            let colorU = speakerColor(for: nodeU.speakerIndex)
            let colorV = speakerColor(for: nodeV.speakerIndex)
            
            let shading: GraphicsContext.Shading
            if nodeU.speakerIndex == nodeV.speakerIndex {
                shading = .color(colorU.opacity(opacity))
            } else {
                shading = .linearGradient(
                     Gradient(colors: [colorU.opacity(opacity), colorV.opacity(opacity)]),
                     startPoint: posU,
                     endPoint: posV
                 )
            }
            
            context.stroke(path, with: shading,
                          style: StrokeStyle(lineWidth: CGFloat(1.0 + (1.0 - normalizedDist) * 2.0)))
        }
    }
    
    private func speakerColor(for speakerIndex: Int) -> Color {
        return clusterColors[speakerIndex % clusterColors.count]
    }
    
    private func drawNodes(context: GraphicsContext, size: CGSize) {
        for (idx, node) in model.nodes.enumerated() {
            let pos = nodeScreenPosition(node, in: size)
            
            // Determine node appearance
            let isSelected = model.selectedNodeIndex == idx
            let isHovered = model.hoveredNodeIndex == idx
            let radius = (isSelected || isHovered) ? selectedNodeRadius : baseNodeRadius
            
            // Color by cluster
            let clusterLabel = node.clusterLabel ?? 0
            let color = clusterColors[clusterLabel % clusterColors.count]
            
            // Shape by speaker
            let shape = SpeakerShape(rawValue: node.speakerIndex % SpeakerShape.allCases.count) ?? .circle
            let shapePath = shape.path(at: pos, radius: radius)
            
            // Draw glow for selected/hovered
            if isSelected || isHovered {
                let glowShape = shape.path(at: pos, radius: radius * 1.5)
                context.fill(glowShape, with: .color(color.opacity(0.3)))
            }
            
            // Draw shape
            context.fill(shapePath, with: .color(color))
            
            // Draw border
            context.stroke(shapePath,
                          with: .color(isSelected ? .white : color.opacity(0.8)),
                          style: StrokeStyle(lineWidth: isSelected ? 2 : 1))
        }
    }
    
    // MARK: - Coordinate Transforms
    
    private func nodeScreenPosition(_ node: GraphNode, in size: CGSize) -> CGPoint {
        let centerX = size.width / 2
        let centerY = size.height / 2
        let graphScale = min(size.width, size.height) * 0.42 * scale * CGFloat(model.config.positionScale) * CGFloat(model.autoScaleFactor)
        
        let totalOffset = CGSize(
            width: offset.width + dragOffset.width,
            height: offset.height + dragOffset.height
        )
        
        return CGPoint(
            x: centerX + CGFloat(node.position.x) * graphScale + totalOffset.width,
            y: centerY + CGFloat(node.position.y) * graphScale + totalOffset.height
        )
    }
    
    private func screenToGraph(_ point: CGPoint, in size: CGSize) -> SIMD2<Float> {
        let centerX = size.width / 2
        let centerY = size.height / 2
        let graphScale = min(size.width, size.height) * 0.42 * scale * CGFloat(model.config.positionScale) * CGFloat(model.autoScaleFactor)
        
        let totalOffset = CGSize(
            width: offset.width + dragOffset.width,
            height: offset.height + dragOffset.height
        )
        
        return SIMD2<Float>(
            Float((point.x - centerX - totalOffset.width) / graphScale),
            Float((point.y - centerY - totalOffset.height) / graphScale)
        )
    }
    
    private func graphToScreen(_ pos: SIMD2<Float>, in size: CGSize) -> CGPoint {
        let centerX = size.width / 2
        let centerY = size.height / 2
        let graphScale = min(size.width, size.height) * 0.42 * scale * CGFloat(model.config.positionScale) * CGFloat(model.autoScaleFactor)
        
        let totalOffset = CGSize(
            width: offset.width + dragOffset.width,
            height: offset.height + dragOffset.height
        )
        
        return CGPoint(
            x: centerX + CGFloat(pos.x) * graphScale + totalOffset.width,
            y: centerY + CGFloat(pos.y) * graphScale + totalOffset.height
        )
    }
    
    // MARK: - Gestures
    
    private var dragGesture: some Gesture {
        DragGesture()
            .onChanged { value in
                dragOffset = value.translation
            }
            .onEnded { value in
                offset = CGSize(
                    width: offset.width + value.translation.width,
                    height: offset.height + value.translation.height
                )
                dragOffset = .zero
            }
    }
    
    private var magnifyGesture: some Gesture {
        MagnifyGesture()
            .onChanged { value in
                scale = max(0.5, min(5.0, value.magnification))
            }
    }
    
    private func handleHover(phase: HoverPhase, canvasSize: CGSize) {
        switch phase {
        case .active(let location):
            let graphPos = screenToGraph(location, in: canvasSize)
            
            if isShiftPressed {
                // Prioritize edge hover when shift is pressed - check ALL edges
                hoveredEdge = findNearestEdge(at: graphPos, in: canvasSize, forceCheckAll: true)
                model.hoveredNodeIndex = nil
            } else {
                // Find nearest node
                var nearestIdx: Int? = nil
                var nearestDist: Float = 0.1
                
                for (idx, node) in model.nodes.enumerated() {
                    let dist = simd_distance(graphPos, node.position)
                    if dist < nearestDist {
                        nearestDist = dist
                        nearestIdx = idx
                    }
                }
                
                model.hoveredNodeIndex = nearestIdx
                
                // Only check edges if not hovering a node
                if nearestIdx == nil {
                    hoveredEdge = findNearestEdge(at: graphPos, in: canvasSize, forceCheckAll: false)
                } else {
                    hoveredEdge = nil
                }
            }
            
        case .ended:
            model.hoveredNodeIndex = nil
            hoveredEdge = nil
        }
    }
    
    private func findNearestEdge(at graphPos: SIMD2<Float>, in canvasSize: CGSize, forceCheckAll: Bool = false) -> (edge: GraphEdge, position: CGPoint)? {
        // Use larger threshold when shift is pressed for easier edge detection
        let threshold: Float = forceCheckAll ? 0.08 : 0.05
        
        var nearestEdge: GraphEdge? = nil
        var nearestDist: Float = threshold
        var nearestPoint: SIMD2<Float> = .zero
        
        // When forceCheckAll (shift pressed), check all thresholded edges
        // Otherwise only check edges from selected/hovered node or all if small graph
        let edgesToCheck: [GraphEdge]
        if forceCheckAll {
            // When shift is pressed, always check all thresholded edges
            edgesToCheck = model.thresholdedEdges
        } else if let selectedIdx = model.selectedNodeIndex ?? model.hoveredNodeIndex {
            edgesToCheck = model.edgesForNode(selectedIdx)
                .filter { $0.distance <= model.config.distanceThreshold }
        } else if model.nodes.count < 30 {
            edgesToCheck = model.thresholdedEdges
        } else {
            return nil
        }
        
        for edge in edgesToCheck {
            guard edge.u < model.nodes.count && edge.v < model.nodes.count else { continue }
            
            let p1 = model.nodes[edge.u].position
            let p2 = model.nodes[edge.v].position
            
            let lineVec = p2 - p1
            let lineLen = simd_length(lineVec)
            guard lineLen > 0.001 else { continue }
            
            let lineDir = lineVec / lineLen
            let pointVec = graphPos - p1
            let t = simd_clamp(simd_dot(pointVec, lineDir), 0, lineLen)
            let closestPoint = p1 + lineDir * t
            
            let dist = simd_distance(graphPos, closestPoint)
            if dist < nearestDist {
                nearestDist = dist
                nearestEdge = edge
                nearestPoint = closestPoint
            }
        }
        
        if let edge = nearestEdge {
            let screenPos = graphToScreen(nearestPoint, in: canvasSize)
            return (edge: edge, position: screenPos)
        }
        
        return nil
    }
    
    private func handleTap(location: CGPoint, canvasSize: CGSize) {
        let graphPos = screenToGraph(location, in: canvasSize)
        
        // Find nearest node
        var nearestIdx: Int? = nil
        var nearestDist: Float = 0.15
        
        for (idx, node) in model.nodes.enumerated() {
            let dist = simd_distance(graphPos, node.position)
            if dist < nearestDist {
                nearestDist = dist
                nearestIdx = idx
            }
        }
        
        if let idx = nearestIdx {
            let node = model.nodes[idx]
            // Toggle selection
            if model.selectedNodeIndex == idx {
                model.selectByNodeIndex(nil)
            } else {
                model.selectByNodeIndex(idx)
                // Notify to navigate to timeline
                onSelectNode?(node.startFrame, node.endFrame, node.speakerIndex)
            }
        } else {
            // Deselect if clicking empty space
            model.selectByNodeIndex(nil)
        }
    }
    
    private func resetView() {
        withAnimation(.easeOut(duration: 0.3)) {
            scale = 1.0
            offset = .zero
            dragOffset = .zero
        }
    }
    
    // MARK: - Tooltips
    
    private func nodeTooltipView(for node: GraphNode) -> some View {
        let startTime = String(format: "%.2f", Float(node.startFrame) * 0.08)
        let endTime = String(format: "%.2f", Float(node.endFrame) * 0.08)
        let clusterLabel = node.clusterLabel ?? -1
        
        return VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text("Cluster \(clusterLabel)")
                    .font(.caption.bold())
                Text("• Spk \(node.speakerIndex)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            Text("\(startTime)s - \(endTime)s")
                .font(.caption2)
        }
        .padding(6)
        .background(Color.black.opacity(0.85))
        .foregroundColor(.white)
        .cornerRadius(4)
    }
    
    private func edgeTooltipView(for edge: GraphEdge) -> some View {
        let similarity = model.cosineSimilarity(u: edge.u, v: edge.v)
        let distance = 1.0 - similarity
        
        return VStack(alignment: .center, spacing: 2) {
            Text("Cosine Distance")
                .font(.caption2)
                .foregroundColor(.secondary)
            Text(String(format: "%.3f", distance))
                .font(.caption.bold())
        }
        .padding(6)
        .background(Color.black.opacity(0.85))
        .foregroundColor(.white)
        .cornerRadius(4)
    }
}

// MARK: - Keyboard Modifier Reader

/// A view that tracks keyboard modifiers
struct KeyboardModifierReader: NSViewRepresentable {
    var onChange: (NSEvent.ModifierFlags) -> Void
    
    func makeNSView(context: Context) -> NSView {
        let view = ModifierTrackingView()
        view.onChange = onChange
        return view
    }
    
    func updateNSView(_ nsView: NSView, context: Context) {
        if let view = nsView as? ModifierTrackingView {
            view.onChange = onChange
        }
    }
    
    class ModifierTrackingView: NSView {
        var onChange: ((NSEvent.ModifierFlags) -> Void)?
        private var monitor: Any?
        
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            
            if window != nil && monitor == nil {
                monitor = NSEvent.addLocalMonitorForEvents(matching: [.flagsChanged]) { [weak self] event in
                    self?.onChange?(event.modifierFlags)
                    return event
                }
            }
        }
        
        override func removeFromSuperview() {
            if let monitor = monitor {
                NSEvent.removeMonitor(monitor)
                self.monitor = nil
            }
            super.removeFromSuperview()
        }
    }
}

// MARK: - Preview

#Preview {
    let model = EmbeddingGraphModel()
    return ClusterPlotView(model: model)
        .frame(width: 500, height: 600)
}
