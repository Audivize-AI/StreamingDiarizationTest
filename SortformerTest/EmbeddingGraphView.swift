//
//  EmbeddingGraphView.swift
//  SortformerTest
//
//  Interactive embedding graph visualization with pan, zoom, and selection.
//

import SwiftUI
import simd

/// Edge visibility mode
public enum EdgeVisibilityMode: String, CaseIterable {
    case off = "Off"
    case selectedOnly = "Selected"
    case all = "All"
}

/// Interactive graph view for embedding visualization
public struct EmbeddingGraphView: View {
    @ObservedObject var model: EmbeddingGraphModel
    
    // MARK: - State
    
    @State private var scale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    @State private var dragOffset: CGSize = .zero
    @State private var edgeVisibility: EdgeVisibilityMode = .selectedOnly
    @State private var hoveredEdge: (edge: GraphEdge, position: CGPoint)? = nil
    
    // Node size
    private let baseNodeRadius: CGFloat = 6
    private let selectedNodeRadius: CGFloat = 10
    
    // Speaker colors (consistent with heatmap view)
    private let speakerColors: [Color] = [
        .red, .green, .blue, .orange
    ]
    
    public init(model: EmbeddingGraphModel) {
        self.model = model
    }
    
    public var body: some View {
        VStack(spacing: 8) {
            // Header with title and stats
            HStack {
                Text("Embedding Graph")
                    .font(.headline)
                
                Spacer()
                
                Text("\(model.nodes.count) embeddings")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 8)
            
            // Main graph canvas
            GeometryReader { geometry in
                let canvasSize = geometry.size
                
                TimelineView(.animation) { timeline in
                    ZStack {
                        // Background
                        Color(white: 0.08)
                        
                        // Canvas for edges and nodes
                        Canvas { context, size in
                            drawGraph(context: context, size: size)
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
                        if let hoveredIdx = model.hoveredNodeIndex, hoveredIdx < model.nodes.count {
                            let node = model.nodes[hoveredIdx]
                            tooltipView(for: node)
                                .position(nodeScreenPosition(node, in: canvasSize))
                                .offset(y: -40)
                        }
                        
                        // Tooltip overlay for edges
                        if let edgeInfo = hoveredEdge {
                            edgeTooltipView(for: edgeInfo.edge)
                                .position(edgeInfo.position)
                                .offset(y: -25)
                        }
                    }
                    .onChange(of: timeline.date) { _, _ in
                        // Step simulation on each animation frame (outside render pass)
                        model.stepSimulation()
                    }
                }
                .clipShape(RoundedRectangle(cornerRadius: 8))
            }
            
            // Controls
            VStack(spacing: 6) {
                // Edge visibility toggle
                HStack {
                    Text("Edges:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Picker("", selection: $edgeVisibility) {
                        ForEach(EdgeVisibilityMode.allCases, id: \.self) { mode in
                            Text(mode.rawValue).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    .frame(maxWidth: 180)
                    
                    Spacer()
                    
                    // Reset view button
                    Button(action: resetView) {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.caption)
                    }
                    .buttonStyle(.plain)
                    .help("Reset view")
                }
                
                // Distance threshold slider
                HStack {
                    Text("Threshold:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Slider(value: Binding(
                        get: { Double(model.config.distanceThreshold) },
                        set: { model.config.distanceThreshold = Float($0) }
                    ), in: 0.1...1.0, step: 0.05)
                    .frame(maxWidth: 120)
                    
                    Text(String(format: "%.2f", model.config.distanceThreshold))
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(width: 35)
                    
                    Spacer()
                    
                    // Force strength slider
                    Text("Force:")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    
                    Slider(value: Binding(
                        get: { Double(model.config.forceStrength) },
                        set: { model.config.forceStrength = Float($0) }
                    ), in: 0.001...0.1, step: 0.005)
                    .frame(maxWidth: 80)
                }
            }
            .padding(.horizontal, 8)
        }
        .padding(8)
        .background(Color(white: 0.05))
        .cornerRadius(12)
    }
    
    // MARK: - Drawing
    
    private func drawGraph(context: GraphicsContext, size: CGSize) {
        let nodes = model.nodes
        guard !nodes.isEmpty else {
            // Draw empty state
            let text = Text("No embeddings yet")
                .font(.caption)
                .foregroundColor(.secondary)
            context.draw(context.resolve(text), at: CGPoint(x: size.width / 2, y: size.height / 2), anchor: .center)
            return
        }
        
        // Draw edges based on visibility mode
        drawEdges(context: context, size: size)
        
        // Draw nodes
        drawNodes(context: context, size: size)
    }
    
    private func drawEdges(context: GraphicsContext, size: CGSize) {
        let edgesToDraw: [GraphEdge]
        
        switch edgeVisibility {
        case .off:
            return
            
        case .selectedOnly:
            if let selectedIdx = model.selectedNodeIndex ?? model.hoveredNodeIndex {
                edgesToDraw = model.edgesForNode(selectedIdx)
                    .filter { $0.distance <= model.config.distanceThreshold }
            } else {
                return
            }
            
        case .all:
            edgesToDraw = model.thresholdedEdges
        }
        
        // Draw edges with opacity and thickness based on cosine similarity
        for edge in edgesToDraw {
            guard edge.u < model.nodes.count && edge.v < model.nodes.count else { continue }
            
            let nodeU = model.nodes[edge.u]
            let nodeV = model.nodes[edge.v]
            
            let posU = nodeScreenPosition(nodeU, in: size)
            let posV = nodeScreenPosition(nodeV, in: size)
            
            // Get cosine similarity (1 - distance) for this edge
            let similarity = model.cosineSimilarity(u: edge.u, v: edge.v)
            
            // Opacity based on similarity (more similar = more opaque)
            let opacity = Double(similarity) * 0.8
            
            // Thickness increases with similarity (range: 0.5 to 4.0)
            let thickness = CGFloat(0.5 + similarity * 3.5)
            
            var path = Path()
            path.move(to: posU)
            path.addLine(to: posV)
            
            // Color based on whether nodes are same speaker
            let edgeColor: Color
            if nodeU.speakerIndex == nodeV.speakerIndex {
                edgeColor = speakerColors[nodeU.speakerIndex % speakerColors.count]
            } else {
                edgeColor = .gray
            }
            
            context.stroke(path, with: .color(edgeColor.opacity(opacity)),
                          style: StrokeStyle(lineWidth: thickness))
        }
    }
    
    private func drawNodes(context: GraphicsContext, size: CGSize) {
        for (idx, node) in model.nodes.enumerated() {
            let pos = nodeScreenPosition(node, in: size)
            
            // Determine node appearance
            let isSelected = model.selectedNodeIndex == idx
            let isHovered = model.hoveredNodeIndex == idx
            let radius = (isSelected || isHovered) ? selectedNodeRadius : baseNodeRadius
            
            let color = speakerColors[node.speakerIndex % speakerColors.count]
            
            // Draw glow for selected/hovered
            if isSelected || isHovered {
                let glowRect = CGRect(
                    x: pos.x - radius * 1.5,
                    y: pos.y - radius * 1.5,
                    width: radius * 3,
                    height: radius * 3
                )
                context.fill(Circle().path(in: glowRect), with: .color(color.opacity(0.3)))
            }
            
            // Draw node circle
            let nodeRect = CGRect(
                x: pos.x - radius,
                y: pos.y - radius,
                width: radius * 2,
                height: radius * 2
            )
            
            context.fill(Circle().path(in: nodeRect), with: .color(color))
            
            // Draw border
            context.stroke(Circle().path(in: nodeRect),
                          with: .color(isSelected ? .white : color.opacity(0.8)),
                          style: StrokeStyle(lineWidth: isSelected ? 2 : 1))
        }
    }
    
    // MARK: - Coordinate Transforms
    
    private func nodeScreenPosition(_ node: GraphNode, in size: CGSize) -> CGPoint {
        let centerX = size.width / 2
        let centerY = size.height / 2
        // Increased base scale (0.4 -> 0.8) and apply config's positionScale
        let graphScale = min(size.width, size.height) * 0.8 * scale * CGFloat(model.config.positionScale)
        
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
        let graphScale = min(size.width, size.height) * 0.8 * scale * CGFloat(model.config.positionScale)
        
        let totalOffset = CGSize(
            width: offset.width + dragOffset.width,
            height: offset.height + dragOffset.height
        )
        
        return SIMD2<Float>(
            Float((point.x - centerX - totalOffset.width) / graphScale),
            Float((point.y - centerY - totalOffset.height) / graphScale)
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
            
            // Find nearest node within a threshold
            var nearestIdx: Int? = nil
            var nearestDist: Float = 0.1  // Threshold in graph coordinates
            
            for (idx, node) in model.nodes.enumerated() {
                let dist = simd_distance(graphPos, node.position)
                if dist < nearestDist {
                    nearestDist = dist
                    nearestIdx = idx
                }
            }
            
            model.hoveredNodeIndex = nearestIdx
            
            // If not hovering over a node, check for edge hover
            if nearestIdx == nil {
                hoveredEdge = findNearestEdge(at: graphPos, in: canvasSize)
            } else {
                hoveredEdge = nil
            }
            
        case .ended:
            model.hoveredNodeIndex = nil
            hoveredEdge = nil
        }
    }
    
    /// Find the nearest edge to a point (returns nil if none within threshold)
    private func findNearestEdge(at graphPos: SIMD2<Float>, in canvasSize: CGSize) -> (edge: GraphEdge, position: CGPoint)? {
        let threshold: Float = 0.05  // Distance threshold in graph coordinates
        
        var nearestEdge: GraphEdge? = nil
        var nearestDist: Float = threshold
        var nearestPoint: SIMD2<Float> = .zero
        
        // Check all visible edges
        let edgesToCheck: [GraphEdge]
        switch edgeVisibility {
        case .off:
            return nil
        case .selectedOnly:
            if let selectedIdx = model.selectedNodeIndex ?? model.hoveredNodeIndex {
                edgesToCheck = model.edgesForNode(selectedIdx)
                    .filter { $0.distance <= model.config.distanceThreshold }
            } else {
                return nil
            }
        case .all:
            edgesToCheck = model.thresholdedEdges
        }
        
        for edge in edgesToCheck {
            guard edge.u < model.nodes.count && edge.v < model.nodes.count else { continue }
            
            let p1 = model.nodes[edge.u].position
            let p2 = model.nodes[edge.v].position
            
            // Find closest point on line segment
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
            // Convert graph position to screen position for tooltip
            let screenPos = graphToScreen(nearestPoint, in: canvasSize)
            return (edge: edge, position: screenPos)
        }
        
        return nil
    }
    
    private func graphToScreen(_ pos: SIMD2<Float>, in size: CGSize) -> CGPoint {
        let centerX = size.width / 2
        let centerY = size.height / 2
        let graphScale = min(size.width, size.height) * 0.8 * scale * CGFloat(model.config.positionScale)
        
        let totalOffset = CGSize(
            width: offset.width + dragOffset.width,
            height: offset.height + dragOffset.height
        )
        
        return CGPoint(
            x: centerX + CGFloat(pos.x) * graphScale + totalOffset.width,
            y: centerY + CGFloat(pos.y) * graphScale + totalOffset.height
        )
    }
    
    private func handleTap(location: CGPoint, canvasSize: CGSize) {
        let graphPos = screenToGraph(location, in: canvasSize)
        
        // Find nearest node
        var nearestIdx: Int? = nil
        var nearestDist: Float = 0.15  // Slightly larger threshold for tap
        
        for (idx, node) in model.nodes.enumerated() {
            let dist = simd_distance(graphPos, node.position)
            if dist < nearestDist {
                nearestDist = dist
                nearestIdx = idx
            }
        }
        
        // Toggle selection using the cross-view API
        if model.selectedNodeIndex == nearestIdx {
            model.selectByNodeIndex(nil)
        } else {
            model.selectByNodeIndex(nearestIdx)
        }
    }
    
    private func resetView() {
        withAnimation(.easeOut(duration: 0.3)) {
            scale = 1.0
            offset = .zero
            dragOffset = .zero
        }
    }
    
    // MARK: - Tooltip
    
    private func tooltipView(for node: GraphNode) -> some View {
        let startTime = String(format: "%.2f", Float(node.startFrame) * 0.08)
        let endTime = String(format: "%.2f", Float(node.endFrame) * 0.08)
        
        return VStack(alignment: .leading, spacing: 2) {
            Text("Speaker \(node.speakerIndex)")
                .font(.caption.bold())
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

// MARK: - Preview

#Preview {
    let model = EmbeddingGraphModel()
    return EmbeddingGraphView(model: model)
        .frame(width: 400, height: 500)
}
