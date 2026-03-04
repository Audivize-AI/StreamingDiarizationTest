import SwiftUI
import CoreGraphics
#if os(macOS)
import AppKit
#endif

enum SpeakerTimelinePalette {
    static let slotColors: [Color] = [.red, .green, .blue, .orange]

    static func slotColor(for slot: Int) -> Color {
        guard slot >= 0 else { return .gray }
        return slotColors[((slot % slotColors.count) + slotColors.count) % slotColors.count]
    }
}

/// Real-time speech diarization heatmap view.
/// Displays speaker probabilities as a viridis-colored heatmap with 4 speaker rows.
struct SpeechPlotView: View {
    /// Timeline with diarization history
    let timeline: SortformerTimeline?
    
    /// Speaker cache predictions (optional, drawn as zeros if nil)
    let spkcachePreds: [Float]?
    
    /// FIFO queue predictions (optional)
    let fifoPreds: [Float]?
    
    /// Right context frames for FIFO alignment
    let chunkRightContext: Int
    
    /// Left context frames for FIFO alignment
    let chunkLeftContext: Int
    
    /// Whether recording is active (for auto-scroll behavior)
    let isRecording: Bool
    
    /// Update trigger to force redraws
    let updateTrigger: Int
    
    /// Segment annotations dictionary
    let segmentAnnotations: [String: String]

    /// Distance from Sortformer segment centroid to nearest cluster centroid.
    let segmentCentroidDistances: [UInt64: Float]

    /// Distance from embedding segment centroid to nearest cluster centroid.
    let embeddingSegmentCentroidDistances: [UUID: Float]
    
    
    /// Callback when a segment is clicked (start, end)
    let onPlaySegment: ((Float, Float) -> Void)?
    
    /// Callback when a segment is clicked for annotation
    let onAnnotateSegment: ((SpeakerSegment) -> Void)?
    
    /// Callback when purge button is tapped for a speaker
    let onPurgeSpeaker: ((Int) -> Void)?
    
    /// Visible frames in the viewport (determines cell width relative to window)
    private let visibleFrames = max(
        globalConfig.fifoLen + globalConfig.chunkLen + globalConfig.chunkRightContext, globalConfig.spkcacheLen)
    
    /// Speaker cache size
    private let spkcacheSize = globalConfig.spkcacheLen
    
    /// Number of speakers (fixed at 4 for Sortformer)
    private let numSpeakers = globalConfig.numSpeakers

    private let yAxisWidth: CGFloat = 62
    private let sectionSpacing: CGFloat = 4
    
    /// Chunk size for virtualization (frames per chunk)
    private let chunkSize = 500
    
    /// Track if we're at the trailing edge (following live updates)
    @State private var isFollowingLive = true
    
    /// Current scroll offset for virtualization
    @State private var scrollOffset: CGFloat = 0
    
    /// Hovered segment info for tooltip
    @State private var hoveredSegment: SpeakerSegment?
    @State private var hoveredEmbeddingSegment: EmbeddingSegment?
    @State private var hoverLocation: CGPoint = .zero
    
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
    
    var body: some View {
        GeometryReader { geometry in
            let availableHeight = geometry.size.height - 90  // Reserved for title/labels/padding
            let plotHeight = availableHeight / 3.0  // Equal height for all 4 plots
            let timelineInset = yAxisWidth + sectionSpacing
            let plotWidth = max(0, geometry.size.width - timelineInset)
            
            VStack(alignment: .leading, spacing: 4) {
                // Title with stats
                Text(titleText)
                    .font(.headline)
                    .padding(.horizontal, 8)
                
                // Main diarization heatmap with labels
                mainHeatmapSection(width: plotWidth, height: plotHeight)
                
                // X-axis label
                Text("Time (frames, 80ms each) - \(totalFrameCount) total frames")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.leading, timelineInset)
                
                // FIFO queue visualization (aligned with main timeline)
                fifoSection(width: plotWidth, height: plotHeight)
                
                // Speaker cache visualization (always shown)
                spkcacheSection(width: plotWidth, height: plotHeight)
            }
        }
        .padding(12)  // Match spacing with title text
        .background(Color(white: 0.05))
        .cornerRadius(12)
    }
    
    // MARK: - Main Heatmap Section
    
    private func mainHeatmapSection(width: CGFloat, height: CGFloat) -> some View {
        let contentHeight = max(0, height)
        let scrollViewportHeight = contentHeight + horizontalScrollbarHeight

        return HStack(alignment: .top, spacing: 4) {
            // Y-axis labels with purge buttons
            VStack(spacing: 0) {
                ForEach(0..<numSpeakers, id: \.self) { speaker in
                    HStack(spacing: 2) {
                        Button(action: {
                            onPurgeSpeaker?(speaker)
                        }) {
                            Image(systemName: "xmark.circle.fill")
                                .font(.system(size: 10))
                                .foregroundColor(speakerColor(forSlot: speaker).opacity(0.7))
                        }
                        .buttonStyle(.plain)
                        .help("Purge speaker \(speaker)")

                        Text("Spk \(speaker)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .frame(maxHeight: .infinity)
                }
            }
            .frame(width: yAxisWidth, height: contentHeight)
            
            // Scrollable heatmap - virtualized chunks
            ScrollViewReader { scrollProxy in
                ScrollView(.horizontal, showsIndicators: true) {
                    let contentWidth = calculatedContentWidth(viewportWidth: width)
                    let cellWidth = contentWidth / CGFloat(max(totalFrameCount, visibleFrames))
                    let totalChunks = max(1, (totalFrameCount + chunkSize - 1) / chunkSize)
                    
                    HStack(spacing: 0) {
                        // Render chunks - LazyHStack would be ideal but doesn't work well with ScrollViewReader
                        ForEach(0..<totalChunks, id: \.self) { chunkIndex in
                            let startFrame = chunkIndex * chunkSize
                            let endFrame = min(startFrame + chunkSize, max(totalFrameCount, visibleFrames))
                            let chunkWidth = CGFloat(endFrame - startFrame) * cellWidth
                            
                            Canvas { context, size in
                                drawChunk(context: context, size: size, 
                                         startFrame: startFrame, endFrame: endFrame,
                                         cellWidth: cellWidth, cellHeight: contentHeight / CGFloat(numSpeakers))
                            }
                            .drawingGroup()
                            .frame(width: chunkWidth, height: contentHeight)
                            .id("chunk-\(chunkIndex)")
                        }
                        
                        // Draw segment overlays on top (single canvas for segments)
                        // Moved to overlay for proper layering
                        
                        // Stable scroll target at the trailing edge
                        Color.clear
                            .frame(width: 1, height: max(1, contentHeight))
                            .id("scrollEnd")
                    }
                    .overlay {
                        // Segment outlines layer
                        Canvas { context, size in
                            drawSegmentOverlays(context: context, size: size, viewportWidth: width)
                        }
                        .allowsHitTesting(false)
                    }
                    .onContinuousHover { phase in
                        switch phase {
                        case .active(let location):
                            hoverLocation = location
                            guard let tl = timeline else {
                                hoveredSegment = nil
                                hoveredEmbeddingSegment = nil
                                return
                            }

                        guard cellWidth > 0 else {
                            hoveredSegment = nil
                            hoveredEmbeddingSegment = nil
                            return
                        }

                        let cellHeight = contentHeight / CGFloat(numSpeakers)
                        guard cellHeight > 0 else {
                            hoveredSegment = nil
                            hoveredEmbeddingSegment = nil
                            return
                            }

                            let totalFrames = max(totalFrameCount, visibleFrames)
                            let hoverFrame = Int(location.x / cellWidth)
                            let hoverSpeaker = Int(location.y / cellHeight)

                            guard hoverFrame >= 0, hoverFrame < totalFrames,
                                  hoverSpeaker >= 0, hoverSpeaker < numSpeakers else {
                                hoveredSegment = nil
                                hoveredEmbeddingSegment = nil
                                return
                            }

                            let resolvedSegment = sortformerSegmentAt(
                                frame: hoverFrame,
                                speakerIndex: hoverSpeaker,
                                timeline: tl
                            )
                            let resolvedEmbedding = resolvedSegment == nil
                                ? embeddingSegmentAt(
                                    frame: hoverFrame,
                                    preferredSpeaker: hoverSpeaker,
                                    timeline: tl
                                )
                                : nil

                            if hoveredSegment?.id != resolvedSegment?.id {
                                hoveredSegment = resolvedSegment
                            }
                            if hoveredEmbeddingSegment?.id != resolvedEmbedding?.id {
                                hoveredEmbeddingSegment = resolvedEmbedding
                            }
                        case .ended:
                            if hoveredSegment != nil {
                                hoveredSegment = nil
                            }
                            if hoveredEmbeddingSegment != nil {
                                hoveredEmbeddingSegment = nil
                            }
                        }
                    }
                    .overlay(alignment: .topLeading) {
                        // Tooltip for hovered segment
                        if let segment = hoveredSegment {
                            let startStr = String(format: "%.2f", segment.startTime)
                            let endStr = String(format: "%.2f", segment.endTime)
                            let durationStr = String(format: "%.2f", segment.endTime - segment.startTime)
                            let idStr = String(segment.id, radix: 16).uppercased()
                            let distanceStr = segmentCentroidDistances[segment.id].map { String(format: "%.3f", $0) }
                            
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Spk \(segment.slot): \(startStr)s - \(endStr)s (\(durationStr)s)")
                                if let distanceStr {
                                    Text("centroid d: \(distanceStr)")
                                        .font(.caption2)
                                        .foregroundColor(.gray)
                                }
                                Text("ID: \(idStr)")
                                    .font(.caption2)
                                    .foregroundColor(.gray)
                            }
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.black.opacity(0.8))
                            .foregroundColor(.white)
                            .cornerRadius(4)
                            .offset(x: hoverLocation.x + 10, y: hoverLocation.y - 30)
                        } else if let embSegment = hoveredEmbeddingSegment {
                            let startStr = String(format: "%.2f", Float(embSegment.startFrame) * 0.08)
                            let endStr = String(format: "%.2f", Float(embSegment.endFrame) * 0.08)
                            let embeddingDistanceStr = embeddingSegmentCentroidDistances[embSegment.id]
                                .map { String(format: "%.3f", $0) }
                            
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Spk \(embSegment.speakerId): \(startStr)s - \(endStr)s")
                                if let embeddingDistanceStr {
                                    Text("cluster d: \(embeddingDistanceStr)")
                                        .font(.caption2)
                                        .foregroundColor(.gray)
                                }
                            }
                            .font(.caption)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Color.black.opacity(0.8))
                            .foregroundColor(.white)
                            .cornerRadius(4)
                            .offset(x: hoverLocation.x + 10, y: hoverLocation.y - 30)
                        }
                    }
                    .onTapGesture(count: 2) { location in
                        // DOUBLE CLICK = ANNOTATE
                        guard !isRecording, let tl = timeline else { return }
                        
                        let cellHeight = contentHeight / CGFloat(numSpeakers)
                        guard cellHeight > 0, cellWidth > 0 else { return }

                        let totalFrames = max(totalFrameCount, visibleFrames)
                        let clickedFrame = Int(location.x / cellWidth)
                        let clickedSpeaker = Int(location.y / cellHeight)

                        guard clickedFrame >= 0, clickedFrame < totalFrames,
                              clickedSpeaker >= 0, clickedSpeaker < numSpeakers else { return }
                        
                        if let segment = sortformerSegmentAt(
                            frame: clickedFrame,
                            speakerIndex: clickedSpeaker,
                            timeline: tl
                        ) {
                            onAnnotateSegment?(segment)
                        }
                    }
                    .onTapGesture(count: 1) { location in
                        // SINGLE CLICK - check for embedding stroke first, then segment play
                        guard !isRecording, let tl = timeline else { return }
                        
                        let cellHeight = contentHeight / CGFloat(numSpeakers)
                        guard cellHeight > 0, cellWidth > 0 else { return }

                        let totalFrames = max(totalFrameCount, visibleFrames)
                        let clickedFrame = Int(location.x / cellWidth)
                        let clickedSpeaker = Int(location.y / cellHeight)

                        guard clickedFrame >= 0, clickedFrame < totalFrames,
                              clickedSpeaker >= 0, clickedSpeaker < numSpeakers else { return }
                        
                        // Otherwise, play the segment
                        if let segment = sortformerSegmentAt(
                            frame: clickedFrame,
                            speakerIndex: clickedSpeaker,
                            timeline: tl
                        ) {
                            onPlaySegment?(segment.startTime, segment.endTime)
                        }
                    }
                    .background(
                        // Track scroll position
                        GeometryReader { contentGeometry in
                            Color.clear.preference(
                                key: ScrollOffsetPreferenceKey.self,
                                value: contentGeometry.frame(in: .named("scrollView")).minX
                            )
                        }
                    )
                }
                .coordinateSpace(name: "scrollView")
                .onPreferenceChange(ScrollOffsetPreferenceKey.self) { offset in
                    scrollOffset = offset
                    
                    // Only detect when user scrolls AWAY from the end
                    // Don't auto-enable following - user must click button
                    if isRecording && isFollowingLive {
                        let contentWidth = calculatedContentWidth(viewportWidth: width)
                        let maxOffset = -(contentWidth - width)
                        
                        // User scrolled away if not within 50 points of end
                        let isAtEnd = offset <= maxOffset + 50
                        if !isAtEnd {
                            isFollowingLive = false
                        }
                    }
                }
                .frame(height: scrollViewportHeight)
                .background(Color(cgColor: ViridisColormap.cgColor(for: 0)))  // Purple background
                .cornerRadius(8)
                .onChange(of: updateTrigger) { _, _ in
                    // Only auto-scroll if following live
                    if isRecording && isFollowingLive {
                        scrollProxy.scrollTo("scrollEnd", anchor: .trailing)
                    }
                }
                .onAppear {
                    scrollProxy.scrollTo("scrollEnd", anchor: .trailing)
                    isFollowingLive = true
                }
                .overlay(alignment: .bottomTrailing) {
                    // Show "Follow Live" button when scrolled away during recording
                    if isRecording && !isFollowingLive {
                        Button(action: {
                            isFollowingLive = true
                            scrollProxy.scrollTo("scrollEnd", anchor: .trailing)
                        }) {
                            HStack(spacing: 4) {
                                Image(systemName: "arrow.right.to.line")
                                Text("Follow Live")
                            }
                            .font(.caption)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(6)
                        }
                        .buttonStyle(.plain)
                        .padding(8)
                    }
                }
            }
        }
    }
    
    // MARK: - FIFO Queue Section
    
    private func fifoSection(width: CGFloat, height: CGFloat) -> some View {
        let fifoLength = (fifoPreds?.count ?? 0) / numSpeakers
        
        return VStack(alignment: .leading, spacing: 4) {
            Text("FIFO Queue (\(fifoLength) frames)")
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.leading, 4)
            
            HStack(alignment: .top, spacing: 4) {
                // Y-axis labels
                VStack(spacing: 0) {
                    ForEach(0..<numSpeakers, id: \.self) { speaker in
                        Text("Spk \(speaker)")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .frame(maxHeight: .infinity)
                    }
                }
                .frame(width: yAxisWidth, height: max(height, 0))
                
                // FIFO heatmap - fixed width, right-aligned content
                Canvas { context, size in
                    drawFifoHeatmap(context: context, size: size)
                }
                .frame(width: width, height: max(height, 0))
                .background(Color(cgColor: ViridisColormap.cgColor(for: 0)))  // Purple background
                .cornerRadius(6)
            }
        }
    }
    
    // MARK: - Speaker Cache Section
    
    private func spkcacheSection(width: CGFloat, height: CGFloat) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Speaker Cache (\(spkcacheSize) frames)")
                .font(.caption)
                .foregroundColor(.secondary)
                .padding(.leading, 4)
            
            HStack(alignment: .top, spacing: 4) {
                // Y-axis labels
                VStack(spacing: 0) {
                    ForEach(0..<numSpeakers, id: \.self) { speaker in
                        Text("Spk \(speaker)")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .frame(maxHeight: .infinity)
                    }
                }
                .frame(width: yAxisWidth, height: max(0, height))
                
                // Speaker cache heatmap
                Canvas { context, size in
                    drawSpkcacheHeatmap(context: context, size: size)
                }
                .id("spkcache-\(updateTrigger)")  // Force redraw on updates
                .frame(width: width, height: max(0, height))
                .background(Color(cgColor: ViridisColormap.cgColor(for: 0)))  // Purple background
                .cornerRadius(6)
            }
        }
    }
    
    // MARK: - Computed Properties
    
    private var titleText: String {
        guard let tl = timeline else {
            return "Live Diarization - No data"
        }
        let totalFrames = tl.cursorFrame + tl.numTentative
        let elapsedSeconds = Float(totalFrames) * 0.08
        return String(format: "Live Diarization - Confirmed: %d | Tentative: %d | Segments: %d + %d | (%.1fs)",
                      tl.cursorFrame, tl.numTentative,
                      tl.finalizedSegments.flatMap(\.self).count, tl.tentativeSegments.flatMap(\.self).count,
                      elapsedSeconds)
    }
    
    private var totalFrameCount: Int {
        guard let tl = timeline else { return 0 }
        return tl.cursorFrame + tl.numTentative
    }

    @inline(__always)
    private func speakerColor(forSlot slot: Int) -> Color {
        SpeakerTimelinePalette.slotColor(for: slot)
    }
    
    private func calculatedContentWidth(viewportWidth: CGFloat) -> CGFloat {
        guard let tl = timeline else {
            return viewportWidth
        }
        let totalFrames = tl.cursorFrame + tl.numTentative
        let cellWidth = viewportWidth / CGFloat(visibleFrames)
        return max(CGFloat(totalFrames) * cellWidth, viewportWidth)
    }
    
    // MARK: - Chunk-Based Drawing Functions
    
    /// Draw a single chunk of the heatmap
    private func drawChunk(context: GraphicsContext, size: CGSize,
                          startFrame: Int, endFrame: Int,
                          cellWidth: CGFloat, cellHeight: CGFloat) {
        guard let tl = timeline else {
            // Draw purple background for empty chunk
            context.fill(Path(CGRect(origin: .zero, size: size)), 
                        with: .color(Color(cgColor: ViridisColormap.cgColor(for: 0))))
            return
        }
        
        let numConfirmed = tl.cursorFrame
        let numTentative = tl.numTentative
        
        // Draw probability cells for this chunk
        for frameIdx in startFrame..<endFrame {
            let localX = CGFloat(frameIdx - startFrame) * cellWidth
            
            for speaker in 0..<numSpeakers {
                let y = CGFloat(speaker) * cellHeight
                let rect = CGRect(x: localX, y: y, width: cellWidth + 0.5, height: cellHeight)
                
                let prob: Float
                if frameIdx < numConfirmed {
                    prob = tl.probability(speaker: speaker, frame: frameIdx)
                } else if frameIdx < numConfirmed + numTentative {
                    let tentativeFrame = frameIdx - numConfirmed
                    prob = tl.tentativeProbability(speaker: speaker, frame: tentativeFrame)
                } else {
                    prob = 0 // Padding
                }
                
                let color = Color(cgColor: ViridisColormap.cgColor(for: prob))
                context.fill(Path(rect), with: .color(color))
            }
        }
        
        // Draw tentative boundary line if it's in this chunk
        if numTentative > 0 && numConfirmed >= startFrame && numConfirmed < endFrame {
            let localX = CGFloat(numConfirmed - startFrame) * cellWidth
            var path = Path()
            path.move(to: CGPoint(x: localX, y: 0))
            path.addLine(to: CGPoint(x: localX, y: size.height))
            context.stroke(path, with: .color(.white.opacity(0.8)),
                          style: StrokeStyle(lineWidth: 2, dash: [5, 3]))
        }
    }
    
    /// Draw segment outlines as an overlay
    private func drawSegmentOverlays(context: GraphicsContext, size: CGSize, viewportWidth: CGFloat) {
        guard let tl = timeline else { return }
        
        let totalFrames = max(tl.cursorFrame + tl.numTentative, visibleFrames)
        let cellWidth = size.width / CGFloat(totalFrames)
        let cellHeight = size.height / CGFloat(numSpeakers)
        var identityBySegmentID: [UInt64: Int] = [:]
        for profile in tl.activeSpeakers.values {
            for segment in profile.finalizedSegments {
                identityBySegmentID[segment.id] = profile.speakerId
            }
            for segment in profile.tentativeSegments {
                identityBySegmentID[segment.id] = profile.speakerId
            }
        }
        for profile in tl.inactiveSpeakers {
            for segment in profile.finalizedSegments {
                identityBySegmentID[segment.id] = profile.speakerId
            }
        }
        
        // Draw row-scoped embedding segment bands (avoid full-height overlays).
        for segment in tl.embeddingSegments {
            drawEmbeddingSegmentBand(
                context: context,
                segment: segment,
                cellWidth: cellWidth,
                cellHeight: cellHeight,
                tentative: false,
                distance: embeddingSegmentCentroidDistances[segment.id]
            )
        }
        
        for segment in tl.tentativeEmbeddingSegments {
            drawEmbeddingSegmentBand(
                context: context,
                segment: segment,
                cellWidth: cellWidth,
                cellHeight: cellHeight,
                tentative: true,
                distance: embeddingSegmentCentroidDistances[segment.id]
            )
        }
        
        // Draw finalized segments (solid) with labels
        for segment in tl.finalizedSegments.flatMap({ $0 }) {
            drawSegmentOutline(context: context, segment: segment, 
                             cellWidth: cellWidth, cellHeight: cellHeight, tentative: false)
            drawSegmentLabel(context: context, segment: segment,
                           cellWidth: cellWidth, cellHeight: cellHeight,
                           identitySpeakerID: identityBySegmentID[segment.id])
        }
        
        // Draw tentative segments (dashed) - no labels for tentative
        for segment in tl.tentativeSegments.flatMap({ $0 }) {
            drawSegmentOutline(context: context, segment: segment, 
                             cellWidth: cellWidth, cellHeight: cellHeight, tentative: true)
        }
        
        // Draw embedding strokes within each speaker row
        drawEmbeddingStrokes(context: context, cellWidth: cellWidth, cellHeight: cellHeight)
    }
    
    /// Draw embedding region strokes within each speaker row
    /// Strokes are staggered if they overlap in time
    private func drawEmbeddingStrokes(context: GraphicsContext, cellWidth: CGFloat, cellHeight: CGFloat) {
        guard let tl = timeline else { return }
        guard cellHeight > 0 else { return }

        let strokeLayout = embeddingStrokeLayout(for: cellHeight)
        let strokeHeight = strokeLayout.strokeHeight
        let strokeMargin = strokeLayout.strokeMargin
        let maxStaggerLevels = strokeLayout.maxStaggerLevels
        
        // Collect all embeddings with their info (including tentative flag)
        var allEmbeddings: [(emb: SpeakerEmbedding, speakerIndex: Int, isTentative: Bool)] = []
        
        // Finalized embeddings
        for segment in tl.embeddingSegments {
            for emb in segment.embeddings {
                allEmbeddings.append((emb, segment.speakerId, false))
            }
        }
        
        // Tentative embeddings
        for segment in tl.tentativeEmbeddingSegments {
            for emb in segment.embeddings {
                allEmbeddings.append((emb, segment.speakerId, true))
            }
        }
        
        guard !allEmbeddings.isEmpty else { return }
        
        // Group by speaker
        var embeddingsBySpeaker: [[Int]] = Array(repeating: [], count: numSpeakers)
        for (idx, (_, speakerIndex, _)) in allEmbeddings.enumerated() {
            if speakerIndex >= 0 && speakerIndex < numSpeakers {
                embeddingsBySpeaker[speakerIndex].append(idx)
            }
        }
        
        // Draw for each speaker
        for speakerIndex in 0..<numSpeakers {
            let indices = embeddingsBySpeaker[speakerIndex]
            guard !indices.isEmpty else { continue }
            
            // Sort by start frame
            let sortedIndices = indices.sorted { allEmbeddings[$0].emb.startFrame < allEmbeddings[$1].emb.startFrame }
            
            // Assign stagger levels to avoid overlap
            var staggerLevels: [Int: Int] = [:]
            var levelEndFrames: [Int] = Array(repeating: -1, count: maxStaggerLevels)
            
            for idx in sortedIndices {
                let emb = allEmbeddings[idx].emb
                // Find first level that doesn't overlap
                var assignedLevel = 0
                for level in 0..<maxStaggerLevels {
                    if levelEndFrames[level] <= emb.startFrame {
                        assignedLevel = level
                        break
                    }
                    if level == maxStaggerLevels - 1 {
                        assignedLevel = level // Force last level if all overlap
                    }
                }
                staggerLevels[idx] = assignedLevel
                levelEndFrames[assignedLevel] = emb.endFrame
            }
            
            // Draw strokes
            let speakerColor = speakerColor(forSlot: speakerIndex)
            let rowY = CGFloat(speakerIndex) * cellHeight
            let stackHeight = CGFloat(maxStaggerLevels) * strokeHeight + CGFloat(max(maxStaggerLevels - 1, 0)) * strokeMargin
            let baseY = rowY + max(0, (cellHeight - stackHeight) * 0.5)
            
            for idx in sortedIndices {
                let (emb, _, isTentative) = allEmbeddings[idx]
                let level = staggerLevels[idx] ?? 0
                
                let x = CGFloat(emb.startFrame) * cellWidth
                let width = CGFloat(emb.endFrame - emb.startFrame) * cellWidth
                let y = baseY + CGFloat(level) * (strokeHeight + strokeMargin)
                
                let strokeRect = CGRect(x: x, y: y, width: max(width, 4), height: strokeHeight)
                
                // Draw the stroke
                let path = Path(roundedRect: strokeRect, cornerRadius: 2)
                
                if isTentative {
                    // Tentative embeddings: dashed border, lighter fill
                    context.fill(path, with: .color(speakerColor.opacity(0.4)))
                    context.stroke(path, with: .color(speakerColor), style: StrokeStyle(lineWidth: 1.5, dash: [3, 2]))
                } else {
                    // Finalized embeddings: solid
                    context.fill(path, with: .color(speakerColor.opacity(0.8)))
                    context.stroke(path, with: .color(speakerColor.opacity(0.5)), style: StrokeStyle(lineWidth: 1))
                }
            }
        }
    }

    /// Returns the visible frame range in content coordinates, with a small guard band.
    private func currentVisibleFrameRange(totalFrames: Int, cellWidth: CGFloat, viewportWidth: CGFloat) -> Range<Int> {
        guard cellWidth > 0, totalFrames > 0 else { return 0..<0 }
        let visibleMinX = max(0, -scrollOffset)
        let visibleMaxX = min(CGFloat(totalFrames) * cellWidth, visibleMinX + viewportWidth)
        let bufferFrames = 4
        let start = max(0, Int(floor(visibleMinX / cellWidth)) - bufferFrames)
        let end = min(totalFrames, Int(ceil(visibleMaxX / cellWidth)) + bufferFrames)
        return start..<max(start, end)
    }

    @inline(__always)
    private func segmentIntersectsVisibleRange(_ startFrame: Int, _ endFrame: Int, _ visibleFrameRange: Range<Int>) -> Bool {
        endFrame > visibleFrameRange.lowerBound && startFrame < visibleFrameRange.upperBound
    }

    @inline(__always)
    private func sortformerSegmentAt(frame: Int, speakerIndex: Int, timeline: SortformerTimeline) -> SpeakerSegment? {
        guard speakerIndex >= 0,
              speakerIndex < timeline.finalizedSegments.count,
              speakerIndex < timeline.tentativeSegments.count else {
            return nil
        }

        if let segment = segmentContaining(frame: frame, in: timeline.finalizedSegments[speakerIndex]) {
            return segment
        }
        return segmentContaining(frame: frame, in: timeline.tentativeSegments[speakerIndex])
    }

    @inline(__always)
    private func embeddingSegmentAt(frame: Int, preferredSpeaker: Int, timeline: SortformerTimeline) -> EmbeddingSegment? {
        let finalized = segmentContaining(frame: frame, in: timeline.embeddingSegments)
        if let finalized, finalized.speakerId == preferredSpeaker {
            return finalized
        }

        let tentative = segmentContaining(frame: frame, in: timeline.tentativeEmbeddingSegments)
        if let tentative, tentative.speakerId == preferredSpeaker {
            return tentative
        }

        return finalized ?? tentative
    }

    @inline(__always)
    private func segmentContaining(frame: Int, in segments: [SpeakerSegment]) -> SpeakerSegment? {
        guard !segments.isEmpty else { return nil }

        var low = 0
        var high = segments.count
        while low < high {
            let mid = (low + high) / 2
            if segments[mid].startFrame <= frame {
                low = mid + 1
            } else {
                high = mid
            }
        }

        if low > 0 {
            let candidate = segments[low - 1]
            if frame >= candidate.startFrame && frame < candidate.endFrame {
                return candidate
            }
        }

        // Fallback for any out-of-order edge cases.
        return segments.first { frame >= $0.startFrame && frame < $0.endFrame }
    }

    @inline(__always)
    private func segmentContaining(frame: Int, in segments: [EmbeddingSegment]) -> EmbeddingSegment? {
        guard !segments.isEmpty else { return nil }

        var low = 0
        var high = segments.count
        while low < high {
            let mid = (low + high) / 2
            if segments[mid].startFrame <= frame {
                low = mid + 1
            } else {
                high = mid
            }
        }

        if low > 0 {
            let candidate = segments[low - 1]
            if frame >= candidate.startFrame && frame < candidate.endFrame {
                return candidate
            }
        }

        // Fallback for any out-of-order edge cases.
        return segments.first { frame >= $0.startFrame && frame < $0.endFrame }
    }
    
    /// Find embedding at a click location
    /// Returns the embedding UUID if found, nil otherwise
    private func findEmbeddingAtLocation(_ location: CGPoint, cellWidth: CGFloat, cellHeight: CGFloat, height: CGFloat) -> UUID? {
        guard let tl = timeline else { return nil }
        guard cellHeight > 0 else { return nil }
        
        let strokeLayout = embeddingStrokeLayout(for: cellHeight)
        let strokeHeight = strokeLayout.strokeHeight
        let strokeMargin = strokeLayout.strokeMargin
        let maxStaggerLevels = strokeLayout.maxStaggerLevels
        
        // Collect all embeddings with their info (including tentative)
        var allEmbeddings: [(emb: SpeakerEmbedding, speakerIndex: Int)] = []
        
        // Finalized embeddings
        for segment in tl.embeddingSegments {
            for emb in segment.embeddings {
                allEmbeddings.append((emb, segment.speakerId))
            }
        }
        
        // Tentative embeddings
        for segment in tl.tentativeEmbeddingSegments {
            for emb in segment.embeddings {
                allEmbeddings.append((emb, segment.speakerId))
            }
        }
        
        guard !allEmbeddings.isEmpty else { return nil }
        
        // Group by speaker
        var embeddingsBySpeaker: [[Int]] = Array(repeating: [], count: numSpeakers)
        for (idx, (_, speakerIndex)) in allEmbeddings.enumerated() {
            if speakerIndex >= 0 && speakerIndex < numSpeakers {
                embeddingsBySpeaker[speakerIndex].append(idx)
            }
        }
        
        // Check each speaker's embeddings
        for speakerIndex in 0..<numSpeakers {
            let indices = embeddingsBySpeaker[speakerIndex]
            guard !indices.isEmpty else { continue }
            
            // Sort by start frame and compute stagger levels (same logic as draw)
            let sortedIndices = indices.sorted { allEmbeddings[$0].emb.startFrame < allEmbeddings[$1].emb.startFrame }
            
            var staggerLevels: [Int: Int] = [:]
            var levelEndFrames: [Int] = Array(repeating: -1, count: maxStaggerLevels)
            
            for idx in sortedIndices {
                let emb = allEmbeddings[idx].emb
                var assignedLevel = 0
                for level in 0..<maxStaggerLevels {
                    if levelEndFrames[level] <= emb.startFrame {
                        assignedLevel = level
                        break
                    }
                    if level == maxStaggerLevels - 1 {
                        assignedLevel = level
                    }
                }
                staggerLevels[idx] = assignedLevel
                levelEndFrames[assignedLevel] = emb.endFrame
            }
            
            let rowY = CGFloat(speakerIndex) * cellHeight
            let stackHeight = CGFloat(maxStaggerLevels) * strokeHeight + CGFloat(max(maxStaggerLevels - 1, 0)) * strokeMargin
            let baseY = rowY + max(0, (cellHeight - stackHeight) * 0.5)
            
            // Check each embedding's hit rect
            for idx in sortedIndices {
                let (emb, _) = allEmbeddings[idx]
                let level = staggerLevels[idx] ?? 0
                
                let x = CGFloat(emb.startFrame) * cellWidth
                let width = CGFloat(emb.endFrame - emb.startFrame) * cellWidth
                let y = baseY + CGFloat(level) * (strokeHeight + strokeMargin)
                
                let hitRect = CGRect(x: x - 2, y: y - 2, width: max(width, 4) + 4, height: strokeHeight + 4)
                
                if hitRect.contains(location) {
                    return emb.id
                }
            }
        }
        
        return nil
    }

    private func embeddingStrokeLayout(for cellHeight: CGFloat) -> (strokeHeight: CGFloat, strokeMargin: CGFloat, maxStaggerLevels: Int) {
        // Keep embedding strokes fully inside each speaker row, even at short plot heights.
        let strokeHeight = max(3, min(6, floor(cellHeight * 0.22)))
        let strokeMargin = max(1, min(3, floor(cellHeight * 0.08)))
        let slotHeight = strokeHeight + strokeMargin
        let maxLevelsByHeight = max(1, Int(floor((cellHeight + strokeMargin) / max(slotHeight, 1))))
        let maxStaggerLevels = min(3, maxLevelsByHeight)
        return (strokeHeight, strokeMargin, maxStaggerLevels)
    }
    
    /// Draw label on a segment
    private func drawSegmentLabel(context: GraphicsContext, segment: SpeakerSegment,
                                  cellWidth: CGFloat, cellHeight: CGFloat,
                                  identitySpeakerID: Int?) {
        let x = CGFloat(segment.startFrame) * cellWidth
        let y = CGFloat(segment.slot) * cellHeight
        let segmentWidth = CGFloat(segment.endFrame - segment.startFrame) * cellWidth
        
        // Prefer annotation text but always include the resolved speaker ID label.
        let key = DiarizerViewModel.segmentKey(segment)
        let identityLabel = identitySpeakerID.map { "ID \($0)" } ?? "Spk \(segment.slot)"
        let distanceLabel = segmentCentroidDistances[segment.id].map { String(format: "d%.3f", $0) }
        let label: String
        if let annotation = segmentAnnotations[key], !annotation.isEmpty {
            if let distanceLabel {
                label = "\(annotation) • \(identityLabel) • \(distanceLabel)"
            } else {
                label = "\(annotation) • \(identityLabel)"
            }
        } else {
            if let distanceLabel {
                label = "\(identityLabel) • \(distanceLabel)"
            } else {
                label = identityLabel
            }
        }
        
        // Only draw if segment is wide enough
        guard segmentWidth > 20 else { return }
        
        // Check if this is a custom annotation (not just a number)
        let isCustomAnnotation = segmentAnnotations[key] != nil
        
        // Create styled text
        let textStyle = Text(label)
            .font(.system(size: isCustomAnnotation ? 11 : 10, weight: .bold))
            .foregroundColor(.white)
        
        let resolved = context.resolve(textStyle)
        let textSize = resolved.measure(in: CGSize(width: segmentWidth - 6, height: cellHeight))
        
        // Draw background pill for better visibility
        let bgRect = CGRect(
            x: x + 3,
            y: y + (cellHeight - textSize.height) / 2 - 2,
            width: min(textSize.width + 8, segmentWidth - 6),
            height: textSize.height + 4
        )
        
        // Use different colors for custom vs default labels
        let bgColor = isCustomAnnotation ? Color.blue.opacity(0.85) : Color.black.opacity(0.6)
        context.fill(Path(roundedRect: bgRect, cornerRadius: 3), with: .color(bgColor))
        
        // Draw text centered in background
        let textPoint = CGPoint(x: bgRect.midX, y: bgRect.midY)
        context.draw(resolved, at: textPoint, anchor: .center)
    }
    
    // MARK: - Legacy Drawing Functions (kept for reference)
    
    private func drawMainHeatmap(context: GraphicsContext, size: CGSize, viewportWidth: CGFloat) {
        guard let tl = timeline else {
            // Draw empty placeholder with purple background
            context.fill(Path(CGRect(origin: .zero, size: size)), 
                        with: .color(Color(cgColor: ViridisColormap.cgColor(for: 0))))
            return
        }
        
        let numConfirmed = tl.cursorFrame
        let numTentative = tl.numTentative
        let totalFrames = numConfirmed + numTentative
        let framesToDraw = max(totalFrames, visibleFrames)
        
        let cellWidth = size.width / CGFloat(framesToDraw)
        let cellHeight = size.height / CGFloat(numSpeakers)
        
        // Always use detailed drawing - Canvas with drawingGroup() is efficient enough
        drawHeatmapDetailed(context: context, timeline: tl,
                           numConfirmed: numConfirmed, numTentative: numTentative,
                           framesToDraw: framesToDraw, cellWidth: cellWidth, cellHeight: cellHeight)
        
        // Draw tentative boundary line
        if numTentative > 0 && numConfirmed > 0 {
            let x = CGFloat(numConfirmed) * cellWidth
            var path = Path()
            path.move(to: CGPoint(x: x, y: 0))
            path.addLine(to: CGPoint(x: x, y: size.height))
            context.stroke(path, with: .color(.white.opacity(0.8)),
                          style: StrokeStyle(lineWidth: 2, dash: [5, 3]))
        }
        
        // Draw segment outlines - finalized segments (solid)
        let segments = tl.finalizedSegments.flatMap { $0 }
        let tentativeSegments = tl.tentativeSegments.flatMap { $0 }
        for segment in segments {
            drawSegmentOutline(context: context, segment: segment, cellWidth: cellWidth,
                             cellHeight: cellHeight, tentative: false)
        }
        
        // Draw segment outlines - tentative segments (dashed)
        for segment in tentativeSegments {
            drawSegmentOutline(context: context, segment: segment, cellWidth: cellWidth,
                             cellHeight: cellHeight, tentative: true)
        }
    }
    
    /// Detailed drawing for timelines - individual rectangles with viewport culling
    private func drawHeatmapDetailed(context: GraphicsContext, timeline tl: SortformerTimeline,
                                     numConfirmed: Int, numTentative: Int,
                                     framesToDraw: Int, cellWidth: CGFloat, cellHeight: CGFloat) {
        // Estimate visible frames based on viewport (visibleFrames constant)
        // We draw all frames since Canvas needs the full content, but drawingGroup()
        // efficiently handles GPU clipping. For very long timelines (>10000 frames),
        // consider chunking or virtualization.
        
        for frameIdx in 0..<framesToDraw {
            let x = CGFloat(frameIdx) * cellWidth
            
            for speaker in 0..<numSpeakers {
                let y = CGFloat(speaker) * cellHeight
                let rect = CGRect(x: x, y: y, width: cellWidth + 0.5, height: cellHeight)
                
                let prob: Float
                if frameIdx < numConfirmed {
                    prob = tl.probability(speaker: speaker, frame: frameIdx)
                } else if frameIdx < numConfirmed + numTentative {
                    let tentativeFrame = frameIdx - numConfirmed
                    prob = tl.tentativeProbability(speaker: speaker, frame: tentativeFrame)
                } else {
                    prob = 0 // Padding
                }
                
                let color = Color(cgColor: ViridisColormap.cgColor(for: prob))
                context.fill(Path(rect), with: .color(color))
            }
        }
    }
    
    /// Bitmap-based drawing for long timelines - render to CGImage then draw once
    private func drawHeatmapAsBitmap(context: GraphicsContext, size: CGSize, timeline tl: SortformerTimeline,
                                     numConfirmed: Int, numTentative: Int,
                                     cellWidth: CGFloat, cellHeight: CGFloat) {
        let totalFrames = numConfirmed + numTentative
        
        // Downsample if needed - limit bitmap to reasonable size
        let maxBitmapWidth = 4000
        let downsampleFactor = max(1, totalFrames / maxBitmapWidth)
        let bitmapWidth = (totalFrames + downsampleFactor - 1) / downsampleFactor
        let bitmapHeight = numSpeakers
        
        // Create bitmap data (RGBA, 4 bytes per pixel)
        var pixelData = [UInt8](repeating: 0, count: bitmapWidth * bitmapHeight * 4)
        
        for x in 0..<bitmapWidth {
            // Sample frame (use max probability in downsampled range for visibility)
            let frameStart = x * downsampleFactor
            let frameEnd = min(frameStart + downsampleFactor, totalFrames)
            
            for speaker in 0..<bitmapHeight {
                var maxProb: Float = 0
                
                for frameIdx in frameStart..<frameEnd {
                    let prob: Float
                    if frameIdx < numConfirmed {
                        prob = tl.probability(speaker: speaker, frame: frameIdx)
                    } else {
                        let tentativeFrame = frameIdx - numConfirmed
                        prob = tl.tentativeProbability(speaker: speaker, frame: tentativeFrame)
                    }
                    maxProb = max(maxProb, prob)
                }
                
                let (r, g, b) = ViridisColormap.rgb(for: maxProb)
                let pixelIndex = (speaker * bitmapWidth + x) * 4
                pixelData[pixelIndex] = r
                pixelData[pixelIndex + 1] = g
                pixelData[pixelIndex + 2] = b
                pixelData[pixelIndex + 3] = 255
            }
        }
        
        // Create CGImage from pixel data
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue)
        
        guard let provider = CGDataProvider(data: Data(pixelData) as CFData),
              let cgImage = CGImage(
                width: bitmapWidth,
                height: bitmapHeight,
                bitsPerComponent: 8,
                bitsPerPixel: 32,
                bytesPerRow: bitmapWidth * 4,
                space: colorSpace,
                bitmapInfo: bitmapInfo,
                provider: provider,
                decode: nil,
                shouldInterpolate: true,  // Enable interpolation for smooth scaling
                intent: .defaultIntent
              ) else {
            // Fallback: draw purple background if bitmap fails
            context.fill(Path(CGRect(origin: .zero, size: size)), 
                        with: .color(Color(cgColor: ViridisColormap.cgColor(for: 0))))
            return
        }
        
        // Draw the bitmap scaled to fill the canvas
        let image = Image(decorative: cgImage, scale: 1.0, orientation: .up)
        context.draw(image, in: CGRect(origin: .zero, size: size))
    }
    
    private func drawSegmentOutline(context: GraphicsContext, segment: SpeakerSegment,
                                    cellWidth: CGFloat, cellHeight: CGFloat, tentative: Bool) {
        let speaker = segment.slot
        guard speaker >= 0 && speaker < numSpeakers else { return }
        let startFrame = segment.startFrame
        let endFrame = segment.endFrame
        
        let x = CGFloat(startFrame) * cellWidth
        let y = CGFloat(speaker) * cellHeight
        let width = CGFloat(endFrame - startFrame) * cellWidth
        let color = speakerColor(forSlot: speaker)
        
        let lineWidth: CGFloat = 2
        let strokeStyle: StrokeStyle = tentative
            ? StrokeStyle(lineWidth: lineWidth, dash: [4, 2])
            : StrokeStyle(lineWidth: lineWidth)

        // Inset so row-0 top border is never clipped at y=0.
        let yInset = lineWidth * 0.5
        let xInset = lineWidth * 0.35
        let rect = CGRect(
            x: x + xInset,
            y: y + yInset,
            width: max(1, width - 2 * xInset),
            height: max(1, cellHeight - lineWidth)
        )
        
        context.stroke(Path(rect), with: .color(color.opacity(0.9)), style: strokeStyle)
    }

    private func drawEmbeddingSegmentBand(
        context: GraphicsContext,
        segment: EmbeddingSegment,
        cellWidth: CGFloat,
        cellHeight: CGFloat,
        tentative: Bool,
        distance: Float?
    ) {
        guard segment.speakerId >= 0 && segment.speakerId < numSpeakers else { return }

        let x = CGFloat(segment.startFrame) * cellWidth
        let width = CGFloat(segment.endFrame - segment.startFrame) * cellWidth
        guard width > 0 else { return }

        let rowY = CGFloat(segment.speakerId) * cellHeight
        let yInset: CGFloat = 1.5
        let rect = CGRect(
            x: x + 0.8,
            y: rowY + yInset,
            width: max(1, width - 1.6),
            height: max(1, cellHeight - 2 * yInset)
        )

        let color = speakerColor(forSlot: segment.speakerId)
        let fillColor = color.opacity(tentative ? 0.09 : 0.16)
        let strokeStyle = tentative
            ? StrokeStyle(lineWidth: 1.4, dash: [4, 2])
            : StrokeStyle(lineWidth: 1.6)

        context.fill(Path(roundedRect: rect, cornerRadius: 2.5), with: .color(fillColor))
        context.stroke(Path(roundedRect: rect, cornerRadius: 2.5), with: .color(color.opacity(0.72)), style: strokeStyle)

        guard let distance, rect.width > 48 else { return }
        let distanceText = Text(String(format: "d%.3f", distance))
            .font(.system(size: 9, weight: .bold, design: .monospaced))
            .foregroundColor(.white)

        let textY = rect.minY + 2
        context.draw(distanceText, at: CGPoint(x: rect.minX + 5, y: textY), anchor: .topLeading)
    }
    
    private func drawSpkcacheHeatmap(context: GraphicsContext, size: CGSize) {
        let cellWidth = size.width / CGFloat(spkcacheSize)
        let cellHeight = size.height / CGFloat(numSpeakers)
        
        let preds = spkcachePreds ?? []
        let actualFrames = preds.count / numSpeakers
        
        for frameIdx in 0..<spkcacheSize {
            let x = CGFloat(frameIdx) * cellWidth
            
            for speaker in 0..<numSpeakers {
                let y = CGFloat(speaker) * cellHeight
                let rect = CGRect(x: x, y: y, width: cellWidth + 0.5, height: cellHeight)
                
                let prob: Float
                if frameIdx < actualFrames {
                    let probIdx = frameIdx * numSpeakers + speaker
                    prob = probIdx < preds.count ? preds[probIdx] : 0
                } else {
                    prob = 0
                }
                
                let color = Color(cgColor: ViridisColormap.cgColor(for: prob))
                context.fill(Path(rect), with: .color(color))
            }
        }
    }
    
    /// Draw FIFO queue heatmap - fixed viewport, right-aligned to show recent frames
    private func drawFifoHeatmap(context: GraphicsContext, size: CGSize) {
        let preds = fifoPreds ?? []
        let fifoFrames = preds.count / numSpeakers
        
        // FIFO uses a fixed viewport of visibleFrames, with content right-aligned
        let cellWidth = size.width / CGFloat(visibleFrames)
        let cellHeight = size.height / CGFloat(numSpeakers)
        
        guard fifoFrames > 0 else {
            // Empty - background color (purple) will show through
            return
        }
        
        // Right-align the FIFO content within the fixed viewport
        // Leave chunkRightContext frames of empty space on the right
        let rightPadding = chunkRightContext
        let startX = size.width - CGFloat(fifoFrames + rightPadding) * cellWidth
        
        // Draw the FIFO predictions
        for frameIdx in 0..<fifoFrames {
            let x = startX + CGFloat(frameIdx) * cellWidth
            
            // Skip if off-screen to the left or right
            if x + cellWidth < 0 || x > size.width { continue }
            
            for speaker in 0..<numSpeakers {
                let y = CGFloat(speaker) * cellHeight
                let rect = CGRect(x: x, y: y, width: cellWidth + 0.5, height: cellHeight)
                
                let probIdx = frameIdx * numSpeakers + speaker
                let prob = probIdx < preds.count ? preds[probIdx] : 0
                
                let color = Color(cgColor: ViridisColormap.cgColor(for: prob))
                context.fill(Path(rect), with: .color(color))
            }
        }
        
        // Draw dashed boundary lines for FIFO bounds
        let fifoStartX = startX
        let fifoEndX = startX + CGFloat(fifoFrames) * cellWidth
        let dashStyle = StrokeStyle(lineWidth: 1.5, dash: [4, 3])
        let boundaryColor = Color.white.opacity(0.6)
        
        // Left boundary
        if fifoStartX >= 0 && fifoStartX <= size.width {
            var leftPath = Path()
            leftPath.move(to: CGPoint(x: fifoStartX, y: 0))
            leftPath.addLine(to: CGPoint(x: fifoStartX, y: size.height))
            context.stroke(leftPath, with: .color(boundaryColor), style: dashStyle)
        }
        
        // Right boundary
        if fifoEndX >= 0 && fifoEndX <= size.width {
            var rightPath = Path()
            rightPath.move(to: CGPoint(x: fifoEndX, y: 0))
            rightPath.addLine(to: CGPoint(x: fifoEndX, y: size.height))
            context.stroke(rightPath, with: .color(boundaryColor), style: dashStyle)
        }
    }
}

#Preview {
    SpeechPlotView(
        timeline: nil,
        spkcachePreds: nil,
        fifoPreds: nil,
        chunkRightContext: globalConfig.chunkRightContext,
        chunkLeftContext: globalConfig.chunkLeftContext,
        isRecording: false,
        updateTrigger: 0,
        segmentAnnotations: [:],
        segmentCentroidDistances: [:],
        embeddingSegmentCentroidDistances: [:],
        onPlaySegment: nil,
        onAnnotateSegment: nil,
        onPurgeSpeaker: nil
    )
    .frame(height: 500)
    .padding()
}

// MARK: - Preference Key for Scroll Tracking

private struct ScrollOffsetPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}
