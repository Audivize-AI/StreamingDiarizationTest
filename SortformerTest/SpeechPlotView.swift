import SwiftUI
import CoreGraphics

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
    
    /// Callback when a segment is clicked (start, end)
    let onPlaySegment: ((Float, Float) -> Void)?
    
    /// Callback when a segment is clicked for annotation
    let onAnnotateSegment: ((SortformerSegment) -> Void)?
    
    /// Visible frames in the viewport (determines cell width relative to window)
    private let visibleFrames = 188
    
    /// Speaker cache size
    private let spkcacheSize = 188
    
    /// Number of speakers (fixed at 4 for Sortformer)
    private let numSpeakers = 4
    
    /// Chunk size for virtualization (frames per chunk)
    private let chunkSize = 500
    
    /// Buffer chunks to render outside visible area
    private let bufferChunks = 2
    
    /// Track if we're at the trailing edge (following live updates)
    @State private var isFollowingLive = true
    
    /// Current scroll offset for virtualization
    @State private var scrollOffset: CGFloat = 0
    
    /// Hovered segment info for tooltip
    @State private var hoveredSegment: SortformerSegment?
    @State private var hoverLocation: CGPoint = .zero
    
    // Segment outline colors per speaker
    private let speakerColors: [Color] = [
        .red, .green, .blue, .orange
    ]
    
    var body: some View {
        GeometryReader { geometry in
            let availableHeight = geometry.size.height - 80  // Reserved for title/labels/padding
            let plotHeight = availableHeight / 3.0  // Equal height for all 3 plots
            // Subtract 49 for Y-axis labels (45) + spacing (4) to get actual plot width
            let plotWidth = geometry.size.width - 49
            
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
                    .padding(.leading, 49)
                
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
        HStack(alignment: .top, spacing: 4) {
            // Y-axis labels
            VStack(spacing: 0) {
                ForEach(0..<numSpeakers, id: \.self) { speaker in
                    Text("Spk \(speaker)")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .frame(maxHeight: .infinity)
                }
            }
            .frame(width: 45, height: height)
            
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
                                         cellWidth: cellWidth, cellHeight: height / CGFloat(numSpeakers))
                            }
                            .drawingGroup()
                            .frame(width: chunkWidth, height: height)
                            .id("chunk-\(chunkIndex)")
                        }
                        
                        // Draw segment overlays on top (single canvas for segments)
                        // Moved to overlay for proper layering
                        
                        // Stable scroll target at the trailing edge
                        Color.clear
                            .frame(width: 1, height: height)
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
                            // Find segment at hover location
                            guard let tl = timeline else {
                                hoveredSegment = nil
                                return
                            }
                            let cellHeight = height / CGFloat(numSpeakers)
                            let hoverFrame = Int(location.x / cellWidth)
                            let hoverSpeaker = Int(location.y / cellHeight)
                            
                            // Check finalized segments
                            if let segment = tl.segments.flatMap(\.self).first(where: {
                                hoverFrame >= $0.startFrame && hoverFrame < $0.endFrame && $0.speakerIndex == hoverSpeaker
                            }) {
                                hoveredSegment = segment
                            } else if let segment = tl.tentativeSegments.flatMap(\.self).first(where: {
                                hoverFrame >= $0.startFrame && hoverFrame < $0.endFrame && $0.speakerIndex == hoverSpeaker
                            }) {
                                hoveredSegment = segment
                            } else {
                                hoveredSegment = nil
                            }
                        case .ended:
                            hoveredSegment = nil
                        }
                    }
                    .overlay(alignment: .topLeading) {
                        // Tooltip for hovered segment
                        if let segment = hoveredSegment {
                            let startStr = String(format: "%.2f", segment.startTime)
                            let endStr = String(format: "%.2f", segment.endTime)
                            let durationStr = String(format: "%.2f", segment.endTime - segment.startTime)
                            
                            Text("Spk \(segment.speakerIndex): \(startStr)s - \(endStr)s (\(durationStr)s)")
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
                        
                        let cellHeight = height / CGFloat(numSpeakers)
                        let clickedFrame = Int(location.x / cellWidth)
                        let clickedSpeaker = Int(location.y / cellHeight)
                        
                        if let segment = tl.segments.flatMap(\.self).first(where: {
                            clickedFrame >= $0.startFrame && clickedFrame < $0.endFrame && $0.speakerIndex == clickedSpeaker
                        }) {
                            onAnnotateSegment?(segment)
                        }
                    }
                    .onTapGesture(count: 1) { location in
                        // SINGLE CLICK = PLAY
                        guard !isRecording, let tl = timeline else { return }
                        
                        let cellHeight = height / CGFloat(numSpeakers)
                        let clickedFrame = Int(location.x / cellWidth)
                        let clickedSpeaker = Int(location.y / cellHeight)
                        
                        if let segment = tl.segments.flatMap(\.self).first(where: { 
                            clickedFrame >= $0.startFrame && clickedFrame < $0.endFrame && $0.speakerIndex == clickedSpeaker
                        }) {
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
                .frame(height: height)
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
                .frame(width: 45, height: height)
                
                // FIFO heatmap - fixed width, right-aligned content
                Canvas { context, size in
                    drawFifoHeatmap(context: context, size: size)
                }
                .frame(width: width, height: height)
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
                .frame(width: 45, height: height)
                
                // Speaker cache heatmap
                Canvas { context, size in
                    drawSpkcacheHeatmap(context: context, size: size)
                }
                .id("spkcache-\(updateTrigger)")  // Force redraw on updates
                .frame(width: width, height: height)
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
        let totalFrames = tl.numFrames + tl.numTentative
        let elapsedSeconds = Float(totalFrames) * 0.08
        return String(format: "Live Diarization - Confirmed: %d | Tentative: %d | Segments: %d + %d | (%.1fs)",
                      tl.numFrames, tl.numTentative,
                      tl.segments.flatMap(\.self).count, tl.tentativeSegments.flatMap(\.self).count,
                      elapsedSeconds)
    }
    
    private var totalFrameCount: Int {
        guard let tl = timeline else { return 0 }
        return tl.numFrames + tl.numTentative
    }
    
    private func calculatedContentWidth(viewportWidth: CGFloat) -> CGFloat {
        guard let tl = timeline else {
            return viewportWidth
        }
        let totalFrames = tl.numFrames + tl.numTentative
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
        
        let numConfirmed = tl.numFrames
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
        
        let totalFrames = max(tl.numFrames + tl.numTentative, visibleFrames)
        let cellWidth = size.width / CGFloat(totalFrames)
        let cellHeight = size.height / CGFloat(numSpeakers)
        
        // Draw finalized segments (solid) with labels
        for segment in tl.segments.flatMap({ $0 }) {
            drawSegmentOutline(context: context, segment: segment, 
                             cellWidth: cellWidth, cellHeight: cellHeight, tentative: false)
            drawSegmentLabel(context: context, segment: segment,
                           cellWidth: cellWidth, cellHeight: cellHeight)
        }
        
        // Draw tentative segments (dashed) - no labels for tentative
        for segment in tl.tentativeSegments.flatMap({ $0 }) {
            drawSegmentOutline(context: context, segment: segment, 
                             cellWidth: cellWidth, cellHeight: cellHeight, tentative: true)
        }
    }
    
    /// Draw label on a segment
    private func drawSegmentLabel(context: GraphicsContext, segment: SortformerSegment,
                                  cellWidth: CGFloat, cellHeight: CGFloat) {
        let x = CGFloat(segment.startFrame) * cellWidth
        let y = CGFloat(segment.speakerIndex) * cellHeight
        let segmentWidth = CGFloat(segment.endFrame - segment.startFrame) * cellWidth
        
        // Get annotation or use speaker index
        let key = DiarizerViewModel.segmentKey(segment)
        let label = segmentAnnotations[key] ?? String(segment.speakerIndex)
        
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
        
        let numConfirmed = tl.numFrames
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
        let segments = tl.segments.flatMap { $0 }
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
    
    private func drawSegmentOutline(context: GraphicsContext, segment: SortformerSegment,
                                    cellWidth: CGFloat, cellHeight: CGFloat, tentative: Bool) {
        let speaker = segment.speakerIndex
        let startFrame = segment.startFrame
        let endFrame = segment.endFrame
        
        let x = CGFloat(startFrame) * cellWidth
        let y = CGFloat(speaker) * cellHeight
        let width = CGFloat(endFrame - startFrame) * cellWidth
        
        let rect = CGRect(x: x, y: y, width: width, height: cellHeight)
        let color = speakerColors[speaker % speakerColors.count]
        
        let lineWidth: CGFloat = 2
        let strokeStyle: StrokeStyle = tentative
            ? StrokeStyle(lineWidth: lineWidth, dash: [4, 2])
            : StrokeStyle(lineWidth: lineWidth)
        
        context.stroke(Path(rect), with: .color(color.opacity(0.9)), style: strokeStyle)
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
        chunkRightContext: 7,
        chunkLeftContext: 1,
        isRecording: false,
        updateTrigger: 0,
        segmentAnnotations: [:],
        onPlaySegment: nil,
        onAnnotateSegment: nil
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
