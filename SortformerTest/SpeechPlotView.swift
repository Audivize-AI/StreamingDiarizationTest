import SwiftUI
import CoreGraphics

/// Real-time speech diarization heatmap view.
/// Displays speaker probabilities as a viridis-colored heatmap with 4 speaker rows.
struct SpeechPlotView: View {
    /// Timeline with diarization history
    let timeline: SortformerTimeline?
    
    /// Speaker cache predictions (optional, drawn as zeros if nil)
    let spkcachePreds: [Float]?
    
    /// Whether recording is active (for auto-scroll behavior)
    let isRecording: Bool
    
    /// Update trigger to force redraws
    let updateTrigger: Int
    
    /// Callback when a segment is clicked (start, end)
    let onPlaySegment: ((Float, Float) -> Void)?
    
    /// Visible frames in the viewport (determines cell width relative to window)
    private let visibleFrames = 188
    
    /// Speaker cache size
    private let spkcacheSize = 188
    
    /// Number of speakers (fixed at 4 for Sortformer)
    private let numSpeakers = 4
    
    /// Maximum frames to draw individually (above this, use bitmap rendering)
    private let maxFramesForDetailedDraw = 5000
    
    /// Track if we're at the trailing edge (following live updates)
    @State private var isFollowingLive = true
    
    // Segment outline colors per speaker
    private let speakerColors: [Color] = [
        .red, .green, .blue, .orange
    ]
    
    var body: some View {
        GeometryReader { geometry in
            let availableHeight = geometry.size.height - 80
            let plotHeight = availableHeight * 0.45  // Same height for both plots
            
            VStack(alignment: .leading, spacing: 8) {
                // Title with stats
                Text(titleText)
                    .font(.headline)
                    .padding(.horizontal, 8)
                
                // Main diarization heatmap with labels
                mainHeatmapSection(width: geometry.size.width - 60, height: plotHeight)
                
                // X-axis label
                Text("Time (frames, 80ms each) - \(totalFrameCount) total frames")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.leading, 49)
                
                Divider()
                
                // Speaker cache visualization (always shown)
                spkcacheSection(width: geometry.size.width - 60, height: plotHeight)
            }
        }
        .padding(8)
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
            
            // Scrollable heatmap - keeps ALL history
            ScrollViewReader { scrollProxy in
                ScrollView(.horizontal, showsIndicators: true) {
                    HStack(spacing: 0) {
                        Canvas { context, size in
                            drawMainHeatmap(context: context, size: size, viewportWidth: width)
                        }
                        .drawingGroup()  // Renders to bitmap - better performance
                        .frame(width: calculatedContentWidth(viewportWidth: width), height: height)
                        .onTapGesture { location in
                            guard !isRecording, let tl = timeline else { return }
                            
                            // Calculate frame and speaker from tap location
                            let cellWidth = calculatedContentWidth(viewportWidth: width) / CGFloat(max(tl.numFrames + tl.numTentative, visibleFrames))
                            let cellHeight = height / CGFloat(numSpeakers)
                            let clickedFrame = Int(location.x / cellWidth)
                            let clickedSpeaker = Int(location.y / cellHeight)
                            
                            // Find segment covering this frame AND speaker
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
                        
                        // Stable scroll target at the trailing edge
                        Color.clear
                            .frame(width: 1, height: height)
                            .id("scrollEnd")
                    }
                }
                .coordinateSpace(name: "scrollView")
                .onPreferenceChange(ScrollOffsetPreferenceKey.self) { offset in
                    // Check if we're near the trailing edge (within 50 points)
                    let contentWidth = calculatedContentWidth(viewportWidth: width)
                    let maxOffset = -(contentWidth - width)
                    let isNearEnd = offset <= maxOffset + 50
                    
                    if isNearEnd && !isFollowingLive {
                        isFollowingLive = true
                    } else if !isNearEnd && isFollowingLive && isRecording {
                        isFollowingLive = false
                    }
                }
                .frame(height: height)
                .background(Color(white: 0.1))
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
                .background(Color(white: 0.1))
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
    
    // MARK: - Optimized Drawing Functions
    
    private func drawMainHeatmap(context: GraphicsContext, size: CGSize, viewportWidth: CGFloat) {
        guard let tl = timeline else {
            // Draw empty placeholder
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(white: 0.1)))
            return
        }
        
        let numConfirmed = tl.numFrames
        let numTentative = tl.numTentative
        let totalFrames = numConfirmed + numTentative
        let framesToDraw = max(totalFrames, visibleFrames)
        
        let cellWidth = size.width / CGFloat(framesToDraw)
        let cellHeight = size.height / CGFloat(numSpeakers)
        
        // Use bitmap rendering for large timelines
        if totalFrames > maxFramesForDetailedDraw {
            drawHeatmapAsBitmap(context: context, size: size, timeline: tl, 
                               numConfirmed: numConfirmed, numTentative: numTentative,
                               cellWidth: cellWidth, cellHeight: cellHeight)
        } else {
            drawHeatmapDetailed(context: context, timeline: tl,
                               numConfirmed: numConfirmed, numTentative: numTentative,
                               framesToDraw: framesToDraw, cellWidth: cellWidth, cellHeight: cellHeight)
        }
        
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
    
    /// Detailed drawing for short timelines - individual rectangles
    private func drawHeatmapDetailed(context: GraphicsContext, timeline tl: SortformerTimeline,
                                     numConfirmed: Int, numTentative: Int,
                                     framesToDraw: Int, cellWidth: CGFloat, cellHeight: CGFloat) {
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
}

#Preview {
    SpeechPlotView(
        timeline: nil,
        spkcachePreds: nil,
        isRecording: false,
        updateTrigger: 0,
        onPlaySegment: nil
    )
    .frame(height: 400)
    .padding()
}

// MARK: - Preference Key for Scroll Tracking

private struct ScrollOffsetPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0
    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}
