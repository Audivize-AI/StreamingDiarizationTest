import SwiftUI

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
    
    /// Auto-scroll delay after user interaction
    @State private var lastScrollTime: Date = Date()
    @State private var userIsScrolling = false
    
    // Segment outline colors per speaker
    private let speakerColors: [Color] = [
        .red, .green, .blue, .orange
    ]
    
    var body: some View {
        GeometryReader { geometry in
            let availableHeight = geometry.size.height - 80
            let mainPlotHeight = availableHeight * 0.6
            let spkcachePlotHeight = availableHeight * 0.3
            
            VStack(alignment: .leading, spacing: 8) {
                // Title with stats
                Text(titleText)
                    .font(.headline)
                    .padding(.horizontal, 8)
                
                // Main diarization heatmap with labels
                mainHeatmapSection(width: geometry.size.width - 60, height: mainPlotHeight)
                
                // X-axis label
                Text("Time (frames, 80ms each) - \(totalFrameCount) total frames")
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .padding(.leading, 49)
                
                Divider()
                
                // Speaker cache visualization (always shown)
                spkcacheSection(width: geometry.size.width - 60, height: spkcachePlotHeight)
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
                    Canvas { context, size in
                        drawMainHeatmap(context: context, size: size, viewportWidth: width)
                    }
                    .id("canvas-\(updateTrigger)")  // Force redraw on updates
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
                }
                .frame(height: height)
                .background(Color(white: 0.1))
                .cornerRadius(8)
                .simultaneousGesture(
                    DragGesture()
                        .onChanged { _ in
                            userIsScrolling = true
                            lastScrollTime = Date()
                        }
                        .onEnded { _ in
                            userIsScrolling = false
                            lastScrollTime = Date()
                        }
                )
                .onChange(of: updateTrigger) { _, _ in
                    // Auto-scroll after 3 seconds of inactivity while recording
                    let timeSinceScroll = Date().timeIntervalSince(lastScrollTime)
                    if isRecording && !userIsScrolling && timeSinceScroll > 3.0 {
                        withAnimation(.easeOut(duration: 0.2)) {
                            scrollProxy.scrollTo("canvas-\(updateTrigger)", anchor: .trailing)
                        }
                    }
                }
                .onAppear {
                    scrollProxy.scrollTo("canvas-\(updateTrigger)", anchor: .trailing)
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
    
    // MARK: - Drawing Functions
    
    private func drawMainHeatmap(context: GraphicsContext, size: CGSize, viewportWidth: CGFloat) {
        guard let tl = timeline else {
            // Draw empty placeholder
            context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(Color(white: 0.1)))
            return
        }
        
        let numConfirmed = tl.numFrames
        let numTentative = tl.numTentative
        let framesToDraw = max(numConfirmed + numTentative, visibleFrames)
        
        let cellWidth = size.width / CGFloat(framesToDraw)
        let cellHeight = size.height / CGFloat(numSpeakers)
        
        // Draw probability cells
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
