import SwiftUI
import UniformTypeIdentifiers
import AppKit

struct ContentView: View {
    @StateObject private var viewModel = DiarizerViewModel()
    @State private var showingFilePicker = false
    @State private var showingExportPicker = false
    @State private var isDragOver = false
    
    // Annotation dialog state
    @State private var showingAnnotationDialog = false
    @State private var selectedSegment: SpeakerSegment?
    @State private var annotationText = ""
    
    
    var body: some View {
        VStack(spacing: 16) {
            // Header with status
            HStack {
                Image(systemName: "waveform.circle.fill")
                    .font(.title)
                    .foregroundColor(viewModel.isRecording ? .green : .blue)
                
                Text("Sortformer Diarization")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Spacer()
                
                // Progress indicators
                if let fileProgress = viewModel.fileProcessingProgress {
                    HStack(spacing: 8) {
                        Text("Processing:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        ProgressView(value: fileProgress)
                            .frame(width: 100)
                        Text("\(Int(fileProgress * 100))%")
                            .font(.caption.monospacedDigit())
                            .foregroundColor(.secondary)
                            .frame(width: 35)
                    }
                } else {
                    // Status indicator
                    HStack(spacing: 6) {
                        Circle()
                            .fill(statusColor)
                            .frame(width: 8, height: 8)
                        Text(viewModel.statusMessage)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
            }
            .padding(.horizontal)
            
            // Speech plot and K-means grouping dendrogram (updates with embedding streams)
            HStack(spacing: 0) {
                // Left side: probability heatmaps + segment-only identity timeline
                VStack(spacing: 10) {
                    ZStack {
                        SpeechPlotView(
                            timeline: viewModel.timeline,
                            spkcachePreds: viewModel.spkcachePreds,
                            fifoPreds: viewModel.fifoPreds,
                            chunkRightContext: viewModel.chunkRightContext,
                            chunkLeftContext: viewModel.chunkLeftContext,
                            isRecording: viewModel.isRecording,
                            updateTrigger: viewModel.updateTrigger,
                            segmentAnnotations: viewModel.segmentAnnotations,
                            segmentCentroidDistances: viewModel.segmentCentroidDistances,
                            embeddingSegmentCentroidDistances: viewModel.embeddingSegmentCentroidDistances,
                            onPlaySegment: { start, end in
                                viewModel.playSegment(startTime: start, endTime: end)
                            },
                            onAnnotateSegment: { segment in
                                // Show annotation dialog
                                selectedSegment = segment
                                annotationText = viewModel.getAnnotation(for: segment) ?? ""
                                showingAnnotationDialog = true
                            },
                            onPurgeSpeaker: { speakerIndex in
                                viewModel.purgeSpeaker(at: speakerIndex)
                            }
                        )

                        // Drag and drop overlay
                        if isDragOver {
                            RoundedRectangle(cornerRadius: 12)
                                .stroke(Color.blue, style: StrokeStyle(lineWidth: 3, dash: [10]))
                                .background(Color.blue.opacity(0.1))
                                .cornerRadius(12)
                                .overlay(
                                    VStack(spacing: 8) {
                                        Image(systemName: "arrow.down.doc.fill")
                                            .font(.system(size: 48))
                                            .foregroundColor(.blue)
                                        Text("Drop audio file here")
                                            .font(.headline)
                                            .foregroundColor(.blue)
                                    }
                                )
                        }
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)

                    SpeakerIdentityTimelineView(
                        timeline: viewModel.timeline,
                        isRecording: viewModel.isRecording,
                        updateTrigger: viewModel.updateTrigger,
                        onPlaySegment: { start, end in
                            viewModel.playSegment(startTime: start, endTime: end)
                        }
                    )
                    .frame(minHeight: 110, maxHeight: 200)
                }
                .layoutPriority(1)
                .frame(minWidth: 400, maxWidth: .infinity, maxHeight: .infinity)

                KMeansDendrogramView(model: viewModel.dendrogramModel)
                    .frame(minWidth: 320, maxWidth: 440, maxHeight: .infinity)
                    .padding(.leading, 12)
            }
            .padding(.horizontal)
            .onDrop(of: [.audio, .fileURL], isTargeted: $isDragOver) { providers in
                handleDrop(providers: providers)
            }
            
            // Control buttons
            HStack(spacing: 16) {
                // Record button
                Button(action: {
                    if viewModel.isRecording {
                        viewModel.stopDiarization()
                    } else {
                        viewModel.startDiarization()
                    }
                }) {
                    HStack {
                        Image(systemName: viewModel.isRecording ? "stop.fill" : "mic.fill")
                        Text(viewModel.isRecording ? "Stop" : "Record")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(viewModel.isRecording ? Color.red : Color.green)
                    .cornerRadius(8)
                }
                .disabled(!viewModel.isReady)
                .buttonStyle(.plain)
                
                // Import button
                Button(action: {
                    showingFilePicker = true
                }) {
                    HStack {
                        Image(systemName: "folder.fill")
                        Text("Import")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color.blue)
                    .cornerRadius(8)
                }
                .disabled(!viewModel.isReady || viewModel.isRecording)
                .buttonStyle(.plain)
                
                // Save button
                Button(action: {
                    saveRecording()
                }) {
                    HStack {
                        Image(systemName: "square.and.arrow.down.fill")
                        Text("Save")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color.orange)
                    .cornerRadius(8)
                }
                .disabled(viewModel.isRecording || viewModel.recordedAudio.isEmpty)
                .buttonStyle(.plain)
                
                // Export segments button
                Button(action: {
                    showingExportPicker = true
                }) {
                    HStack {
                        Image(systemName: "doc.text.fill")
                        Text("Export")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color.purple)
                    .cornerRadius(8)
                }
                .disabled(viewModel.isRecording || viewModel.timeline == nil)
                .buttonStyle(.plain)
                

            }
            .padding(.bottom)
        }
        .padding(.top)
        .frame(minWidth: 1150, minHeight: 650)
        .fileImporter(
            isPresented: $showingFilePicker,
            allowedContentTypes: [
                .audio,      // Generic audio
                .wav,        // WAV
                .mp3,        // MP3
                .aiff,       // AIFF
                .mpeg4Audio, // M4A/AAC
                UTType("com.apple.coreaudio-format")!, // CAF
                UTType("org.xiph.flac") ?? .audio,     // FLAC
            ],
            allowsMultipleSelection: false
        ) { result in
            switch result {
            case .success(let urls):
                if let url = urls.first {
                    Task {
                        await viewModel.loadAudioFile(from: url)
                    }
                }
            case .failure(let error):
                print("File picker error: \(error)")
            }
        }
        .fileExporter(
            isPresented: $showingExportPicker,
            document: SegmentDocument(content: exportContent),
            contentType: .plainText,
            defaultFilename: "segments.txt"
        ) { result in
            switch result {
            case .success(let url):
                print("Exported segments to: \(url)")
            case .failure(let error):
                print("Export error: \(error)")
            }
        }
        .sheet(isPresented: $showingAnnotationDialog) {
            if let segment = selectedSegment {
                VStack(spacing: 16) {
                    Text("Annotate Segment")
                        .font(.headline)
                    
                    Text("Speaker \(segment.slot): \(String(format: "%.2f", segment.startTime))s - \(String(format: "%.2f", segment.endTime))s")
                        .foregroundColor(.secondary)
                    
                    TextField("Label (e.g., name or ID)", text: $annotationText)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 200)
                    
                    HStack(spacing: 12) {
                        Button("Play") {
                            viewModel.playSegment(startTime: segment.startTime, endTime: segment.endTime)
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Clear") {
                            annotationText = ""
                            viewModel.setAnnotation(for: segment, label: "")
                        }
                        .buttonStyle(.bordered)
                        
                        Button("Save") {
                            viewModel.setAnnotation(for: segment, label: annotationText)
                            showingAnnotationDialog = false
                        }
                        .buttonStyle(.borderedProminent)
                        
                        Button("Cancel") {
                            showingAnnotationDialog = false
                        }
                        .buttonStyle(.bordered)
                    }
                }
                .padding(24)
                .frame(minWidth: 350, minHeight: 180)
            }
        }
    }
    
    private var statusColor: Color {
        if viewModel.isRecording {
            return .green
        } else if viewModel.isReady {
            return .blue
        } else {
            return .orange
        }
    }
    
    /// Generate export content for segments
    private var exportContent: String {
        guard let tl = viewModel.timeline else {
            return "speaker_id,start_time,end_time\n"
        }
        
        var output = "speaker_id,start_time,end_time\n"
        
        // Collect all segments and sort by start time
        let allSegments = tl.segments.flatMap { $0 }.sorted { $0.startTime < $1.startTime }
        
        for segment in allSegments {
            // Use annotation if available, otherwise use original speaker index
            let label = viewModel.getAnnotation(for: segment) ?? String(segment.slot)
            let startStr = String(format: "%.2f", segment.startTime)
            let endStr = String(format: "%.2f", segment.endTime)
            output += "\(label),\(startStr),\(endStr)\n"
        }
        
        return output
    }
    
    private func saveRecording() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.wav]
        panel.nameFieldStringValue = "recording.wav"
        panel.title = "Save Recording"
        
        panel.begin { response in
            if response == .OK, let url = panel.url {
                Task { @MainActor in
                    viewModel.saveRecording(to: url)
                }
            }
        }
    }
    
    private func handleDrop(providers: [NSItemProvider]) -> Bool {
        guard let provider = providers.first else { return false }
        
        // Check if it's an audio file or file URL
        let supportedTypes: [UTType] = [.audio, .wav, .mp3, .aiff, .mpeg4Audio, .fileURL]
        guard supportedTypes.contains(where: { provider.hasItemConformingToTypeIdentifier($0.identifier) }) else {
            return false
        }
        
        // Use loadFileRepresentation for proper file access
        _ = provider.loadFileRepresentation(forTypeIdentifier: UTType.audio.identifier) { url, error in
            guard let url = url else {
                // Try as generic file URL
                provider.loadItem(forTypeIdentifier: UTType.fileURL.identifier) { item, error in
                    if let data = item as? Data,
                       let fileURL = URL(dataRepresentation: data, relativeTo: nil) {
                        Task { @MainActor in
                            await viewModel.loadAudioFile(from: fileURL)
                        }
                    }
                }
                return
            }
            
            // Copy to temp location since the provided URL is temporary
            let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(url.lastPathComponent)
            try? FileManager.default.removeItem(at: tempURL)
            try? FileManager.default.copyItem(at: url, to: tempURL)
            
            Task { @MainActor in
                await viewModel.loadAudioFile(from: tempURL)
            }
        }
        return true
    }
}

// MARK: - Segment Document for Export

struct SegmentDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.plainText] }
    
    let content: String
    
    init(content: String) {
        self.content = content
    }
    
    init(configuration: ReadConfiguration) throws {
        if let data = configuration.file.regularFileContents {
            content = String(data: data, encoding: .utf8) ?? ""
        } else {
            content = ""
        }
    }
    
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        let data = content.data(using: .utf8) ?? Data()
        return FileWrapper(regularFileWithContents: data)
    }
}

#Preview {
    ContentView()
}
