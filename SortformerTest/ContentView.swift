//
//  ContentView.swift
//  SortformerTest
//
//  Real-time speech diarization with Sortformer
//

import SwiftUI
import UniformTypeIdentifiers
import AppKit

struct ContentView: View {
    @StateObject private var viewModel = DiarizerViewModel()
    @State private var showingFilePicker = false
    @State private var isDragOver = false
    
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
            .padding(.horizontal)
            
            // Speech plot - fills available space
            ZStack {
                SpeechPlotView(
                    timeline: viewModel.timeline,
                    spkcachePreds: viewModel.spkcachePreds,
                    isRecording: viewModel.isRecording,
                    updateTrigger: viewModel.updateTrigger,
                    onPlaySegment: { start, end in
                        viewModel.playSegment(startTime: start, endTime: end)
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
            .frame(maxHeight: .infinity)
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
            }
            .padding(.bottom)
        }
        .padding(.top)
        .frame(minWidth: 800, minHeight: 500)
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

#Preview {
    ContentView()
}
