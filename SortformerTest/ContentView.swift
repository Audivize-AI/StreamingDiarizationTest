//
//  ContentView.swift
//  SortformerTest
//
//  Real-time speech diarization with Sortformer
//

import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = DiarizerViewModel()
    
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
            SpeechPlotView(
                timeline: viewModel.timeline,
                spkcachePreds: viewModel.spkcachePreds,
                isRecording: viewModel.isRecording,
                updateTrigger: viewModel.updateTrigger,
                onPlaySegment: { start, end in
                    viewModel.playSegment(startTime: start, endTime: end)
                }
            )
            .frame(maxHeight: .infinity)
            .padding(.horizontal)
            
            // Control buttons
            HStack(spacing: 20) {
                Button(action: {
                    if viewModel.isRecording {
                        viewModel.stopDiarization()
                    } else {
                        viewModel.startDiarization()
                    }
                }) {
                    HStack {
                        Image(systemName: viewModel.isRecording ? "stop.fill" : "mic.fill")
                        Text(viewModel.isRecording ? "Stop" : "Start")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding(.horizontal, 24)
                    .padding(.vertical, 12)
                    .background(viewModel.isRecording ? Color.red : Color.green)
                    .cornerRadius(8)
                }
                .disabled(!viewModel.isReady)
                .buttonStyle(.plain)
            }
            .padding(.bottom)
        }
        .padding(.top)
        .frame(minWidth: 700, minHeight: 450)
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
}

#Preview {
    ContentView()
}
