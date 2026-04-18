//
//  ContentView.swift
//  LS-EEND-Test
//

import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @StateObject private var vm = DiarizerViewModel()

    enum Mode: String, CaseIterable, Identifiable {
        case file, microphone, enroll
        var id: String { rawValue }
        var label: String {
            switch self {
            case .file: return "File"
            case .microphone: return "Microphone"
            case .enroll: return "Enroll"
            }
        }
    }

    @State private var mode: Mode = .file
    @State private var showProcessImporter: Bool = false
    @State private var showEnrollImporter: Bool = false
    @State private var enrollName = "Speaker"

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            controls
            Divider()
            TimelineHeatmapView(
                finalized: vm.finalizedPredictions,
                tentative: vm.tentativePredictions,
                finalizedFrameCount: vm.finalizedFrameCount,
                tentativeFrameCount: vm.tentativeFrameCount,
                numSpeakers: vm.numSpeakerChannels,
                frameDurationSeconds: vm.frameDurationSeconds,
                speakerNames: vm.speakers,
                windowSeconds: 60
            )
            .frame(minHeight: 140, idealHeight: 200)
            .padding(.horizontal)
            Divider()
            SegmentListView(
                segments: vm.segments,
                speakers: vm.speakers,
                rename: { slot, name in vm.renameSpeaker(slot: slot, to: name) }
            )
        }
        // Two separate importers — each bound to its own @State bool so the
        // completion handler can act on a single, stable action. Avoids the
        // old shared-state timing workaround where isPresented was read
        // before the setter had run.
        .fileImporter(
            isPresented: $showProcessImporter,
            allowedContentTypes: [UTType.audio],
            allowsMultipleSelection: false
        ) { result in
            guard case .success(let urls) = result, let url = urls.first else { return }
            Task { await vm.processFile(url) }
        }
        .fileImporter(
            isPresented: $showEnrollImporter,
            allowedContentTypes: [UTType.audio],
            allowsMultipleSelection: false
        ) { result in
            guard case .success(let urls) = result, let url = urls.first else { return }
            Task { await vm.enroll(name: enrollName, from: url) }
        }
    }

    private var header: some View {
        HStack {
            Text("LS-EEND Diarizer")
                .font(.title2).bold()
            Spacer()
            ModelPickerView(variant: $vm.variant, stepSize: $vm.stepSize)
                .disabled(vm.isProcessing)
                .onChange(of: vm.variant) { _, newValue in
                    Task { await vm.reload(variant: newValue) }
                }
                .onChange(of: vm.stepSize) { _, newValue in
                    Task { await vm.reload(stepSize: newValue) }
                }
        }
        .padding()
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Mode", selection: $mode) {
                ForEach(Mode.allCases) { m in Text(m.label).tag(m) }
            }
            .pickerStyle(.segmented)
            .accessibilityLabel("Mode")

            switch mode {
            case .file:
                HStack {
                    Button("Pick audio file") { showProcessImporter = true }
                        .disabled(!isReady || vm.isProcessing)
                    if vm.isProcessing {
                        ProgressView(value: vm.progress).frame(width: 120)
                    }
                    Spacer()
                    Button("Reset") { vm.reset() }
                        .disabled(vm.isProcessing)
                }
            case .microphone:
                HStack {
                    Button("Start mic") {
                        Task {
                            do { try await vm.startMicrophone() }
                            catch { vm.statusMessage = error.localizedDescription }
                        }
                    }.disabled(!isReady)
                    Button("Stop & finalize") {
                        Task { await vm.stopMicrophone() }
                    }
                    Spacer()
                    Button("Reset") { vm.reset() }
                }
            case .enroll:
                HStack {
                    TextField("Name", text: $enrollName).frame(maxWidth: 160)
                    Button("Enroll from file") { showEnrollImporter = true }
                        .disabled(!isReady || enrollName.isEmpty)
                    Spacer()
                    Text("LS-EEND learns speakers online — enrolling conditions state and names the first-detected slot.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }

            HStack {
                loadingIndicator
                Text(vm.statusMessage).font(.caption).foregroundStyle(.secondary)
                Spacer()
            }
        }
        .padding()
    }

    private var isReady: Bool {
        if case .ready = vm.loadState { return true }
        return false
    }

    @ViewBuilder
    private var loadingIndicator: some View {
        switch vm.loadState {
        case .idle:
            EmptyView()
        case .loading(let p):
            HStack(spacing: 4) {
                ProgressView(value: p).frame(width: 80)
                Text("Loading…").font(.caption2)
            }
        case .ready:
            Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed(let msg):
            Label(msg, systemImage: "exclamationmark.triangle.fill")
                .font(.caption2)
                .foregroundStyle(.red)
                .lineLimit(1)
        }
    }
}

#Preview {
    ContentView()
}
