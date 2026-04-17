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
    @State private var importerKind: ImporterKind?
    @State private var enrollName = "Speaker"

    private enum ImporterKind: Identifiable {
        case process, enroll
        var id: Self { self }
    }

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
        .fileImporter(
            isPresented: Binding(
                get: { importerKind != nil },
                // Don't clear importerKind here — SwiftUI toggles isPresented
                // off *before* the completion handler runs, so clearing now
                // makes the completion see a nil kind and skip the dispatch.
                set: { _ in }
            ),
            allowedContentTypes: [UTType.audio],
            allowsMultipleSelection: false
        ) { result in
            let kind = importerKind
            importerKind = nil
            guard case .success(let urls) = result, let url = urls.first else {
                return
            }
            print("[ContentView] fileImporter completed kind=\(String(describing: kind)) url=\(url.path)")
            switch kind {
            case .process:
                Task { await vm.processFile(url) }
            case .enroll:
                Task { await vm.enroll(name: enrollName, from: url) }
            case .none:
                break
            }
        }
    }

    private var header: some View {
        HStack {
            Text("LS-EEND Diarizer")
                .font(.title2).bold()
            Spacer()
            ModelPickerView(
                variant: Binding(
                    get: { vm.variant },
                    set: { v in Task { await vm.reload(variant: v) } }
                ),
                stepSize: Binding(
                    get: { vm.stepSize },
                    set: { s in Task { await vm.reload(stepSize: s) } }
                )
            )
            .disabled(vm.isProcessing)
        }
        .padding()
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Mode", selection: $mode) {
                ForEach(Mode.allCases) { m in Text(m.label).tag(m) }
            }
            .pickerStyle(.segmented)

            switch mode {
            case .file:
                HStack {
                    Button("Pick audio file") { importerKind = .process }
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
                        Task { try? await vm.startMicrophone() }
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
                    Button("Enroll from file") { importerKind = .enroll }
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
