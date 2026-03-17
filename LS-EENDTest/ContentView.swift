import AppKit
import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = LSEENDDemoViewModel()

    var body: some View {
        HSplitView {
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    modelSection
                    sessionSection
                    displaySection
                    speakerSection
                    statusSection
                }
                .padding(16)
            }
            .frame(minWidth: 360, idealWidth: 420)

            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    LSEENDHeatmapView(snapshot: viewModel.binarySnapshot)
                    LSEENDHeatmapView(snapshot: viewModel.probabilitySnapshot)
                }
                .padding(16)
            }
        }
        .frame(minWidth: 1280, minHeight: 900)
    }

    private var modelSection: some View {
        GroupBox("Model") {
            VStack(alignment: .leading, spacing: 12) {
                Picker("Variant", selection: $viewModel.selectedVariant) {
                    ForEach(LSEENDVariant.allCases) { variant in
                        Text(variant.rawValue).tag(variant)
                    }
                }
                .onChange(of: viewModel.selectedVariant) { _, newValue in
                    viewModel.selectVariant(newValue)
                }

                Toggle("Use custom model paths", isOn: $viewModel.useCustomPaths)
                    .onChange(of: viewModel.useCustomPaths) { _, newValue in
                        if !newValue {
                            viewModel.applyVariantDefaults()
                        }
                    }

                pathEditor(
                    title: "Model package",
                    text: $viewModel.modelPath,
                    browseAction: viewModel.browseModelPath
                )

                pathEditor(
                    title: "Metadata JSON",
                    text: $viewModel.metadataPath,
                    browseAction: viewModel.browseMetadataPath
                )

                HStack {
                    Button("Reload Model") {
                        viewModel.reloadModel()
                    }
                    Button("Use Variant Defaults") {
                        viewModel.useCustomPaths = false
                        viewModel.applyVariantDefaults()
                    }
                }

                Text(viewModel.modelInfoText)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var sessionSection: some View {
        GroupBox("Session") {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Button("Start Microphone") {
                        viewModel.startMicrophone()
                    }
                    Button("Start File") {
                        viewModel.startSimulation()
                    }
                    Button("Stop") {
                        viewModel.stopCapture()
                    }
                    Button("Reset") {
                        viewModel.resetTimeline()
                    }
                }

                pathEditor(
                    title: "Simulation file",
                    text: $viewModel.simulationPath,
                    browseAction: viewModel.browseSimulationFile
                )

                HStack {
                    VStack(alignment: .leading) {
                        Text("Block (s)")
                        TextField("", value: $viewModel.blockSeconds, format: .number.precision(.fractionLength(2)))
                            .textFieldStyle(.roundedBorder)
                    }
                    VStack(alignment: .leading) {
                        Text("Refresh (s)")
                        TextField("", value: $viewModel.refreshSeconds, format: .number.precision(.fractionLength(2)))
                            .textFieldStyle(.roundedBorder)
                    }
                    VStack(alignment: .leading) {
                        Text("Simulation speed")
                        TextField("", value: $viewModel.simulateSpeed, format: .number.precision(.fractionLength(2)))
                            .textFieldStyle(.roundedBorder)
                    }
                }

                HStack {
                    Button("Save RTTM") {
                        viewModel.saveRTTM()
                    }
                    Button("Save Session JSON") {
                        viewModel.saveSessionJSON()
                    }
                }
            }
        }
    }

    private var displaySection: some View {
        GroupBox("Display") {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading) {
                        Text("Window (s)")
                        TextField("", value: $viewModel.windowSeconds, format: .number.precision(.fractionLength(0)))
                            .textFieldStyle(.roundedBorder)
                    }
                    VStack(alignment: .leading) {
                        Text("Threshold")
                        TextField("", value: $viewModel.threshold, format: .number.precision(.fractionLength(2)))
                            .textFieldStyle(.roundedBorder)
                    }
                    VStack(alignment: .leading) {
                        Text("Median width")
                        TextField("", value: $viewModel.medianWidth, format: .number)
                            .textFieldStyle(.roundedBorder)
                    }
                }
            }
        }
    }

    private var speakerSection: some View {
        GroupBox("Speakers") {
            VStack(alignment: .leading, spacing: 8) {
                if viewModel.displayOrder.isEmpty {
                    Text("Run inference to populate speaker tracks.")
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(Array(viewModel.displayOrder.enumerated()), id: \.offset) { index, _ in
                        HStack {
                            Text("Row \(index + 1)")
                                .frame(width: 56, alignment: .leading)
                            TextField("Speaker", text: labelBinding(for: index))
                                .textFieldStyle(.roundedBorder)
                            Button("Up") {
                                viewModel.moveSpeaker(row: index, offset: -1)
                            }
                            Button("Down") {
                                viewModel.moveSpeaker(row: index, offset: 1)
                            }
                        }
                    }
                    Button("Reset Order") {
                        viewModel.resetSpeakerOrder()
                    }
                }
            }
        }
    }

    private var statusSection: some View {
        GroupBox("Status") {
            VStack(alignment: .leading, spacing: 8) {
                Text(viewModel.statusText)
                Text(viewModel.sourceText)
                    .foregroundStyle(.secondary)
                Text(viewModel.bufferText)
                    .foregroundStyle(.secondary)
                Text(viewModel.inferenceText)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func pathEditor(title: String, text: Binding<String>, browseAction: @escaping () -> Void) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
            HStack {
                TextField(title, text: text)
                    .textFieldStyle(.roundedBorder)
                Button("Browse", action: browseAction)
            }
        }
    }

    private func labelBinding(for row: Int) -> Binding<String> {
        Binding(
            get: {
                guard row < viewModel.displayOrder.count else { return "" }
                return viewModel.speakerLabels[viewModel.displayOrder[row]]
            },
            set: { newValue in
                viewModel.updateSpeakerLabel(row: row, label: newValue)
            }
        )
    }
}

private struct LSEENDHeatmapView: View {
    let snapshot: LSEENDHeatmapSnapshot

    var body: some View {
        GroupBox(snapshot.title) {
            if snapshot.matrix.rows == 0 || snapshot.matrix.columns == 0 {
                Text("No inference yet")
                    .frame(maxWidth: .infinity, minHeight: 220)
                    .foregroundStyle(.secondary)
            } else {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .top, spacing: 12) {
                        VStack(alignment: .trailing, spacing: 0) {
                            ForEach(Array(snapshot.speakerLabels.enumerated()), id: \.offset) { index, label in
                                Text(label)
                                    .font(.caption)
                                    .frame(height: max(18, 360 / CGFloat(max(snapshot.matrix.columns, 1))), alignment: .trailing)
                            }
                        }
                        .padding(.top, 6)

                        Canvas { context, size in
                            drawHeatmap(context: &context, size: size)
                        }
                        .frame(minHeight: 260, idealHeight: 360, maxHeight: 420)
                        .background(Color(NSColor.windowBackgroundColor))
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(Color.secondary.opacity(0.2), lineWidth: 1)
                        )
                    }

                    HStack {
                        Text(String(format: "%.1f s", snapshot.startSeconds))
                        Spacer()
                        Text(String(format: "%.1f s", snapshot.endSeconds))
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func drawHeatmap(context: inout GraphicsContext, size: CGSize) {
        let rows = snapshot.matrix.rows
        let columns = snapshot.matrix.columns
        let cellWidth = size.width / CGFloat(max(rows, 1))
        let cellHeight = size.height / CGFloat(max(columns, 1))

        for rowIndex in 0..<rows {
            for columnIndex in 0..<columns {
                let value = snapshot.matrix[rowIndex, columnIndex]
                let rect = CGRect(
                    x: CGFloat(rowIndex) * cellWidth,
                    y: CGFloat(columns - columnIndex - 1) * cellHeight,
                    width: max(1, cellWidth),
                    height: max(1, cellHeight)
                )
                context.fill(Path(rect), with: .color(color(for: value)))
            }
        }

        if snapshot.binary, !snapshot.vadActiveRanges.isEmpty {
            let duration = max(snapshot.endSeconds - snapshot.startSeconds, 0.001)
            for range in snapshot.vadActiveRanges where range.end > range.start {
                let startRatio = (range.start - snapshot.startSeconds) / duration
                let endRatio = (range.end - snapshot.startSeconds) / duration
                let clampedStart = max(0, min(1, startRatio))
                let clampedEnd = max(clampedStart, min(1, endRatio))
                let overlay = CGRect(
                    x: size.width * CGFloat(clampedStart),
                    y: 0,
                    width: size.width * CGFloat(clampedEnd - clampedStart),
                    height: size.height
                )
                context.fill(Path(overlay), with: .color(Color(red: 0.0, green: 0.72, blue: 0.72).opacity(0.18)))
            }
        }

        if let previewStartSeconds = snapshot.previewStartSeconds,
           let previewEndSeconds = snapshot.previewEndSeconds,
           previewEndSeconds > previewStartSeconds {
            let duration = max(snapshot.endSeconds - snapshot.startSeconds, 0.001)
            let startRatio = (previewStartSeconds - snapshot.startSeconds) / duration
            let endRatio = (previewEndSeconds - snapshot.startSeconds) / duration
            let overlay = CGRect(
                x: size.width * CGFloat(startRatio),
                y: 0,
                width: size.width * CGFloat(endRatio - startRatio),
                height: size.height
            )
            let bandHeight = min(8, max(4, size.height * 0.03))
            let topBand = CGRect(
                x: overlay.minX,
                y: 0,
                width: overlay.width,
                height: bandHeight
            )
            context.fill(Path(topBand), with: .color(.orange.opacity(0.85)))

            let startMarker = CGRect(x: overlay.minX, y: 0, width: 1, height: size.height)
            context.fill(Path(startMarker), with: .color(.orange))

            if overlay.width > 1 {
                let endMarker = CGRect(x: max(overlay.maxX - 1, overlay.minX), y: 0, width: 1, height: size.height)
                context.fill(Path(endMarker), with: .color(.orange.opacity(0.8)))
            }
        }
    }

    private func color(for value: Float) -> Color {
        if snapshot.binary {
            return value > 0 ? .black : .white
        }
        let clipped = max(0, min(1, Double(value)))
        let hue = 0.72 - (0.58 * clipped)
        let brightness = 0.25 + (0.75 * clipped)
        return Color(hue: hue, saturation: 0.85, brightness: brightness)
    }
}

#Preview {
    ContentView()
}
