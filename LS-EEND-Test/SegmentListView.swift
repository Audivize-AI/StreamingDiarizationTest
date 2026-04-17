//
//  SegmentListView.swift
//  LS-EEND-Test
//

import SwiftUI

struct SegmentListView: View {
    let segments: [DiarizerSegment]
    let speakers: [Int: String?]
    let rename: (Int, String) -> Void

    @State private var editingSlot: Int?
    @State private var editingName: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text("Segments (\(segments.count))").font(.headline)
                Spacer()
            }
            .padding(.horizontal)
            .padding(.top, 8)

            if segments.isEmpty {
                Text("No segments yet.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                List {
                    ForEach(segments) { seg in
                        HStack(spacing: 12) {
                            Text(speakerLabel(slot: seg.speakerIndex))
                                .frame(width: 120, alignment: .leading)
                                .onTapGesture(count: 2) {
                                    editingSlot = seg.speakerIndex
                                    editingName = speakers[seg.speakerIndex].flatMap { $0 } ?? ""
                                }
                            Text(String(format: "%6.2fs → %6.2fs", seg.startTime, seg.endTime))
                                .font(.system(.body, design: .monospaced))
                            Text(String(format: "%.2fs", seg.duration))
                                .foregroundStyle(.secondary)
                                .font(.caption)
                            Spacer()
                            if !seg.isFinalized {
                                Text("tentative")
                                    .font(.caption2)
                                    .padding(.horizontal, 6).padding(.vertical, 2)
                                    .background(.yellow.opacity(0.2),
                                                in: RoundedRectangle(cornerRadius: 4))
                            }
                        }
                        .padding(.vertical, 2)
                    }
                }
            }
        }
        .sheet(item: Binding(
            get: { editingSlot.map { SlotID(slot: $0) } },
            set: { editingSlot = $0?.slot }
        )) { id in
            VStack(spacing: 12) {
                Text("Rename slot \(id.slot)").font(.headline)
                TextField("Name", text: $editingName)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 240)
                HStack {
                    Button("Cancel") { editingSlot = nil }
                    Button("Save") {
                        rename(id.slot, editingName)
                        editingSlot = nil
                    }
                    .keyboardShortcut(.defaultAction)
                }
            }
            .padding()
        }
    }

    private func speakerLabel(slot: Int) -> String {
        speakers[slot].flatMap { $0 } ?? "Speaker \(slot)"
    }
}

private struct SlotID: Identifiable {
    let slot: Int
    var id: Int { slot }
}
