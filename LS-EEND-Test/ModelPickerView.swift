//
//  ModelPickerView.swift
//  LS-EEND-Test
//

import SwiftUI

struct ModelPickerView: View {
    @Binding var variant: LSEENDVariant
    @Binding var stepSize: LSEENDStepSize

    var body: some View {
        HStack(spacing: 12) {
            Picker("Variant", selection: $variant) {
                ForEach(LSEENDVariant.allCases, id: \.self) { v in
                    Text(v.subPath).tag(v)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 140)

            Picker("Step", selection: $stepSize) {
                ForEach(LSEENDStepSize.allCases, id: \.self) { s in
                    Text(s.suffix).tag(s)
                }
            }
            .labelsHidden()
            .frame(maxWidth: 100)
        }
    }
}
