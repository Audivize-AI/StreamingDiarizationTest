//
//  Item.swift
//  Diarization Test Visualizer
//
//  Placeholder model to satisfy prior template references.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date

    init(timestamp: Date = .now) {
        self.timestamp = timestamp
    }
}
