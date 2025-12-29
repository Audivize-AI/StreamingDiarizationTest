//
//  Item.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 12/28/25.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
