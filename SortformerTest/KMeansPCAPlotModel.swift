import Foundation
import simd

struct KMeansPCAPlotPoint: Identifiable, Equatable, Sendable {
    let id: UUID
    let position: SIMD3<Float>
    let speakerID: Int
    let slot: Int
    let clusterID: Int
    let isInactive: Bool
}

struct KMeansPCAPlotModel: Equatable, Sendable {
    let points: [KMeansPCAPlotPoint]
    let updatedAt: Date

    init(
        points: [KMeansPCAPlotPoint],
        updatedAt: Date = Date()
    ) {
        self.points = points
        self.updatedAt = updatedAt
    }

    static let empty = KMeansPCAPlotModel(points: [], updatedAt: .distantPast)

    var isEmpty: Bool {
        points.isEmpty
    }

    var speakerIDs: [Int] {
        Array(Set(points.map(\.speakerID))).sorted()
    }

    var slots: [Int] {
        Array(Set(points.map(\.slot))).sorted()
    }

    var clusterCount: Int {
        Set(points.map(\.clusterID).filter { $0 >= 0 }).count
    }
}
