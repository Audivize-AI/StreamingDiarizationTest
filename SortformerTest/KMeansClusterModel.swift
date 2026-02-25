import Foundation

struct KMeansEmbeddingPointModel: Identifiable, Hashable {
    let id: UUID
    let slot: Int
    let clusterIndex: Int
    let isTentative: Bool
    let isOutlier: Bool
    let distanceToCentroid: Float
    let x: Float
    let y: Float
}

struct KMeansCentroidModel: Identifiable, Hashable {
    enum Source: String, Hashable {
        case finalized
        case tentative
    }

    let slot: Int
    let clusterIndex: Int
    let source: Source
    let memberCount: Int
    let averageDistance: Float
    let x: Float
    let y: Float

    var id: String {
        "\(slot)-\(clusterIndex)-\(source.rawValue)"
    }
}

struct KMeansClusterPlotModel: Equatable {
    let points: [KMeansEmbeddingPointModel]
    let finalizedCentroids: [KMeansCentroidModel]
    let tentativeCentroids: [KMeansCentroidModel]
    let updatedAt: Date

    init(
        points: [KMeansEmbeddingPointModel],
        finalizedCentroids: [KMeansCentroidModel],
        tentativeCentroids: [KMeansCentroidModel],
        updatedAt: Date = Date()
    ) {
        self.points = points
        self.finalizedCentroids = finalizedCentroids
        self.tentativeCentroids = tentativeCentroids
        self.updatedAt = updatedAt
    }

    static let empty = KMeansClusterPlotModel(
        points: [],
        finalizedCentroids: [],
        tentativeCentroids: [],
        updatedAt: .distantPast
    )

    var isEmpty: Bool {
        points.isEmpty && finalizedCentroids.isEmpty && tentativeCentroids.isEmpty
    }

    var slotIndices: [Int] {
        let pointSlots = points.map(\.slot)
        let finalizedSlots = finalizedCentroids.map(\.slot)
        let tentativeSlots = tentativeCentroids.map(\.slot)
        return Array(Set(pointSlots + finalizedSlots + tentativeSlots)).sorted()
    }

    var outlierCount: Int {
        points.filter(\.isOutlier).count
    }
}
