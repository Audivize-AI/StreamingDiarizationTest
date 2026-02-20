import Foundation

struct AHCDendrogramNodeModel: Identifiable, Hashable {
    let id: Int
    let matrixIndex: Int
    let leftChild: Int
    let rightChild: Int
    let speakerIndex: Int
    let count: Int
    let weight: Float
    let mergeDistance: Float
    let mustLink: Bool
    let exceedsLinkageThreshold: Bool

    var isLeaf: Bool {
        leftChild < 0 || rightChild < 0
    }
}

struct AHCDendrogramModel: Equatable {
    let rootIndex: Int
    let activeLeafCount: Int
    let nodes: [AHCDendrogramNodeModel]
    let updatedAt: Date

    init(rootIndex: Int, activeLeafCount: Int, nodes: [AHCDendrogramNodeModel], updatedAt: Date = Date()) {
        self.rootIndex = rootIndex
        self.activeLeafCount = activeLeafCount
        self.nodes = nodes
        self.updatedAt = updatedAt
    }

    static let empty = AHCDendrogramModel(rootIndex: -1, activeLeafCount: 0, nodes: [], updatedAt: .distantPast)

    var isEmpty: Bool {
        rootIndex < 0 || nodes.isEmpty
    }

    var maxMergeDistance: Float {
        nodes
            .filter { !$0.isLeaf && $0.mergeDistance.isFinite }
            .map(\.mergeDistance)
            .max() ?? 0
    }

    var nodesById: [Int: AHCDendrogramNodeModel] {
        Dictionary(uniqueKeysWithValues: nodes.map { ($0.id, $0) })
    }
}
