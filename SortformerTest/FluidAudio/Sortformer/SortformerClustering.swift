//
//  SortformerClustering.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/21/26.
//
    
import Foundation
import Accelerate

//
//public class SortformerClustering {
//    
//}
//
//public class SpeakerID {
//    public let id: Int
//    private static var nextID: Int = 0
//    
//    public init() {
//        id = SpeakerID.nextID
//        SpeakerID.nextID += 1
//    }
//}
//
//public class GallaryCluster: Identifiable, Hashable {
//    public let id: UUID
//    public var leftChild: DendrogramNode?
//    public var rightChild: DendrogramNode?
//    public var parent: DendrogramNode?
//    
//    public var speakerID: SpeakerID?
//    public private(set) var count: Int
//    public var constraints: Set<DendrogramNode> = []
//    public var embedding: [Float]
//    public private(set) var embeddingMagnitude: Float
//    
//    public static let embeddingDims: Int = 192
//    public static let embeddingLength: UInt = UInt(embeddingDims)
//    
//    public init(
//        leftChild: DendrogramNode,
//        rightChild: DendrogramNode,
//        speakerID: SpeakerID
//    ) {
//        self.id = UUID()
//        self.leftChild = leftChild
//        self.rightChild = rightChild
//        self.count = leftChild.count + rightChild.count
//        self.constraints = leftChild.constraints.union(rightChild.constraints)
//        
//        var leftWeight = Float(leftChild.count)
//        var rightWeight = Float(rightChild.count)
//        self.embedding = Array(repeating: 0, count: Self.embeddingDims)
//        vDSP_vsmsma(
//            leftChild.embedding, 1,
//            &leftWeight,
//            rightChild.embedding, 1,
//            &rightWeight,
//            &self.embedding, 1,
//            Self.embeddingLength
//        )
//        var scale = 1 / sqrt(vDSP.sumOfSquares(self.embedding))
//        embeddingMagnitude
//        leftChild.parent = self
//        rightChild.parent = self
//        leftChild.speakerID = speakerID
//        rightChild.speakerID = speakerID
//    }
//    
//    public static func == (lhs: DendrogramNode, rhs: DendrogramNode) -> Bool {
//        lhs.id == rhs.id
//    }
//    
//    public func hash(into hasher: inout Hasher) {
//        hasher.combine(id)
//    }
//}
