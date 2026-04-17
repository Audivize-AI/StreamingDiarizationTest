//
//  LSEENDTypes.swift
//  LS-EEND-Test
//
//  Created by Benjamin Lee on 4/16/26.
//

import Foundation
import CoreML


public typealias LSEENDVariant = ModelNames.LSEEND.Variant
public typealias LSEENDStepSize = ModelNames.LSEEND.StepSize


public struct LSEENDMetadata: Codable {
    public let chunkSize: Int
    public let frameDurationSeconds: Float
    public let maxSpeakers: Int
    public let sampleRate: Int
    public let maxNspks: Int
    public let hopLength: Int
    public let winLength: Int
    public let nMels: Int
    public let contextSize: Int
    public let subsampling: Int
    public let convDelay: Int
    public let nUnits: Int
    public let nHeads: Int
    public let encNLayers: Int
    public let decNLayers: Int
    public let convKernelSize: Int
    
    public var headDim: Int { nUnits / nHeads }
    public var featDim: Int { (2 * contextSize + 1) * nMels }
    public var nFFT: Int {
        1 << (Int.bitWidth - (winLength - 1).leadingZeroBitCount)
    }
}

public struct LSEENDState {
    var encRetKv: MLMultiArray
    var encRetScale: MLMultiArray
    var encConvCache: MLMultiArray
    var cnnWindow: MLMultiArray
    var decRetKv: MLMultiArray
    var decRetScale: MLMultiArray
    
    init(
        encRetKv: MLMultiArray,
        encRetScale: MLMultiArray,
        encConvCache: MLMultiArray,
        cnnWindow: MLMultiArray,
        decRetKv: MLMultiArray,
        decRetScale: MLMultiArray,
    ) {
        self.encRetKv = encRetKv
        self.encRetScale = encRetScale
        self.encConvCache = encConvCache
        self.cnnWindow = cnnWindow
        self.decRetKv = decRetKv
        self.decRetScale = decRetScale
    }
    
    init(from metadata: borrowing LSEENDMetadata) throws {
        let Lenc = NSNumber(value: metadata.encNLayers)
        let Ldec = NSNumber(value: metadata.decNLayers)
        let H = NSNumber(value: metadata.nHeads)
        let hd = NSNumber(value: metadata.headDim)
        let D = NSNumber(value: metadata.nUnits)
        let K = NSNumber(value: metadata.convKernelSize)
        let Kcnn = NSNumber(value: 2 * metadata.convDelay)
        let nSpk = NSNumber(value: metadata.maxNspks)
        
        self.encRetKv = try MLMultiArray(shape: [Lenc, 1, H, hd, hd], dataType: .float32)
        self.encRetScale = try MLMultiArray(shape: [Lenc, 1], dataType: .float32)
        self.encConvCache = try MLMultiArray(shape: [Lenc, 1, K, D], dataType: .float32)
        self.cnnWindow = try MLMultiArray(shape: [1, D, Kcnn], dataType: .float32)
        self.decRetKv = try MLMultiArray(shape: [Ldec, nSpk, H, hd, hd], dataType: .float32)
        self.decRetScale = try MLMultiArray(shape: [Ldec, 1], dataType: .float32)
    }
    
    func copy() -> LSEENDState {
        LSEENDState(
            encRetKv: encRetKv.copy() as! MLMultiArray,
            encRetScale: encRetScale.copy() as! MLMultiArray,
            encConvCache: encConvCache.copy() as! MLMultiArray,
            cnnWindow: cnnWindow.copy() as! MLMultiArray,
            decRetKv: decRetKv.copy() as! MLMultiArray,
            decRetScale: decRetScale.copy() as! MLMultiArray,
        )
    }
    
    func copy(to dst: inout LSEENDState) {
        memcpy(dst.encRetKv.dataPointer, encRetKv.dataPointer,
               encRetKv.count * MemoryLayout<Float>.stride)
        memcpy(dst.encRetScale.dataPointer, encRetScale.dataPointer,
               encRetScale.count * MemoryLayout<Float>.stride)
        memcpy(dst.encConvCache.dataPointer, encConvCache.dataPointer,
               encConvCache.count * MemoryLayout<Float>.stride)
        memcpy(dst.cnnWindow.dataPointer, cnnWindow.dataPointer,
               cnnWindow.count * MemoryLayout<Float>.stride)
        memcpy(dst.decRetKv.dataPointer, decRetKv.dataPointer,
               decRetKv.count * MemoryLayout<Float>.stride)
        memcpy(dst.decRetScale.dataPointer, decRetScale.dataPointer,
               decRetScale.count * MemoryLayout<Float>.stride)
    }
}

public struct LSEENDPreprocessorState {
    var audioOffset: Int = 0
    var audioDeque: [Float] = []
    
}

public enum LSEENDError: Error, LocalizedError {
    case initializationFailed(String)
    case inferenceFailed(String)
}
