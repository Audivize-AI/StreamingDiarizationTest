//
//  LSEENDState.swift
//  LS-EEND-Test
//
//  Created by Benjamin Lee on 4/21/26.
//

import Foundation
import CoreML
import Accelerate

public struct LSEENDState: ~Copyable {
    public var encRetKv: MLMultiArray
    public var encRetScale: MLMultiArray
    public var encConvCache: MLMultiArray
    public var cnnWindow: MLMultiArray
    public var decRetKv: MLMultiArray
    public var decRetScale: MLMultiArray
    
    public init(
        encRetKv: MLMultiArray,
        encRetScale: MLMultiArray,
        encConvCache: MLMultiArray,
        cnnWindow: MLMultiArray,
        decRetKv: MLMultiArray,
        decRetScale: MLMultiArray
    ) {
        self.encRetKv = encRetKv
        self.encRetScale = encRetScale
        self.encConvCache = encConvCache
        self.cnnWindow = cnnWindow
        self.decRetKv = decRetKv
        self.decRetScale = decRetScale
    }
    
    public init(from metadata: borrowing LSEENDMetadata) throws {
        let Lenc = NSNumber(value: metadata.encNLayers)
        let Ldec = NSNumber(value: metadata.decNLayers)
        let H = NSNumber(value: metadata.nHeads)
        let hd = NSNumber(value: metadata.headDim)
        let D = NSNumber(value: metadata.nUnits)
        let K = NSNumber(value: metadata.convKernelSize)
        let Kcnn = NSNumber(value: 2 * metadata.convDelay)
        let nSpk = NSNumber(value: metadata.maxNspks)
        
        func makeArray(shape: [NSNumber]) throws -> MLMultiArray {
            try ANEMemoryUtils.createAlignedArray(shape: shape, dataType: .float32)
        }
        
        self.init(
            encRetKv: try makeArray(shape: [Lenc, 1, H, hd, hd]),
            encRetScale: try makeArray(shape: [Lenc, 1]),
            encConvCache: try makeArray(shape: [Lenc, 1, K, D]),
            cnnWindow: try makeArray(shape: [1, D, Kcnn]),
            decRetKv: try makeArray(shape: [Ldec, nSpk, H, hd, hd]),
            decRetScale: try makeArray(shape: [Ldec, 1])
        )
        
        self.reset()
    }
    
    public func copy() -> LSEENDState {
        func clone(_ src: MLMultiArray) -> MLMultiArray {
            let dst = try! ANEMemoryUtils.createAlignedArray(
                shape: src.shape, dataType: src.dataType
            )
            ANEMemoryUtils.strideAwareCopy(from: src, to: dst)
            return dst
        }
        return LSEENDState(
            encRetKv: clone(encRetKv),
            encRetScale: clone(encRetScale),
            encConvCache: clone(encConvCache),
            cnnWindow: clone(cnnWindow),
            decRetKv: clone(decRetKv),
            decRetScale: clone(decRetScale)
        )
    }
    
    public func copy(to dst: borrowing LSEENDState) {
        ANEMemoryUtils.strideAwareCopy(from: encRetKv, to: dst.encRetKv)
        ANEMemoryUtils.strideAwareCopy(from: encRetScale, to: dst.encRetScale)
        ANEMemoryUtils.strideAwareCopy(from: encConvCache, to: dst.encConvCache)
        ANEMemoryUtils.strideAwareCopy(from: cnnWindow, to: dst.cnnWindow)
        ANEMemoryUtils.strideAwareCopy(from: decRetKv, to: dst.decRetKv)
        ANEMemoryUtils.strideAwareCopy(from: decRetScale, to: dst.decRetScale)
    }
    
    public func reset() {
        func clear(_ buffer: MLMultiArray) {
            buffer.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
                guard let base = buf.baseAddress else { return }
                memset(base, 0, buf.count * MemoryLayout<Float>.stride)
            }
        }
        
        clear(encRetKv)
        clear(encRetScale)
        clear(encConvCache)
        clear(cnnWindow)
        clear(decRetKv)
        clear(decRetScale)
    }
}

