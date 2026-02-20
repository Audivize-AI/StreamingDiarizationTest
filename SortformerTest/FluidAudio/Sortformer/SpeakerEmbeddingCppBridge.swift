import Foundation

@_cdecl("swiftSpeakerEmbeddingCreate")
public func speakerEmbeddingCreate(
    swiftPtrOut: UnsafeMutablePointer<UnsafeRawPointer>,
    idIn: UnsafePointer<UInt64>?,
    idOut: UnsafeMutablePointer<UInt64>?,
    vectorOut: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>
) {
    let embedding: SpeakerEmbedding
    if let idIn {
        let uuidTuple = idIn.withMemoryRebound(to: uuid_t.self, capacity: 1) { $0.pointee }
        embedding = SpeakerEmbedding(
            id: UUID(uuid: uuidTuple),
            unitEmbedding: Array(repeating: 0.0, count: EmbeddingConfig.embeddingFeatures),
            startFrame: 0,
            endFrame: 0,
        )
    } else {
        embedding = SpeakerEmbedding(
            unitEmbedding: Array(repeating: 0.0, count: EmbeddingConfig.embeddingFeatures),
            startFrame: 0,
            endFrame: 0,
        )
    }
    
    // Extract vector pointer
    vectorOut.pointee = embedding.embedding.baseAddress
    
    // Retain the swift object
    swiftPtrOut.pointee = UnsafeRawPointer(Unmanaged.passRetained(embedding).toOpaque())
    
    // Copy UUID
    if let idOut {
        withUnsafeBytes(of: embedding.id) { uuidBytes in
            let base = uuidBytes.baseAddress!.assumingMemoryBound(to: UInt64.self)
            idOut[0] = base[0]
            idOut[1] = base[1]
        }
    }
}

@_cdecl("swiftSpeakerEmbeddingLoad")
public func speakerEmbeddingLoad(
    _ embeddingPtr: UnsafeRawPointer,
    idOut: UnsafeMutablePointer<UInt64>,
    vectorOut: UnsafeMutablePointer<UnsafeMutablePointer<Float>?>,
    weightOut: UnsafeMutablePointer<Float>
) {
    // Retain the embedding
    let embedding = Unmanaged<SpeakerEmbedding>
        .fromOpaque(embeddingPtr)
        .retain()
        .takeUnretainedValue()
    
    // Copy UUID
    withUnsafeBytes(of: embedding.id) { uuidBytes in
        let base = uuidBytes.baseAddress!.assumingMemoryBound(to: UInt64.self)
        idOut[0] = base[0]
        idOut[1] = base[1]
    }
    
    // Extract vector pointer
    vectorOut.pointee = embedding.embedding.baseAddress
    
    // Copy length
    weightOut.pointee = Float(embedding.length) / 31.0
}

@_cdecl("swiftSpeakerEmbeddingRetain")
public func speakerEmbeddingRetain(_ embeddingPtr: UnsafeRawPointer) {
    _ = Unmanaged<SpeakerEmbedding>.fromOpaque(embeddingPtr).retain()
}

@_cdecl("swiftSpeakerEmbeddingRelease")
public func speakerEmbeddingRelease(_ ptr: UnsafeRawPointer) {
    Unmanaged<SpeakerEmbedding>.fromOpaque(ptr).release()
}
