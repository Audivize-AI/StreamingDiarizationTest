import Foundation

public enum ANEMemoryError: Error, LocalizedError {
    case invalidAudioData
    case processingFailed(String)
    case memoryAllocationFailed
    case invalidArrayBounds

    public var errorDescription: String? {
        switch self {
        case .invalidAudioData:
            return "Invalid audio data provided."
        case .processingFailed(let message):
            return "Processing failed: \(message)"
        case .memoryAllocationFailed:
            return "Failed to allocate ANE-aligned memory."
        case .invalidArrayBounds:
            return "Array bounds exceeded for zero-copy view."
        }
    }
}
