import Foundation
import CoreML
import Accelerate

public struct TitaNetConfig {
    let inputLength: Int = 31 /// Number of input Sortformer frames
    let minRequiredFrames: Int = 13 /// Minimum number of input Sortformer frames
    let frameDuration: Float = 0.08
    let subsamplingFactor: Int = 8
    let melFeatures: Int = 80
    let melStride: Int = 160
    let melWindow: Int = 400
    let melPadTo: Int = 16
    var melLength: Int { inputLength * subsamplingFactor }
    var paddedMelLength: Int { ((melLength - 1) / melPadTo + 1) * melPadTo }
    var audioSignalLength: Int { melLength * melStride }
    var inputDuration: Float { Float(inputLength) * frameDuration }
}

public struct TitaNetEmbeddingExtractor {
    public let config: TitaNetConfig
    public let model: MLModel
    public let preprocessor: NeMoMelSpectrogram
    private let memoryOptimizer: ANEMemoryOptimizer
    private let speechArray: MLMultiArray
    private let lengthArray: MLMultiArray
    private let sampleRate: Float = 16_000
    
    init(config: TitaNetConfig) throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        
        self.config = config
        self.model = try TitaNet_small_2_48s(configuration: configuration).model
        self.preprocessor = NeMoMelSpectrogram(nMels: 80, padTo: 16)
        self.memoryOptimizer = ANEMemoryOptimizer()
        self.speechArray = try memoryOptimizer.createAlignedArray(
            shape: [1,
                    NSNumber(value: config.melFeatures),
                    NSNumber(value: config.paddedMelLength)],
            dataType: .float32
        )
        self.lengthArray = try memoryOptimizer.createAlignedArray(
            shape: [1],
            dataType: .int32
        )
    }
    
    public func getEmbedding<C>(
        from audioSignal: C,
        length: Int? = nil
    ) throws -> [Float] where C: AccelerateBuffer & Collection, C.Element == Float, C.Index == Int {
        let length = length ?? audioSignal.count
        // Preprocess audio
        let (mels, melLength, _) = preprocessor.computeFlat(audio: audioSignal)
        
        return try getEmbedding(mels: mels, melLength: melLength)
    }
    
    public func getEmbedding<C>(
        mels: C,
        melLength: Int
    ) throws -> [Float] where C: Collection, C.Element == Float {
        // Ensure input fits
        guard melLength > 0 else {
            throw TitaNetError.invalidAudioInput("Empty audio input")
        }
        
        guard melLength <= config.paddedMelLength else {
            fatalError("Audio too long for this model (\(melLength) > \(config.paddedMelLength))")
        }
        
        // Copy inputs to MLMultiArrays
        memoryOptimizer.optimizedCopy(
            from: mels,
            to: speechArray,
            pad: true
        )
        
        lengthArray[0] = NSNumber(value: Int32(melLength))
        
        // Build input
        let input = try MLDictionaryFeatureProvider(dictionary: [
            "processed_signal": MLFeatureValue(multiArray: speechArray),
            "length": MLFeatureValue(multiArray: lengthArray)
        ])
        
        // Get prediction
        let output = try model.prediction(from: input)
        
        guard let embedding = output.featureValue(for: "embedding")?.shapedArrayValue(of: Float.self)?.scalars else {
            throw TitaNetError.predictionFailed("Missing embedding output")
        }
        
        return embedding
    }
    

    
    
}


public enum TitaNetError: Error, LocalizedError {
    case invalidAudioInput(String)
    case predictionFailed(String)
}
