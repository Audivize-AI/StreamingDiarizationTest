import Foundation

/// Minimal Sortformer configuration shim used by the model registry helpers.
///
/// The app target no longer ships the full Sortformer runtime, but `ModelNames`
/// still exposes bundle-selection helpers that depend on a configuration type.
/// Keeping this small compatibility layer avoids breaking the existing model
/// registry API while the LS-EEND demo replaces the sample UI.
public enum SortformerConfig: String, CaseIterable, Sendable {
    case gradientDescentV2
    case gradientDescentV2_1
    case nvidiaLowLatencyV2
    case nvidiaLowLatencyV2_1
    case nvidiaHighLatencyV2
    case nvidiaHighLatencyV2_1

    public var modelVariant: ModelNames.Sortformer.Variant? {
        switch self {
        case .gradientDescentV2:
            return .gradientDecentV2
        case .gradientDescentV2_1:
            return .gradientDecentV2_1
        case .nvidiaLowLatencyV2:
            return .nvidiaLowLatencyV2
        case .nvidiaLowLatencyV2_1:
            return .nvidiaLowLatencyV2_1
        case .nvidiaHighLatencyV2:
            return .nvidiaHighLatencyV2
        case .nvidiaHighLatencyV2_1:
            return .nvidiaHighLatencyV2_1
        }
    }

    public func isCompatible(with other: SortformerConfig) -> Bool {
        self == other
    }
}
