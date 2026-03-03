import Foundation

//let globalConfig: SortformerConfig = .default
//let globalConfig: SortformerConfig = .nvidiaHighLatency
let globalConfig: SortformerConfig = .nvidiaLowLatency

let globalTimelineConfig: SortformerTimelineConfig = .default(for: globalConfig)

let globalEmbeddingConfig: EmbeddingConfig = .large3_04s
