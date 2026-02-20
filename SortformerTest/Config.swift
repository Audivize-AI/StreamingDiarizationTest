//
//  Config.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/13/26.
//

import Foundation
import AHCClustering

let globalConfig: SortformerConfig = .nvidiaHighLatency
//let globalConfig: SortformerConfig = .nvidiaLowLatency

/// Linkage policy used by the live dendrogram visualization.
/// Set to `.wardLinkage` only if you explicitly want Ward + log-scaled merge distances.
let dendrogramLinkagePolicy: LinkagePolicyType = .upgma
