//
//  IndexUtils.swift
//  SortformerTest
//
//  Created by Benjamin Lee on 1/13/26.
//

import Foundation

public struct IndexUtils {
    public static func roundDiv(_ a: Int, _ b: Int) -> Int {
        return (a + b/2) / b
    }
    
    public static func ceilDiv(_ a: Int, _ b: Int) -> Int {
        return (a + b - 1) / b 
    }
    
    public static func nextMultiple(of step: Int, for value: Int) -> Int {
        return ceilDiv(value, step) * step
    }
}
