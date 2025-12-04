//
//  SpeechHistory.swift
//  DiarizationTest
//
//  Created by Benjamin Lee on 12/2/25.
//

import Foundation
import FluidAudio
import OrderedCollections

class DiarizationHistory {
    public private(set) var speakers: [String: DiarizedTrack] = [:]
    
    var segmentCount: Int { speakers.values.reduce(0) { $0 + $1.count } }
    var speakerCount: Int { speakers.count }

    init() {
        
    }
    
    public func upsert(segment newSegment: TimedSpeakerSegment) {
        if let speaker = speakers[newSegment.speakerId] {
            speaker.add(segment: newSegment)
        } else {
            speakers[newSegment.speakerId] = DiarizedTrack(from: newSegment)
        }
    }
    
    public func getHistory() -> [DiarizedSegment] {
        var indices: [Int] = Array(repeating: 0, count: speakerCount)
        var result: [DiarizedSegment] = []
        for i in 0..<segmentCount {
            var nextSegment: DiarizedSegment? = nil
            var nextIndex: Int = 0
            for (i, speaker) in speakers.values.enumerated() where indices[i] < speaker.count {
                let segment = speaker.segments[indices[i]]
                
                if nextSegment?.startsAfter(segment) != false {
                    nextSegment = segment
                    nextIndex = i
                }
            }
            
            guard let nextSegment else {
                print("WARNING: INCOMPLETE DIARIZATION HISTORY")
                return result
            }
            
            indices[nextIndex] += 1
            result.append(nextSegment)
        }
        
        return result
    }
}
