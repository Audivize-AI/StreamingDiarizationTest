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

    init() {}
    
    public func upsert(segment newSegment: TimedSpeakerSegment) {
        let id = findSpeaker(for: newSegment)
        if let speaker = speakers[id] {
            speaker.add(segment: newSegment)
        } else {
            let diarizedSegment = DiarizedSegment(from: newSegment, speakerId: id)
            
            // remove misassigned segments
            for speaker in speakers.values where speaker.id != id {
                if let segment = speaker.reassignLastSegment(ifMatching: newSegment, to: id) {
                    diarizedSegment.absorb(segment)
                }
            }
            
            speakers[id] = DiarizedTrack(from: diarizedSegment)
        }
    }
    
    public func getHistory() -> [DiarizedSegment] {
        var indices: [Int] = Array(repeating: 0, count: speakerCount)
        var result: [DiarizedSegment] = []
        for _ in (0..<segmentCount) {
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
    
    private func findSpeaker(for segment: TimedSpeakerSegment) -> String {
        var bestDistance = Config.embeddingThreshold
        var bestId: String? = nil
        
        for (id, speaker) in speakers {
            let distance = speaker.embedding.cosineDistance(to: segment.embedding)
            if distance < bestDistance {
                bestId = id
                bestDistance = distance
            }
        }
        
        return bestId ?? String(speakerCount + 1)
    }
}
