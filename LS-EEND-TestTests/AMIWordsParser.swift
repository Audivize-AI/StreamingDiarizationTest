//
//  AMIWordsParser.swift
//  LS-EEND-TestTests
//
//  Paper-style AMI reference: word-level forced-alignment turns from
//  `words/{meeting}.{A,B,C,D}.words.xml`, with consecutive same-speaker
//  words merged when the gap ≤ `mergeGap`. Kaldi's AMI recipe derives
//  its RTTM reference this way; the transcriber `segments.xml` turns
//  are looser (include silence inside monologues) and inflate DER.
//
//  Matches Python `coreml/ami_words_ref.py:load_word_turns`.
//

import Foundation
@testable import LS_EEND_Test

enum AMIWordsParser {

    /// Parse all four `{meeting}.{A|B|C|D}.words.xml` files in
    /// `wordsDir`. Each `<w>` becomes a speech region (skip
    /// `punc="true"`); consecutive same-speaker regions with gap ≤
    /// `mergeGap` are merged. Vocalsounds/nonvocalsounds/pauses are
    /// excluded (Kaldi default — keeps reference tight).
    static func parse(
        meeting: String,
        wordsDir: URL,
        mergeGap: Double = 0.5
    ) -> [SpeakerSegment] {
        var out: [SpeakerSegment] = []
        for ch in ["A", "B", "C", "D"] {
            let url = wordsDir.appendingPathComponent(
                "\(meeting).\(ch).words.xml", isDirectory: false)
            guard FileManager.default.fileExists(atPath: url.path),
                  let data = try? Data(contentsOf: url)
            else { continue }
            let delegate = _Delegate(label: ch)
            let parser = XMLParser(data: data)
            parser.delegate = delegate
            parser.shouldProcessNamespaces = false
            guard parser.parse() else { continue }

            // Merge consecutive same-speaker intervals with gap ≤ mergeGap.
            let sorted = delegate.intervals.sorted { $0.start < $1.start }
            var merged: [(start: Double, end: Double)] = []
            for iv in sorted {
                if let last = merged.last, iv.start - last.end <= mergeGap {
                    merged[merged.count - 1].end = max(last.end, iv.end)
                } else {
                    merged.append(iv)
                }
            }
            for iv in merged {
                out.append(SpeakerSegment(speaker: ch,
                                          start: iv.start, end: iv.end))
            }
        }
        return out
    }

    private final class _Delegate: NSObject, XMLParserDelegate {
        let label: String
        var intervals: [(start: Double, end: Double)] = []

        init(label: String) { self.label = label }

        func parser(_ parser: XMLParser, didStartElement elementName: String,
                    namespaceURI: String?, qualifiedName qName: String?,
                    attributes attributeDict: [String: String]) {
            let tag = elementName.split(separator: ":").last.map(String.init)
                ?? elementName
            // Kaldi / paper reference includes words only — vocalsounds,
            // nonvocalsounds, pauses, disfmarkers, gaps are excluded.
            guard tag == "w",
                  attributeDict["punc"] != "true",
                  let s = attributeDict["starttime"].flatMap(Double.init),
                  let e = attributeDict["endtime"].flatMap(Double.init),
                  e > s
            else { return }
            intervals.append((start: s, end: e))
        }
    }
}
