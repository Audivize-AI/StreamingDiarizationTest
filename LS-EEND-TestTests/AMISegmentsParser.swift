//
//  AMISegmentsParser.swift
//  LS-EEND-TestTests
//
//  Reads AMI NXT-style `{meeting}.{A,B,C,D}.segments.xml` files and emits
//  `[SpeakerSegment]` using the channel letter as the reference label —
//  byte-equivalent to what `LS-EEND/coreml/eval_ami_der.py` builds in
//  Python via pyannote's `Annotation`.
//

import Foundation
@testable import LS_EEND_Test

enum AMISegmentsParser {

    /// Parse all four `{meeting}.{A|B|C|D}.segments.xml` files in `segsDir`.
    /// Missing / malformed channels are silently skipped, matching the
    /// Python loader's `ET.ParseError` handling.
    static func parse(meeting: String, segsDir: URL) -> [SpeakerSegment] {
        var out: [SpeakerSegment] = []
        for ch in ["A", "B", "C", "D"] {
            let url = segsDir.appendingPathComponent("\(meeting).\(ch).segments.xml",
                                                    isDirectory: false)
            guard FileManager.default.fileExists(atPath: url.path),
                  let data = try? Data(contentsOf: url)
            else { continue }
            let delegate = _Delegate(label: ch)
            let parser = XMLParser(data: data)
            parser.delegate = delegate
            parser.shouldProcessNamespaces = false
            if parser.parse() {
                out.append(contentsOf: delegate.segments)
            }
        }
        return out
    }

    private final class _Delegate: NSObject, XMLParserDelegate {
        let label: String
        var segments: [SpeakerSegment] = []
        init(label: String) { self.label = label }

        func parser(_ parser: XMLParser, didStartElement elementName: String,
                    namespaceURI: String?, qualifiedName qName: String?,
                    attributes attributeDict: [String: String]) {
            // `elementName` has the namespace prefix stripped by
            // XMLParser (e.g. `nite:segment` → `segment`). Suffix-match to
            // handle both prefixed and default-namespace variants.
            guard elementName.hasSuffix("segment"),
                  let s = attributeDict["transcriber_start"].flatMap(Double.init),
                  let e = attributeDict["transcriber_end"].flatMap(Double.init),
                  e > s
            else { return }
            segments.append(SpeakerSegment(speaker: label, start: s, end: e))
        }
    }
}
