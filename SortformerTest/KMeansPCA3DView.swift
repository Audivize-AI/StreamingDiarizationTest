import SwiftUI
import SceneKit

private enum KMeansPalette {
    static let panelTop = Color(red: 0.97, green: 0.98, blue: 0.99)
    static let panelBottom = Color(red: 0.91, green: 0.94, blue: 0.97)
    static let chartTop = Color(red: 0.99, green: 0.99, blue: 1.0)
    static let chartBottom = Color(red: 0.94, green: 0.96, blue: 0.99)
    static let border = Color(red: 0.33, green: 0.46, blue: 0.60)
    static let title = Color(red: 0.08, green: 0.18, blue: 0.30)
    static let subtitle = Color(red: 0.28, green: 0.38, blue: 0.50)
}

private struct SpeakerClusterKey: Hashable {
    let speakerID: Int
    let clusterID: Int
}

private struct PointStyleKey: Hashable {
    let slot: Int
    let speakerID: Int
    let clusterID: Int
}

struct KMeansPCA3DView: View {
    let model: KMeansPCAPlotModel

    @State private var scene: SCNScene = SCNScene()
    @State private var sceneUpdateTask: Task<Void, Never>?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header

            ZStack {
                RoundedRectangle(cornerRadius: 14, style: .continuous)
                    .fill(
                        LinearGradient(
                            colors: [KMeansPalette.chartTop, KMeansPalette.chartBottom],
                            startPoint: .top,
                            endPoint: .bottom
                        )
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 14, style: .continuous)
                            .stroke(KMeansPalette.border.opacity(0.22), lineWidth: 1)
                    )

                if model.isEmpty {
                    emptyState
                } else {
                    SceneView(
                        scene: scene,
                        pointOfView: nil,
                        options: [.allowsCameraControl, .autoenablesDefaultLighting]
                    )
                    .padding(6)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            footer
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 16, style: .continuous)
                .fill(
                    LinearGradient(
                        colors: [KMeansPalette.panelTop, KMeansPalette.panelBottom],
                        startPoint: .topLeading,
                        endPoint: .bottomTrailing
                    )
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 16, style: .continuous)
                        .stroke(KMeansPalette.border.opacity(0.34), lineWidth: 1.1)
                )
                .shadow(color: Color.black.opacity(0.10), radius: 8, x: 0, y: 3)
        )
        .onAppear {
            scheduleSceneUpdate(immediate: true)
        }
        .onChange(of: model.updatedAt) { _, _ in
            scheduleSceneUpdate(immediate: false)
        }
        .onDisappear {
            sceneUpdateTask?.cancel()
        }
    }

    private var header: some View {
        HStack(alignment: .top, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text("K-means PCA")
                    .font(.system(size: 14, weight: .bold, design: .rounded))
                    .foregroundStyle(KMeansPalette.title)
                Text("3D interactive embedding space")
                    .font(.system(size: 11, weight: .medium, design: .monospaced))
                    .foregroundStyle(KMeansPalette.subtitle)
            }

            Spacer(minLength: 6)

            metricPill(label: "Points", value: "\(model.points.count)", tint: KMeansPalette.title)
            metricPill(label: "Clusters", value: "\(model.clusterCount)", tint: KMeansPalette.subtitle)
        }
    }

    private var footer: some View {
        HStack(spacing: 8) {
            Text("Hue = speaker ID")
                .font(.system(size: 10, weight: .semibold, design: .rounded))
                .foregroundStyle(KMeansPalette.subtitle)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(Color.white.opacity(0.56))
                )

            Text("Brightness = cluster ID")
                .font(.system(size: 10, weight: .semibold, design: .rounded))
                .foregroundStyle(KMeansPalette.subtitle)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    RoundedRectangle(cornerRadius: 7, style: .continuous)
                        .fill(Color.white.opacity(0.56))
                )

            Spacer(minLength: 4)

            Text("Drag rotate, scroll zoom, secondary drag pan")
                .font(.system(size: 10, weight: .medium, design: .rounded))
                .foregroundStyle(KMeansPalette.subtitle)
        }
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "point.3.connected.trianglepath.dotted")
                .font(.system(size: 24, weight: .semibold))
                .foregroundStyle(KMeansPalette.subtitle.opacity(0.85))
            Text("Plot will appear once embeddings are extracted")
                .font(.system(size: 11, weight: .medium, design: .rounded))
                .foregroundStyle(KMeansPalette.subtitle)
        }
    }

    private func metricPill(label: String, value: String, tint: Color) -> some View {
        VStack(spacing: 1) {
            Text(label.uppercased())
                .font(.system(size: 8, weight: .bold, design: .monospaced))
                .foregroundStyle(tint.opacity(0.85))
            Text(value)
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(KMeansPalette.title)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .fill(Color.white.opacity(0.60))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 7, style: .continuous)
                .stroke(tint.opacity(0.28), lineWidth: 0.9)
        )
    }

    private func scheduleSceneUpdate(immediate: Bool) {
        sceneUpdateTask?.cancel()
        let snapshot = model
        sceneUpdateTask = Task { @MainActor in
            if !immediate {
                try? await Task.sleep(nanoseconds: 120_000_000)
            }
            guard !Task.isCancelled else { return }
            scene = makeScene(for: snapshot)
        }
    }

    private func makeScene(for snapshot: KMeansPCAPlotModel) -> SCNScene {
        let newScene = SCNScene()

        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        cameraNode.camera?.zNear = 0.001
        cameraNode.camera?.zFar = 100
        cameraNode.position = SCNVector3(0, 0, 3.0)
        newScene.rootNode.addChildNode(cameraNode)

        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 420
        newScene.rootNode.addChildNode(ambient)

        addAxes(to: newScene)
        addPoints(to: newScene, points: snapshot.points)

        return newScene
    }

    private func addAxes(to scene: SCNScene) {
        let axisLength: CGFloat = 1.3
        let axisRadius: CGFloat = 0.003

        func axisNode(from start: SCNVector3, to end: SCNVector3, color: NSColor) -> SCNNode {
            let vector = end - start
            let distance = vector.length()

            let cylinder = SCNCylinder(radius: axisRadius, height: distance)
            let material = SCNMaterial()
            material.diffuse.contents = color.withAlphaComponent(0.62)
            material.lightingModel = .constant
            cylinder.materials = [material]

            let node = SCNNode(geometry: cylinder)
            node.position = (start + end) / 2
            node.eulerAngles = vector.eulerAngles()
            node.castsShadow = false
            return node
        }

        scene.rootNode.addChildNode(
            axisNode(
                from: SCNVector3(-axisLength, 0, 0),
                to: SCNVector3(axisLength, 0, 0),
                color: NSColor.systemRed
            )
        )
        scene.rootNode.addChildNode(
            axisNode(
                from: SCNVector3(0, -axisLength, 0),
                to: SCNVector3(0, axisLength, 0),
                color: NSColor.systemGreen
            )
        )
        scene.rootNode.addChildNode(
            axisNode(
                from: SCNVector3(0, 0, -axisLength),
                to: SCNVector3(0, 0, axisLength),
                color: NSColor.systemBlue
            )
        )
    }

    private func addPoints(to scene: SCNScene, points: [KMeansPCAPlotPoint]) {
        let brightnessLookup = brightnessLookup(for: points)
        var geometryCache: [PointStyleKey: SCNGeometry] = [:]

        for point in points {
            let style = PointStyleKey(slot: normalizedSlot(point.slot), speakerID: point.speakerID, clusterID: point.clusterID)
            let geometry: SCNGeometry
            if let cached = geometryCache[style] {
                geometry = cached
            } else {
                let created = geometryForSlot(style.slot)
                let material = SCNMaterial()
                material.diffuse.contents = pointColor(point, lookup: brightnessLookup)
                material.specular.contents = NSColor.white.withAlphaComponent(0.08)
                material.lightingModel = .lambert
                created.materials = [material]
                geometryCache[style] = created
                geometry = created
            }

            let node = SCNNode(geometry: geometry)
            node.position = SCNVector3(point.position)
            node.castsShadow = false
            scene.rootNode.addChildNode(node)
        }
    }

    private func brightnessLookup(for points: [KMeansPCAPlotPoint]) -> [SpeakerClusterKey: CGFloat] {
        var clusterIDsBySpeaker: [Int: Set<Int>] = [:]
        for point in points where point.clusterID >= 0 {
            clusterIDsBySpeaker[point.speakerID, default: []].insert(point.clusterID)
        }

        var result: [SpeakerClusterKey: CGFloat] = [:]
        for (speakerID, clusterIDs) in clusterIDsBySpeaker {
            let sorted = clusterIDs.sorted()
            guard !sorted.isEmpty else { continue }

            if sorted.count == 1, let onlyCluster = sorted.first {
                result[SpeakerClusterKey(speakerID: speakerID, clusterID: onlyCluster)] = 0.88
                continue
            }

            let minBrightness: CGFloat = 0.42
            let maxBrightness: CGFloat = 0.96
            let denom = CGFloat(sorted.count - 1)
            for (index, clusterID) in sorted.enumerated() {
                let t = CGFloat(index) / max(denom, 1)
                let brightness = minBrightness + (maxBrightness - minBrightness) * t
                result[SpeakerClusterKey(speakerID: speakerID, clusterID: clusterID)] = brightness
            }
        }

        return result
    }

    private func pointColor(_ point: KMeansPCAPlotPoint, lookup: [SpeakerClusterKey: CGFloat]) -> NSColor {
        guard point.speakerID >= 0 else {
            return NSColor.systemGray.withAlphaComponent(0.90)
        }

        let hue = CGFloat((point.speakerID * 53).quotientAndRemainder(dividingBy: 360).remainder) / 360.0
        let key = SpeakerClusterKey(speakerID: point.speakerID, clusterID: point.clusterID)
        let brightness = lookup[key] ?? (point.clusterID < 0 ? 0.30 : 0.86)
        return NSColor(calibratedHue: hue, saturation: 0.82, brightness: brightness, alpha: 0.96)
    }

    private func geometryForSlot(_ slot: Int) -> SCNGeometry {
        let size: CGFloat = 0.026

        switch normalizedSlot(slot) {
        case 0:
            let sphere = SCNSphere(radius: size * 0.58)
            sphere.segmentCount = 8
            return sphere
        case 1:
            return SCNBox(width: size, height: size, length: size, chamferRadius: size * 0.14)
        case 2:
            return SCNPyramid(width: size, height: size * 1.15, length: size)
        case 3:
            let cone = SCNCone(topRadius: 0, bottomRadius: size * 0.50, height: size * 1.28)
            cone.radialSegmentCount = 8
            return cone
        case 4:
            let cylinder = SCNCylinder(radius: size * 0.46, height: size * 1.15)
            cylinder.radialSegmentCount = 8
            return cylinder
        default:
            let torus = SCNTorus(ringRadius: size * 0.43, pipeRadius: size * 0.17)
            torus.ringSegmentCount = 10
            torus.pipeSegmentCount = 6
            return torus
        }
    }

    private func normalizedSlot(_ slot: Int) -> Int {
        ((slot % 6) + 6) % 6
    }
}

private extension SCNVector3 {
    init(_ v: SIMD3<Float>) {
        self.init(SCNFloat(v.x), SCNFloat(v.y), SCNFloat(v.z))
    }

    static func +(lhs: SCNVector3, rhs: SCNVector3) -> SCNVector3 {
        SCNVector3(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z)
    }

    static func -(lhs: SCNVector3, rhs: SCNVector3) -> SCNVector3 {
        SCNVector3(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z)
    }

    static func /(lhs: SCNVector3, rhs: CGFloat) -> SCNVector3 {
        let divisor = SCNFloat(rhs)
        return SCNVector3(lhs.x / divisor, lhs.y / divisor, lhs.z / divisor)
    }

    func length() -> CGFloat {
        sqrt(x * x + y * y + z * z)
    }

    func eulerAngles() -> SCNVector3 {
        let yaw = atan2(x, z)
        let pitch = atan2(y, sqrt(x * x + z * z))
        return SCNVector3(-pitch, yaw, 0)
    }
}
