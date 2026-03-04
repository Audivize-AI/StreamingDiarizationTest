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
    @State private var cameraNode: SCNNode = KMeansPCA3DView.makeCameraNode()
    @State private var sceneDebounceTask: Task<Void, Never>?
    @State private var pendingSceneSnapshot: KMeansPCAPlotModel?
    @State private var isSceneBuildInFlight = false
    @State private var sceneCancelGeneration: UInt64 = 0

    private static let defaultCameraPosition = SCNVector3(2.15, 1.45, 3.10)
    private static let ambientNodeName = "kmeans-ambient-light"
    private static let contentNodeName = "kmeans-content-node"
    private static let sceneBuildQueue = DispatchQueue(label: "SortformerTest.SceneKit.Build", qos: .utility)

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
                        pointOfView: cameraNode,
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
            ensureSceneScaffold()
            scheduleSceneUpdate(immediate: true)
        }
        .onChange(of: model.updatedAt) { _, _ in
            scheduleSceneUpdate(immediate: false)
        }
        .onDisappear {
            sceneDebounceTask?.cancel()
            sceneDebounceTask = nil
            pendingSceneSnapshot = nil
            sceneCancelGeneration &+= 1
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
            Text("Hue = speaker slot")
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

            Button(action: resetCamera) {
                Label("Reset Camera", systemImage: "camera.rotate")
                    .font(.system(size: 10, weight: .semibold, design: .rounded))
                    .foregroundStyle(KMeansPalette.subtitle)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .fill(Color.white.opacity(0.62))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 7, style: .continuous)
                            .stroke(KMeansPalette.border.opacity(0.22), lineWidth: 0.8)
                    )
            }
            .buttonStyle(.plain)
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

    private static func makeCameraNode() -> SCNNode {
        let node = SCNNode()
        let camera = SCNCamera()
        camera.zNear = 0.001
        camera.zFar = 100
        node.camera = camera
        node.position = defaultCameraPosition
        node.look(at: SCNVector3Zero)
        return node
    }

    private func attachCamera(to targetScene: SCNScene) {
        if cameraNode.parent !== targetScene.rootNode {
            cameraNode.removeFromParentNode()
            targetScene.rootNode.addChildNode(cameraNode)
        }
    }

    private func ensureSceneScaffold() {
        attachCamera(to: scene)
        if scene.rootNode.childNode(withName: Self.ambientNodeName, recursively: false) == nil {
            let ambient = SCNNode()
            ambient.name = Self.ambientNodeName
            ambient.light = SCNLight()
            ambient.light?.type = .ambient
            ambient.light?.intensity = 420
            scene.rootNode.addChildNode(ambient)
        }
    }

    private func applyContentNode(_ contentNode: SCNNode) {
        if let existing = scene.rootNode.childNode(withName: Self.contentNodeName, recursively: false) {
            existing.removeFromParentNode()
        }
        contentNode.name = Self.contentNodeName
        scene.rootNode.addChildNode(contentNode)
    }

    private func resetCamera() {
        cameraNode.simdTransform = matrix_identity_float4x4
        cameraNode.position = Self.defaultCameraPosition
        cameraNode.look(at: SCNVector3Zero)
    }

    private func scheduleSceneUpdate(immediate: Bool) {
        pendingSceneSnapshot = model
        if immediate {
            startNextSceneBuildIfNeeded()
            return
        }
        
        guard sceneDebounceTask == nil else {
            return
        }
        
        sceneDebounceTask = Task { @MainActor in
            defer { sceneDebounceTask = nil }
            try? await Task.sleep(nanoseconds: 120_000_000)
            guard !Task.isCancelled else { return }
            startNextSceneBuildIfNeeded()
        }
    }
    
    private func startNextSceneBuildIfNeeded() {
        guard !isSceneBuildInFlight,
              let next = pendingSceneSnapshot else {
            return
        }
        
        pendingSceneSnapshot = nil
        isSceneBuildInFlight = true
        let cancellationToken = sceneCancelGeneration
        
        Self.sceneBuildQueue.async {
            let builtContent = Self.makeContentNode(for: next)
            Task { @MainActor in
                if cancellationToken == sceneCancelGeneration {
                    ensureSceneScaffold()
                    applyContentNode(builtContent)
                }
                isSceneBuildInFlight = false
                startNextSceneBuildIfNeeded()
            }
        }
    }

    private static func makeContentNode(for snapshot: KMeansPCAPlotModel) -> SCNNode {
        let contentNode = SCNNode()
        addAxes(to: contentNode)
        addPoints(to: contentNode, points: snapshot.points)
        return contentNode
    }

    private static func addAxes(to parent: SCNNode) {
        let axisLength: CGFloat = 1.3
        let axisThickness: CGFloat = 0.006

        func material(_ color: NSColor) -> SCNMaterial {
            let material = SCNMaterial()
            material.diffuse.contents = color.withAlphaComponent(0.70)
            material.lightingModel = .constant
            return material
        }

        let xAxis = SCNBox(
            width: axisLength * 2,
            height: axisThickness,
            length: axisThickness,
            chamferRadius: axisThickness * 0.25
        )
        xAxis.materials = [material(.systemRed)]
        parent.addChildNode(SCNNode(geometry: xAxis))

        let yAxis = SCNBox(
            width: axisThickness,
            height: axisLength * 2,
            length: axisThickness,
            chamferRadius: axisThickness * 0.25
        )
        yAxis.materials = [material(.systemGreen)]
        parent.addChildNode(SCNNode(geometry: yAxis))

        let zAxis = SCNBox(
            width: axisThickness,
            height: axisThickness,
            length: axisLength * 2,
            chamferRadius: axisThickness * 0.25
        )
        zAxis.materials = [material(.systemBlue)]
        parent.addChildNode(SCNNode(geometry: zAxis))
    }

    private static func addPoints(to parent: SCNNode, points: [KMeansPCAPlotPoint]) {
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
            parent.addChildNode(node)
        }
    }

    private static func brightnessLookup(for points: [KMeansPCAPlotPoint]) -> [SpeakerClusterKey: CGFloat] {
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

    private static func pointColor(_ point: KMeansPCAPlotPoint, lookup: [SpeakerClusterKey: CGFloat]) -> NSColor {
        guard point.speakerID >= 0 else {
            return NSColor.systemGray.withAlphaComponent(0.90)
        }

        let paletteIndex = point.slot >= 0 ? point.slot : point.speakerID
        let baseColor: NSColor
        switch ((paletteIndex % 4) + 4) % 4 {
        case 0: baseColor = .systemRed
        case 1: baseColor = .systemGreen
        case 2: baseColor = .systemBlue
        default: baseColor = .systemOrange
        }

        let color = baseColor.usingColorSpace(.deviceRGB) ?? baseColor
        var hue: CGFloat = 0
        var saturation: CGFloat = 0.8
        var baseBrightness: CGFloat = 0.8
        var baseAlpha: CGFloat = 1
        _ = color.getHue(&hue, saturation: &saturation, brightness: &baseBrightness, alpha: &baseAlpha)
        let key = SpeakerClusterKey(speakerID: point.speakerID, clusterID: point.clusterID)
        let brightness = lookup[key] ?? (point.clusterID < 0 ? 0.30 : 0.86)
        let saturationAdjusted: CGFloat = point.isInactive ? max(0.16, saturation * 0.35) : max(0.58, saturation)
        let adjustedBrightness: CGFloat = point.isInactive ? brightness * 0.88 : brightness
        let alpha: CGFloat = point.isInactive ? 0.80 : 0.96
        return NSColor(calibratedHue: hue, saturation: saturationAdjusted, brightness: adjustedBrightness, alpha: alpha)
    }

    private static func geometryForSlot(_ slot: Int) -> SCNGeometry {
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

    private static func normalizedSlot(_ slot: Int) -> Int {
        ((slot % 6) + 6) % 6
    }
}

private extension SCNVector3 {
    init(_ v: SIMD3<Float>) {
        self.init(SCNFloat(v.x), SCNFloat(v.y), SCNFloat(v.z))
    }
}
