// Contract-shape + conformance tests — everything here runs OFFLINE (no assets, no CoreAI
// runtime): manifest facts, configuration round-trip, MAT-1..5, CAN-1..3. The live pipeline
// gate is the CLI parity harness (Swift golden gate vs the shared PyTorch fixtures), not a test.
import Foundation
import MLXServeConformance
import MLXToolKit
import Testing

@testable import CoreAIMoebius

@Suite("Manifest and configuration")
struct ManifestTests {
    @Test func manifestIsImageInpaintAndPermissive() {
        let m = CoreAIMoebiusInpaintPackage.manifest
        #expect(m.license.weightLicense == .mit)                      // C7
        #expect(m.license.portCodeLicense == .mit)                    // C8
        #expect(m.requirements.os.minMacOS == SemanticVersion(major: 27, minor: 0, patch: 0))
        // GPU, deliberately NOT .coreMLANE: full-model ANE is blocked on coreai-models#138 and
        // declaring a pool the package cannot use would mislead the governor.
        #expect(m.requirements.requiredBackends == [.metalGPU])
        #expect(m.requirements.footprints.map(\.quant) == [.fp16])
        #expect((m.requirements.footprints[0].peakActivationBytes ?? 0) > 0,
                "split footprint: activation declared, not folded into resident")
        #expect(m.surfaces.count == 1)
        #expect(m.surfaces[0].capability == .imageInpaint)
        #expect(m.surfaces[0].name == "coreai-moebius-inpaint")       // own PackageID surface
    }

    @Test func registrationConstructs() {
        _ = CoreAIMoebiusInpaintPackage.registration
    }

    @Test func configurationCodableRoundTrip() throws {
        var c = CoreAIMoebiusConfiguration(steps: 12, compute: .neuralEngine)
        c.modelsRootDirectory = URL(fileURLWithPath: "/tmp")   // must NOT survive encoding
        let back = try JSONDecoder().decode(CoreAIMoebiusConfiguration.self,
                                            from: JSONEncoder().encode(c))
        #expect(back.steps == 12)
        #expect(back.compute == .neuralEngine)
        #expect(back.variant == .places2)
        #expect(back.quant == .fp16)
        #expect(back.modelsRootDirectory == nil,
                "store root is engine-injected state, never serialized configuration")
    }
}

@Suite("MAT-1..5 — WeightSourcing declarations")
struct MaterializationTests {
    static func materializedDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("coreai-moebius-mat-\(UUID().uuidString)")
        for asset in CoreAIMoebiusConfiguration.assetNames {
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent(asset), withIntermediateDirectories: true)
            try Data("stub".utf8).write(to: dir.appendingPathComponent(asset)
                .appendingPathComponent("main.mlirb"))
        }
        try Data("stub".utf8).write(
            to: dir.appendingPathComponent(CoreAIMoebiusConfiguration.tableName))
        return dir
    }

    @Test func matGate() throws {
        let dir = try Self.materializedDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let report = MaterializationConformance.check(
            freshConfiguration: CoreAIMoebiusConfiguration(),
            satisfiedConfiguration: CoreAIMoebiusConfiguration(modelDirectory: dir))
        #expect(report.passed, "\(report.summary)")
    }

    @Test func halfMaterializedReadsAsMissing() throws {
        // The strict-probe lesson: UNet present but VAE/table absent must read MISSING.
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("coreai-moebius-half-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("moebius-unet-fp16-b2.aimodel"),
            withIntermediateDirectories: true)
        try Data("stub".utf8).write(to: dir.appendingPathComponent("moebius-unet-fp16-b2.aimodel")
            .appendingPathComponent("main.mlirb"))
        defer { try? FileManager.default.removeItem(at: dir) }

        let c = CoreAIMoebiusConfiguration(modelDirectory: dir)
        #expect(!c.missingWeightSources(storeRoot: nil).isEmpty)
        #expect(c.resolveWeightsDirectory() == nil)
    }

    @Test func sourcesDeclareEveryAsset() {
        let sources = CoreAIMoebiusConfiguration().weightSources
        #expect(sources.count == 1)
        #expect(sources[0].repo == "coreai-community/Moebius-CoreAI")
        let matching = sources[0].matching ?? []
        #expect(matching.count == 4)   // 3 .aimodel globs + the table
        #expect(matching.contains("embedding_table.npy"))
        #expect(matching.contains("moebius-unet-fp16-b2.aimodel/*"))
    }
}

@Suite("CAN-1..3 — cancellation")
struct CancellationTests {
    @Test func preCancelledRunPropagates() async {
        let package = CoreAIMoebiusInpaintPackage(configuration: CoreAIMoebiusConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: InpaintRequest(image: Image(format: .png, data: Data()),
                                    mask: Image(format: .png, data: Data())))
        #expect(report.passed, "\(report.summary)")
    }

    @Test func cadenceDeclaration() {
        // Same posture as the MLX sibling: 19-step loop → per-denoise-step cadence, bracketed by
        // post-encode and pre-postprocess seams. Sub-second exemption would be dishonest here.
        let report = CancellationConformance.checkCadence(
            manifest: CoreAIMoebiusInpaintPackage.manifest,
            posture: .cadence([.init(phase: .denoise, unit: .step)]))
        #expect(report.passed, "\(report.summary)")
    }
}
