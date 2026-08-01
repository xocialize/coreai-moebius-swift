// CoreAIMoebiusInpaintPackage.swift
//
// The MLXEngine `imageInpaint` package over the CoreAI core — the SECOND backend behind the
// capability the MLX sibling (`moebius-inpaint`) already serves, registerable beside it under its
// own PackageID. 🔑 This product carries no MLX: MLXToolKit only, per the CoreAIRealESRGAN
// precedent (runtime-agnostic capability model).
//
// ⚠️ macOS 27+ (CoreAI.framework). The MLX sibling is the fallback below 27; a host injects this
// package through an external-registration seam when its deployment target allows.
//
// Request semantics, metaData keys (`seed`/`cfgScale`/`paste`), 512² preprocessing, and the
// blurred-mask paste mirror the MLX sibling EXACTLY — the two backends must be interchangeable
// behind a planner; divergence here is a defect, not a feature.

import CoreGraphics
import Foundation
import ImageIO
import MLXToolKit
import MoebiusCoreAI
import UniformTypeIdentifiers

@InferenceActor
public final class CoreAIMoebiusInpaintPackage: ModelPackage {
    public typealias Configuration = CoreAIMoebiusConfiguration

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7 weights: MIT (hustvl HF card; VAE MIT via PixelHacker/SDXL). C8 port code: MIT.
            // The .aimodel assets are derived works of the MIT weights (export scripts public in
            // moebius-m0/coreai — reproducible, not binaries of unknown provenance).
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "coreai-community/Moebius-CoreAI",
                                   revision: "main", tier: 3),
            requirements: RequirementsManifest(
                // PROVISIONAL split footprint from the CLI verify harness (unified memory during
                // the 19-step loop + VAE calls; assets 1.0 GB on disk, host buffers are small
                // because activations live inside the MPSGraph executables). ⚠️ Marked for in-app
                // re-baseline via ValidationHarness before any registry Eff claim — the fleet rule
                // is that smoke figures are provisional until phys_footprint is measured in-app.
                footprints: [
                    QuantFootprint(quant: .fp16, residentBytes: 1_200_000_000,
                                   peakActivationBytes: 2_000_000_000)
                ],
                // GPU is the measured backend. NOT .coreMLANE: full-model ANE compilation is
                // blocked on apple/coreai-models#138; declaring ANE here would promise a pool
                // the package cannot use (and the governor budgets pools by this field).
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 27, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                InpaintContract.descriptor(
                    name: "coreai-moebius-inpaint",
                    summary: "Diffusion object removal / inpainting (Moebius 0.22B, 512-native) "
                        + "on CoreAI static graphs — the same quality tier as moebius-inpaint, "
                        + "2.4× faster per UNet forward (49.8 vs 117.6 ms, M5 Max).",
                    modes: [])
            ])
    }

    private let configuration: Configuration
    private var pipeline: MoebiusPipeline_CoreAI?

    public nonisolated init(configuration: Configuration) { self.configuration = configuration }

    public func load() async throws {
        guard pipeline == nil else { return }
        // Engine-executed materialization (contract 1.24) has already run for dir-less configs;
        // reaching this guard with sources missing means no store was set, or a non-engine caller.
        guard configuration.missingWeightSources(storeRoot: configuration.modelsRootDirectory)
            .isEmpty,
            let dir = configuration.resolveWeightsDirectory()
        else { throw CoreAIMoebiusPackageError.weightsNotMaterialized(configuration.variant.repo) }

        let compute: MoebiusPipeline_CoreAI.Compute = switch configuration.compute {
        case .gpu: .gpu
        case .neuralEngine: .neuralEngine
        case .cpu: .cpu
        }
        let built = MoebiusPipeline_CoreAI(weightsDirectory: dir, compute: compute)
        built.numSteps = configuration.steps
        // Pay E5RT specialization at load (MAT semantics: preparation is visible, the first
        // inference is not secretly slow). Cold-cache ~40 s for the UNet graph, OS-cached after.
        try await built.prepare()
        pipeline = built
    }

    public func unload() async {
        // No MLX pool to flush — executables and their working sets are owned by the OS E5RT
        // cache; dropping the reference releases the process-side buffers.
        pipeline = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run(), before validation.
        try Task.checkCancellation()
        guard request.capability == .imageInpaint, let req = request as? InpaintRequest else {
            throw CoreAIMoebiusPackageError.unsupportedCapability(request.capability)
        }
        if pipeline == nil { try await load() }
        guard let pipe = pipeline else { throw CoreAIMoebiusPackageError.notLoaded }

        let sourceCG = try Self.decode(req.image)
        let maskCG = try Self.decode(req.mask)

        if let cfg = req.metaData["cfgScale"].flatMap(Self.doubleValue) {
            pipe.guidanceScale = Float(cfg)
        }
        let seed = req.metaData["seed"].flatMap(Self.intValue) ?? 0
        let paste = req.metaData["paste"].flatMap(Self.boolValue) ?? true

        // ── preprocess at the model's native 512², encode both planes in one b2 forward ───────
        RunProgress.report(.encode)
        let model = MoebiusImageIO_CoreAI.prepare(source: sourceCG, mask: maskCG)
        let (latents, maskedLatents) = try await pipe.encodeLatents(
            image: model.image, maskedImage: model.maskedImage)
        let noise = MoebiusImageIO_CoreAI.offsetNoise(
            count: MoebiusPipeline_CoreAI.latentCount, channels: 4,
            seed: UInt64(bitPattern: Int64(seed)), offset: pipe.noiseOffset)
        try Task.checkCancellation()   // pre-denoise seam: encode is a real chunk of work

        // ── 19 denoise steps + decode; per-step cancellation + progress ──────────────────────
        let inputs = MoebiusPipeline_CoreAI.Inputs(
            latents: latents, maskedLatents: maskedLatents,
            maskLatent: model.maskLatent, noise: noise)
        let decoded = try await pipe.run(inputs) { step, total in
            try Task.checkCancellation()
            RunProgress.report(.denoise, step: step, totalSteps: total)
        }

        // ── composite back at the ORIGINAL resolution ────────────────────────────────────────
        try Task.checkCancellation()   // pre-postprocess seam
        RunProgress.report(.postprocess)
        let output: CGImage
        if paste {
            output = try MoebiusImageIO_CoreAI.pasteAtOriginal(
                fill512: decoded, source: sourceCG, mask: maskCG)
        } else {
            output = try MoebiusImageIO_CoreAI.toCGImage(
                decoded, width: MoebiusImageIO_CoreAI.side, height: MoebiusImageIO_CoreAI.side)
        }
        let png = try Self.encodePNG(output)
        return InpaintResponse(image: Image(format: .png, data: png,
                                            width: output.width, height: output.height))
    }

    // MARK: metaData unwrap (MetaValue is a bare enum; the fleet pattern is local helpers)

    private nonisolated static func doubleValue(_ v: MetaValue) -> Double? {
        if case .double(let d) = v { return d }
        if case .int(let i) = v { return Double(i) }
        return nil
    }
    private nonisolated static func intValue(_ v: MetaValue) -> Int? {
        if case .int(let i) = v { return i }
        if case .double(let d) = v { return Int(d) }
        return nil
    }
    private nonisolated static func boolValue(_ v: MetaValue) -> Bool? {
        if case .bool(let b) = v { return b }
        return nil
    }

    // MARK: image codec

    private nonisolated static func decode(_ image: Image) throws -> CGImage {
        guard let src = CGImageSourceCreateWithData(image.data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { throw CoreAIMoebiusPackageError.decodeFailed }
        return cg
    }

    private nonisolated static func encodePNG(_ cg: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil)
        else { throw CoreAIMoebiusPackageError.encodeFailed }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw CoreAIMoebiusPackageError.encodeFailed }
        return data as Data
    }

    public enum CoreAIMoebiusPackageError: Error, CustomStringConvertible {
        case unsupportedCapability(Capability)
        case weightsNotMaterialized(String)
        case notLoaded
        case decodeFailed, encodeFailed

        public var description: String {
            switch self {
            case .unsupportedCapability(let c): return "coreai-moebius-inpaint cannot serve \(c)"
            case .weightsNotMaterialized(let r):
                return "assets for \(r) are not materialized (no store root and no modelDirectory)"
            case .notLoaded: return "package not loaded"
            case .decodeFailed: return "could not decode input image"
            case .encodeFailed: return "could not encode output image"
            }
        }
    }
}

public extension CoreAIMoebiusInpaintPackage {
    nonisolated static var registration: PackageRegistration {
        .of(CoreAIMoebiusInpaintPackage.self)
    }
}
