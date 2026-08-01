import Foundation
import MLXToolKit

/// Which published CoreAI Moebius build to load. Places2 only, matching the MLX sibling.
public enum CoreAIMoebiusVariant: String, Codable, Sendable, CaseIterable {
    case places2

    /// One consolidated repo: UNet + VAE encoder/decoder + embedding table + card. Assets are
    /// too large to vendor as SPM resources (UNet 452 MB) — the WeightSourcing/materialization
    /// path is the design here, unlike the realesrgan donor's vendored kilobyte-scale models.
    public var repo: String {
        switch self {
        case .places2: return "coreai-community/Moebius-CoreAI"
        }
    }
}

/// Init-time configuration for `CoreAIMoebiusInpaintPackage` (C9).
public struct CoreAIMoebiusConfiguration: PackageConfiguration, ModelStorable, QuantConfigured,
    WeightSourcing {
    public var variant: CoreAIMoebiusVariant
    /// fp16 is the only dtype an .aimodel ships. MEASURED at 68.3 dB vs the shared PyTorch golden
    /// (rel 9.338e-04 — the same fp16 floor as the MLX sibling's 9.257e-04): BatchNorm constants
    /// are fp64-precomputed at export, so the subnormal running_var channels survive the cast.
    public var quant: Quant
    /// Denoise steps (reference default 20; `strength 0.99` makes the effective count 19).
    public var steps: Int
    /// gpu is the measured backend (49.8 ms/forward). neuralEngine is BLOCKED on
    /// apple/coreai-models#138 and silently falls back to GPU today; the knob exists so the
    /// package inherits the ANE by re-export when the OS fixes the compiler.
    public var compute: Compute
    /// Explicit weights directory (must contain all four assets) — honored before any store probe.
    public var modelDirectory: URL?
    /// Set by the engine from its `ModelStore`; excluded from `Codable`.
    public var modelsRootDirectory: URL?

    public enum Compute: String, Sendable, Codable {
        case gpu, neuralEngine, cpu
    }

    public init(variant: CoreAIMoebiusVariant = .places2, quant: Quant = .fp16, steps: Int = 20,
                compute: Compute = .gpu, modelDirectory: URL? = nil,
                modelsRootDirectory: URL? = nil) {
        self.variant = variant
        self.quant = quant
        self.steps = steps
        self.compute = compute
        self.modelDirectory = modelDirectory
        self.modelsRootDirectory = modelsRootDirectory
    }

    private enum CodingKeys: String, CodingKey { case variant, quant, steps, compute }

    /// The four files load() reads. `.aimodel` is a DIRECTORY — each is declared as a
    /// `<name>.aimodel/*` glob so the downloader materializes its three inner files.
    static let assetNames = [
        "moebius-unet-fp16-b2.aimodel",
        "moebius-vae-encoder-fp32-b2.aimodel",
        "moebius-vae-decoder-fp16-b1.aimodel",
    ]
    static let tableName = "embedding_table.npy"

    // MARK: WeightSourcing

    /// Every asset declared explicitly: a half-materialized snapshot (UNet without VAE) must read
    /// as missing rather than fail at load — the strict-probe lesson from TRELLIS.
    public var weightSources: [WeightSource] {
        [WeightSource(role: "main", repo: variant.repo,
                      matching: Self.assetNames.map { "\($0)/*" } + [Self.tableName])]
    }

    public func missingWeightSources(storeRoot: URL?) -> [WeightSource] {
        if let dir = modelDirectory, Self.assetsPresent(in: dir) { return [] }
        return defaultMissingWeightSources(storeRoot: storeRoot)
    }

    /// Directory the loader reads from: escape hatch, then the engine's flat store layout, then a
    /// hub-client snapshot. `nil` means materialization has not happened (load()'s offline guard).
    public func resolveWeightsDirectory() -> URL? {
        if let dir = modelDirectory, Self.assetsPresent(in: dir) { return dir }
        guard let root = modelsRootDirectory else { return nil }
        let store = ModelStore(root: root)
        if let flat = store.directory(for: variant.repo), Self.assetsPresent(in: flat) {
            return flat
        }
        if let snapshot = store.snapshotDirectory(for: variant.repo, revision: nil),
           Self.assetsPresent(in: snapshot) {
            return snapshot
        }
        return nil
    }

    /// Strict: all three .aimodel payloads (their main.mlirb) AND the embedding table.
    static func assetsPresent(in dir: URL) -> Bool {
        let fm = FileManager.default
        for asset in assetNames {
            let payload = dir.appendingPathComponent(asset).appendingPathComponent("main.mlirb")
            if !fm.fileExists(atPath: payload.path) { return false }
        }
        return fm.fileExists(atPath: dir.appendingPathComponent(tableName).path)
    }
}
