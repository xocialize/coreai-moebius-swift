// MoebiusPipeline_CoreAI.swift
//
// Role: the Moebius 0.22B diffusion-inpainting pipeline on CoreAI — three static-shape .aimodel
// graphs (UNet b2-CFG, VAE encoder b2, VAE decoder b1) driven by host-side DDIM. A transcription
// of the MLX sibling's `MoebiusPipeline` (itself gated line-by-line against the reference
// `RemovalSDXLPipeline_BatchMode`), with `[Float]` host math in place of MLXArray:
//
//     encode([image, masked]) ─→ [noisy(4) | mask(1) | masked(4)] → UNet ─CFG→ DDIM ─×19→ decode
//
// The three reference quirks, restated because each one yields a plausible-but-wrong image:
//   1. NOT from pure noise: `strength 0.99` noises the CLEAN latents at t=900; 20 steps run 19.
//   2. CFG is batch-doubled through ONE forward, [uncond, cond] = table rows [10..19] / [0..9].
//   3. Channel order is noisy(4) + mask(1) + masked(4) — the mask sits in the MIDDLE.
//
// ⚠️ Compute default is .gpu, stated plainly: full-model ANE compilation is blocked on an
// ANECCompiler bug (apple/coreai-models#138 — two 64²-level transformers per graph break the
// input-channel-split pass). Requesting .neuralEngine today compiles-fails and SILENTLY falls
// back to GPU (measured); the case exists so the package inherits the ANE when the OS fixes
// #138, but no caller should believe it means ANE execution until the GPU-idle signature says so.
//
// Assets are exported by `moebius-m0/coreai/export_unet.py` / `export_vae.py` — the fp16 UNet
// carries the rank-safe λ rewrites + fp64-precomputed BatchNorm constants (68.3 dB vs the shared
// PyTorch golden; 49.8 ms/forward on M5 Max, 2.4× the MLX sibling).

import CoreAI
import Foundation

/// Moebius on CoreAI — fp16, static 512²/64² shapes, 19-step DDIM CFG-2 inpainting.
public final class MoebiusPipeline_CoreAI: @unchecked Sendable {

    public enum Compute: String, Sendable {
        case gpu            // the measured backend (49.8 ms/forward)
        case neuralEngine   // blocked on coreai-models#138; silently falls back to GPU today
        case cpu

        var options: SpecializationOptions {
            switch self {
            case .gpu:          return SpecializationOptions(preferredComputeUnitKind: .gpu)
            case .neuralEngine: return SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
            case .cpu:          return SpecializationOptions(preferredComputeUnitKind: .cpu)
            }
        }
    }

    // Fixed by the static exports (and by the checkpoint — rel_pos_emb is spatially baked).
    public static let side = 512
    public static let latentSide = 64
    public static let latentChannels = 4
    public static let contextTokens = 10
    public static let contextDim = 3072
    public static let numEmbeddings = 20
    public static let scalingFactor: Float = 0.13025

    public static let unetAsset = "moebius-unet-fp16-b2.aimodel"
    // Encoder is fp32: at fp16 the SD-VAE encoder reads 45.6 dB (investigate band) and NaNs on
    // the CPU lane — activation range, the classic sdxl-vae-fp16-fix problem — and mixed
    // precision does not lower in a CoreAI graph (measured). One encode per image; fp32's ~2×
    // cost is invisible next to the 19-step loop. Decoder measured 68.5 dB at fp16 [PASS].
    public static let encoderAsset = "moebius-vae-encoder-fp32-b2.aimodel"
    public static let decoderAsset = "moebius-vae-decoder-fp16-b1.aimodel"
    public static let embeddingAsset = "embedding_table.npy"

    public var numSteps: Int = 20
    public var strength: Float = 0.99
    public var guidanceScale: Float = 2.5      // the reference CLI default
    public var noiseOffset: Float = 0.0357

    public let weightsDirectory: URL
    public let compute: Compute

    private var unet: InferenceFunction?
    private var encoder: InferenceFunction?
    private var decoder: InferenceFunction?
    private var context: [Float16] = []        // [2·10·3072], precomputed [uncond, cond]
    private let loadLock = NSLock()

    public init(weightsDirectory: URL, compute: Compute = .gpu) {
        self.weightsDirectory = weightsDirectory
        self.compute = compute
    }

    // MARK: - Loading

    /// Load + specialize all three graphs and precompute the conditioning constant. First call
    /// per machine pays E5RT specialization (~40 s for the UNet, OS-cached after); MAT semantics
    /// say that cost belongs here, never inside the first user-visible run.
    public func prepare() async throws {
        if loadLock.withLock({ unet != nil }) { return }

        let fnU = try await loadFunction(Self.unetAsset)
        let fnE = try await loadFunction(Self.encoderAsset)
        let fnD = try await loadFunction(Self.decoderAsset)

        // Conditioning: embedding_table.npy [20, 3072] f32 → rows [10..19 | 0..9] as fp16.
        let table = try Self.readNPY(
            weightsDirectory.appendingPathComponent(Self.embeddingAsset),
            expectCount: Self.numEmbeddings * Self.contextDim)
        var ctx = [Float16]()
        ctx.reserveCapacity(2 * Self.contextTokens * Self.contextDim)
        let half = Self.numEmbeddings / 2
        for row in half ..< Self.numEmbeddings {          // uncond half
            ctx.append(contentsOf: table[(row * Self.contextDim) ..< ((row + 1) * Self.contextDim)]
                .map(Float16.init))
        }
        for row in 0 ..< half {                           // cond half
            ctx.append(contentsOf: table[(row * Self.contextDim) ..< ((row + 1) * Self.contextDim)]
                .map(Float16.init))
        }

        loadLock.withLock {
            unet = fnU
            encoder = fnE
            decoder = fnD
            context = ctx
        }
    }

    private func loadFunction(_ asset: String) async throws -> InferenceFunction {
        let url = weightsDirectory.appendingPathComponent(asset)
        guard FileManager.default.fileExists(atPath: url.path) else {
            throw MoebiusCoreAIError.assetMissing(url.path)
        }
        let model = try await AIModel(contentsOf: url, options: compute.options)
        guard let fn = try model.loadFunction(named: "main") else {
            throw MoebiusCoreAIError.functionMissing(asset)
        }
        return fn
    }

    // MARK: - Pipeline

    public struct Inputs {
        public var latents: [Float]         // [4·64·64], already × scalingFactor
        public var maskedLatents: [Float]   // [4·64·64]
        public var maskLatent: [Float]      // [64·64], 1 = remove
        public var noise: [Float]           // [4·64·64], offset already folded in

        public init(latents: [Float], maskedLatents: [Float],
                    maskLatent: [Float], noise: [Float]) {
            self.latents = latents
            self.maskedLatents = maskedLatents
            self.maskLatent = maskLatent
            self.noise = noise
        }
    }

    /// VAE-encode `[image, masked]` (batch 2, ONE forward — the export was shaped for exactly
    /// this call) → posterior means × scalingFactor.
    public func encodeLatents(image: [Float], maskedImage: [Float]) async throws
        -> (latents: [Float], maskedLatents: [Float]) {
        guard let encoder = loadLock.withLock({ self.encoder }) else {
            throw MoebiusCoreAIError.functionMissing(Self.encoderAsset)
        }
        let n = 3 * Self.side * Self.side
        // fp32 in/out — the encoder asset is fp32 (see encoderAsset note).
        var input = NDArray(shape: [2, 3, Self.side, Self.side], scalarType: .float32)
        input.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0 ..< n { ptr[i] = image[i] }
            for i in 0 ..< n { ptr[n + i] = maskedImage[i] }
        }
        var outputs = try await encoder.run(inputs: ["x": input])
        let mean = try Self.floats(from: &outputs, count: 2 * Self.latentCount, fp32: true)
        let sf = Self.scalingFactor
        let latents = mean[0 ..< Self.latentCount].map { $0 * sf }
        let masked = mean[Self.latentCount ..< (2 * Self.latentCount)].map { $0 * sf }
        return (Array(latents), Array(masked))
    }

    /// Run the 19-step denoise loop and decode. Returns the image in `[0,1]` CHW.
    ///
    /// `onStep` fires after every committed DDIM step and MAY THROW — the cooperative-cancellation
    /// seam (CAN-3 cadence: per denoise step); a thrown CancellationError propagates unchanged.
    public func run(_ inputs: Inputs,
                    onStep: (@Sendable (Int, Int) throws -> Void)? = nil) async throws
        -> [Float] {
        guard let unet = loadLock.withLock({ self.unet }),
              let decoder = loadLock.withLock({ self.decoder }) else {
            throw MoebiusCoreAIError.functionMissing(Self.unetAsset)
        }
        let ctx = loadLock.withLock { context }

        var sched = DDIMScheduler()
        sched.setTimesteps(numSteps)
        sched.applyStrength(strength)
        let timesteps = sched.timesteps

        var noisy = sched.addNoise(inputs.latents, noise: inputs.noise, timestep: timesteps[0])

        // Static graph inputs: context is constant across steps; sample/timestep refill per step.
        var contextND = NDArray(shape: [2, Self.contextTokens, Self.contextDim],
                                scalarType: .float16)
        contextND.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0 ..< ctx.count { ptr[i] = ctx[i] }
        }

        let lc = Self.latentCount              // 4·64·64

        for (i, t) in timesteps.enumerated() {
            let pred = try await Self.unetCall(
                unet, contextND: contextND, noisy: noisy, maskLatent: inputs.maskLatent,
                maskedLatents: inputs.maskedLatents, timestep: t)

            // CFG: uncond is batch 0, cond batch 1 (the input-id packing order).
            var predCFG = [Float](repeating: 0, count: lc)
            let g = guidanceScale
            for j in 0 ..< lc {
                let u = pred[j], c = pred[lc + j]
                predCFG[j] = u + (c - u) * g
            }

            noisy = sched.step(modelOutput: predCFG, timestep: t, sample: noisy)
            try onStep?(i + 1, timesteps.count)
        }

        // Decode (batch 1; graph wants UNSCALED latents).
        var latentND = NDArray(shape: [1, Self.latentChannels, Self.latentSide, Self.latentSide],
                               scalarType: .float16)
        let sf = Self.scalingFactor
        latentND.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0 ..< lc { ptr[i] = Float16(noisy[i] / sf) }
        }
        var outputs = try await decoder.run(inputs: ["x": latentND])
        let decoded = try Self.floats(from: &outputs, count: 3 * Self.side * Self.side)
        return decoded.map { ($0 + 1) / 2 }    // [-1,1] → [0,1]
    }

    /// One raw CFG-doubled UNet forward (both batches identical except conditioning) — the same
    /// packing `run()` uses, exposed so the parity gate covers the packing, not just the graph.
    /// Returns the raw `[2·4·64·64]` epsilon prediction (pre-CFG).
    public func unetForward(noisy: [Float], maskLatent: [Float], maskedLatents: [Float],
                            timestep: Int) async throws -> [Float] {
        guard let unet = loadLock.withLock({ self.unet }) else {
            throw MoebiusCoreAIError.functionMissing(Self.unetAsset)
        }
        let ctx = loadLock.withLock { context }
        var contextND = NDArray(shape: [2, Self.contextTokens, Self.contextDim],
                                scalarType: .float16)
        contextND.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, _ in
            for i in 0 ..< ctx.count { ptr[i] = ctx[i] }
        }
        return try await Self.unetCall(unet, contextND: contextND, noisy: noisy,
                                       maskLatent: maskLatent, maskedLatents: maskedLatents,
                                       timestep: timestep)
    }

    private static func unetCall(_ unet: InferenceFunction, contextND: NDArray, noisy: [Float],
                                 maskLatent: [Float], maskedLatents: [Float],
                                 timestep t: Int) async throws -> [Float] {
        let lc = latentCount
        let mc = latentSide * latentSide
        let perBatch = 9 * mc
        var sample = NDArray(shape: [2, 9, latentSide, latentSide], scalarType: .float16)
        sample.mutableView(as: Float16.self).withUnsafeMutablePointer { ptr, _, _ in
            for b in 0 ..< 2 {
                let base = b * perBatch
                for i in 0 ..< lc { ptr[base + i] = Float16(noisy[i]) }               // noisy(4)
                for i in 0 ..< mc { ptr[base + lc + i] = Float16(maskLatent[i]) }     // mask(1)
                for i in 0 ..< lc { ptr[base + lc + mc + i] = Float16(maskedLatents[i]) }
            }
        }
        var timestep = NDArray(shape: [2], scalarType: .float32)
        timestep.mutableView(as: Float.self).withUnsafeMutablePointer { ptr, _, _ in
            ptr[0] = Float(t); ptr[1] = Float(t)
        }
        var outputs = try await unet.run(inputs: [
            "sample": sample, "timestep": timestep, "encoder_hidden_states": contextND,
        ])
        return try floats(from: &outputs, count: 2 * lc)
    }

    // MARK: - helpers

    public static var latentCount: Int { latentChannels * latentSide * latentSide }

    private static func floats(from outputs: inout InferenceFunction.Outputs,
                               count: Int, fp32: Bool = false) throws -> [Float] {
        guard let value = outputs.remove("noise_pred") ?? outputs.remove("out")
            ?? outputs.remove("output"),
            let nd = value.ndArray else {
            throw MoebiusCoreAIError.functionMissing("output tensor")
        }
        var result = [Float](repeating: 0, count: count)
        if fp32 {
            nd.view(as: Float.self).withUnsafePointer { src, _, _ in
                for i in 0 ..< count { result[i] = src[i] }
            }
        } else {
            nd.view(as: Float16.self).withUnsafePointer { src, _, _ in
                for i in 0 ..< count { result[i] = Float(src[i]) }
            }
        }
        return result
    }

    /// Minimal .npy reader for the one file we ship: v1/v2 header, little-endian f4, C-order.
    public static func readNPY(_ url: URL, expectCount: Int) throws -> [Float] {
        let data = try Data(contentsOf: url)
        guard data.count > 10, data.prefix(6) == Data([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]) else {
            throw MoebiusCoreAIError.badNPY("bad magic in \(url.lastPathComponent)")
        }
        let major = data[6]
        let headerLen: Int
        let headerStart: Int
        if major == 1 {
            headerLen = Int(data[8]) | (Int(data[9]) << 8)
            headerStart = 10
        } else {
            headerLen = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16)
                | (Int(data[11]) << 24)
            headerStart = 12
        }
        let header = String(decoding: data[headerStart ..< (headerStart + headerLen)],
                            as: UTF8.self)
        guard header.contains("'descr': '<f4'"), header.contains("'fortran_order': False") else {
            throw MoebiusCoreAIError.badNPY("expected little-endian f4 C-order, got: \(header)")
        }
        let payload = data[(headerStart + headerLen)...]
        guard payload.count >= expectCount * 4 else {
            throw MoebiusCoreAIError.badNPY(
                "expected \(expectCount) floats, have \(payload.count / 4)")
        }
        return payload.withUnsafeBytes { raw in
            let f = raw.bindMemory(to: Float32.self)
            return (0 ..< expectCount).map { Float(bitPattern: f[$0].bitPattern) }
        }
    }
}
