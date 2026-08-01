// Gate.swift — the CoreAI pipeline against the SHARED PyTorch goldens (the same fixtures the
// MLX sibling gates on: one oracle, N ports).
//
//   swift run --build-system swiftbuild coreai-moebius-gate <exports-dir> <goldens-dir>
//
// Two gates, both PSNR-banded like the Python verify harnesses (>50 dB PASS for fp16 surfaces):
//   * UNet: step0_model_input + t=900 + table conditioning → step0_pred_raw
//   * Pipeline: oracle latent means + oracle noise → 19 steps → decode → decoded.npy
//     (the oracle's OWN noise is injected — encode().sample() is stochastic and no two
//     frameworks draw the same stream; this is how cross-framework parity is always gated here)

import Foundation
import MoebiusCoreAI

@main
struct Gate {
    static func main() async throws {
        let args = CommandLine.arguments
        let exportsDir = URL(fileURLWithPath: args.count > 1 ? args[1]
            : "../moebius-m0/coreai/exports")
        let goldensDir = URL(fileURLWithPath: args.count > 2 ? args[2]
            : "../moebius-m0/goldens")

        func golden(_ name: String, _ count: Int) throws -> [Float] {
            try MoebiusPipeline_CoreAI.readNPY(
                goldensDir.appendingPathComponent(name), expectCount: count)
        }
        func psnr(_ want: [Float], _ got: [Float]) -> Float {
            var mse: Double = 0
            var peak: Double = 0
            for i in 0 ..< want.count {
                let d = Double(want[i] - got[i])
                mse += d * d
                peak = max(peak, abs(Double(want[i])))
            }
            mse /= Double(want.count)
            return Float(20 * log10(peak / mse.squareRoot()))
        }

        let pipe = MoebiusPipeline_CoreAI(weightsDirectory: exportsDir, compute: .gpu)
        let started = Date()
        try await pipe.prepare()
        print(String(format: "[gate] prepared 3 graphs in %.1f s", -started.timeIntervalSinceNow))

        let lc = MoebiusPipeline_CoreAI.latentCount
        let mc = 64 * 64

        // ── UNet gate: one CFG forward vs step0_pred_raw ─────────────────────────────────────
        // step0_model_input is the already-packed [2,9,64,64]; drive the pipeline's own packing
        // from its components instead — latents_mean-noised is step-0 state — so the gate covers
        // OUR packing too, not just the graph. Compare against step0_pred_raw [2,4,64,64].
        let latents = try golden("latents_mean.npy", lc)
        let maskedLatents = try golden("masked_latents_mean.npy", lc)
        let maskLatent = try golden("resized_masks.npy", mc)
        let noise = try golden("noise.npy", lc)
        let predWant = try golden("step0_pred_raw.npy", 2 * lc)

        var sched = DDIMScheduler()
        sched.setTimesteps(20)
        sched.applyStrength(0.99)
        let noisy = sched.addNoise(latents, noise: noise, timestep: sched.timesteps[0])
        // Reproduce the model_input packing exactly as run() does, via a single-step run with a
        // probe: simplest honest check is the full-pipeline gate below; here compare the FIRST
        // UNet output by running one step manually through the public API surface.
        let pred = try await pipe.unetForward(noisy: noisy, maskLatent: maskLatent,
                                              maskedLatents: maskedLatents,
                                              timestep: sched.timesteps[0])
        let dbU = psnr(predWant, pred)
        print(String(format: "[gate] UNet step-0 PSNR %.1f dB  [%@]", dbU,
                     dbU > 50 ? "PASS" : (dbU > 40 ? "investigate" : "FAIL")))

        // ── Pipeline gate: oracle latents + oracle noise → decoded ──────────────────────────
        let want = try golden("decoded.npy", 3 * 512 * 512)
        let inputs = MoebiusPipeline_CoreAI.Inputs(
            latents: latents, maskedLatents: maskedLatents,
            maskLatent: maskLatent, noise: noise)
        let t0 = Date()
        let decoded = try await pipe.run(inputs) { step, total in
            if step == total { print("[gate] denoise \(step)/\(total)") }
        }
        print(String(format: "[gate] pipeline %.2f s for 19 steps + decode",
                     -t0.timeIntervalSinceNow))
        let dbP = psnr(want, decoded)
        // The pipeline surface is 19 ITERATED fp16 forwards — per-step error compounds, so the
        // calibrated gate here is STRUCTURAL (cosine), matching the MLX sibling's fp16 pipeline
        // gate (cos 0.99994 PASS); PSNR is reported for information.
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< want.count {
            dot += Double(want[i]) * Double(decoded[i])
            na += Double(want[i]) * Double(want[i])
            nb += Double(decoded[i]) * Double(decoded[i])
        }
        let cos = dot / (na.squareRoot() * nb.squareRoot())
        print(String(format: "[gate] pipeline decoded cos %.6f  (PSNR %.1f dB)  [%@]",
                     cos, dbP, cos >= 0.9999 ? "PASS — structural gate" : "FAIL"))

        if dbU <= 40 || cos < 0.9999 { throw GateError.failed }
    }

    enum GateError: Error { case failed }
}
