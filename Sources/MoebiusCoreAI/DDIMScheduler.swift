import Foundation

/// DDIM configured exactly as Moebius' `build_pipeline` does — a direct transcription of the
/// MLX sibling's scheduler (itself gated 1:1 against diffusers): `beta_start 0.00085`,
/// `beta_end 0.012`, `scaled_linear` (linear in SQRT space then squared), 1000 train steps,
/// `eta = 0`, epsilon prediction, `clip_sample false`, `set_alpha_to_one true`.
///
/// Operates on plain `[Float]` — latents are 2·4·64·64 = 32K elements; scheduler math is host
/// work in this backend, the accelerator only sees the UNet/VAE graphs.
public struct DDIMScheduler: Sendable {
    public let alphasCumprod: [Float]
    public let numTrainTimesteps: Int
    public private(set) var timesteps: [Int] = []
    public private(set) var numInferenceSteps: Int = 0

    public init(betaStart: Float = 0.00085, betaEnd: Float = 0.012,
                numTrainTimesteps: Int = 1000) {
        self.numTrainTimesteps = numTrainTimesteps
        var cumulative: Float = 1
        var acc: [Float] = []
        acc.reserveCapacity(numTrainTimesteps)
        let lo = betaStart.squareRoot(), hi = betaEnd.squareRoot()
        for i in 0 ..< numTrainTimesteps {
            let t = numTrainTimesteps == 1 ? 0 : Float(i) / Float(numTrainTimesteps - 1)
            let s = lo + (hi - lo) * t
            let beta = s * s
            cumulative *= (1 - beta)
            acc.append(cumulative)
        }
        self.alphasCumprod = acc
    }

    /// `[950, 900, …, 0]` for 20 steps over 1000 train steps — descending, `step_ratio` apart.
    public mutating func setTimesteps(_ steps: Int) {
        numInferenceSteps = steps
        let ratio = numTrainTimesteps / steps
        timesteps = (0 ..< steps).map { $0 * ratio }.reversed()
    }

    /// The pipeline's `strength = 0.99` drops the leading timestep, leaving 19 of 20.
    public mutating func applyStrength(_ strength: Float) {
        let initSteps = min(Int(Float(numInferenceSteps) * strength), numInferenceSteps)
        let start = max(numInferenceSteps - initSteps, 0)
        timesteps = Array(timesteps.dropFirst(start))
    }

    /// `q(x_t | x_0)` — used once, to noise the CLEAN latents at the first timestep.
    public func addNoise(_ original: [Float], noise: [Float], timestep: Int) -> [Float] {
        let a = alphasCumprod[timestep]
        let sa = a.squareRoot(), sn = (1 - a).squareRoot()
        return zip(original, noise).map { $0 * sa + $1 * sn }
    }

    /// One deterministic DDIM step (eta = 0), epsilon-prediction, unclamped x₀.
    public func step(modelOutput: [Float], timestep: Int, sample: [Float]) -> [Float] {
        let prev = timestep - numTrainTimesteps / numInferenceSteps
        let alphaProd = alphasCumprod[timestep]
        let alphaProdPrev = prev >= 0 ? alphasCumprod[prev] : Float(1)
        let betaProd = 1 - alphaProd
        let sBeta = betaProd.squareRoot()
        let sAlpha = alphaProd.squareRoot()
        let sPrev = alphaProdPrev.squareRoot()
        let sDir = (1 - alphaProdPrev).squareRoot()
        var out = [Float](repeating: 0, count: sample.count)
        for i in 0 ..< sample.count {
            let predOriginal = (sample[i] - modelOutput[i] * sBeta) / sAlpha
            out[i] = predOriginal * sPrev + modelOutput[i] * sDir
        }
        return out
    }
}
