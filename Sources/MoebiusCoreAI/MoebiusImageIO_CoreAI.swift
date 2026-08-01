import Accelerate
import CoreGraphics
import Foundation

/// CGImage ↔ planar-float plumbing, mirroring the MLX sibling's `MoebiusImageIO` (itself gated
/// against the reference `_preprocess`/`_post_process`) with plain `[Float]` in place of MLX:
/// binarize mask → resize to 512² → re-binarize → masked = image·(1−mask); latent mask by NEAREST
/// 512→64; paste back through the mask at ORIGINAL resolution. Any behavioural divergence from
/// the sibling here is a defect — the two backends must be interchangeable behind a planner.
public enum MoebiusImageIO_CoreAI {

    public static let side = 512
    public static let latentSide = 64

    public struct Prepared {
        public let image: [Float]        // [3·512·512] CHW, [-1,1]
        public let maskedImage: [Float]  // [3·512·512] CHW
        public let maskLatent: [Float]   // [64·64], 1 = remove
    }

    public static func prepare(source: CGImage, mask: CGImage) -> Prepared {
        let image01 = rgbCHW(resize(source, width: side, height: side))    // [3·n], [0,1]
        let scaled = image01.map { $0 * 2 - 1 }
        let maskFull = binarizedMask(resize(mask, width: side, height: side))  // [n]
        let n = side * side
        var masked = scaled
        for c in 0 ..< 3 {
            for i in 0 ..< n where maskFull[i] >= 0.5 {
                masked[c * n + i] = 0
            }
        }
        // F.interpolate default = NEAREST: index floor(i·8) = i·8.
        var maskLatent = [Float](repeating: 0, count: latentSide * latentSide)
        for y in 0 ..< latentSide {
            for x in 0 ..< latentSide {
                maskLatent[y * latentSide + x] = maskFull[(y * 8) * side + (x * 8)]
            }
        }
        return Prepared(image: scaled, maskedImage: masked, maskLatent: maskLatent)
    }

    /// The reference's two RNG draws: base noise + `noise_offset`·per-channel offset. Seeded and
    /// deterministic per seed. ⚠️ NOT bit-identical to the MLX sibling's stream (different RNGs);
    /// per-backend determinism is the contract, cross-backend bit-parity is not — parity gates
    /// inject the shared fixture noise instead.
    public static func offsetNoise(count: Int, channels: Int, seed: UInt64,
                                   offset: Float) -> [Float] {
        var rng = SplitMix64(seed: seed)
        var noise = gaussians(count: count, rng: &rng)
        let perChannel = gaussians(count: channels, rng: &rng)
        let chunk = count / channels
        for c in 0 ..< channels {
            let o = offset * perChannel[c]
            for i in 0 ..< chunk {
                noise[c * chunk + i] += o
            }
        }
        return noise
    }

    /// Composite the 512² fill back into the FULL-RESOLUTION source through a BLURRED mask —
    /// `result·soft + source·(1−soft)` with soft = 3× iterated 3×3 box blur of the binary mask,
    /// exactly the sibling's `MoebiusPipeline.paste` (blurRadius 3, edge-replicated).
    public static func pasteAtOriginal(fill512 chw: [Float], source: CGImage,
                                       mask: CGImage) throws -> CGImage {
        let w = source.width, h = source.height
        let fillCG = try toCGImage(chw, width: side, height: side)
        let fillFull = rgbCHW(resize(fillCG, width: w, height: h))
        let sourceFull = rgbCHW(source)
        // ⚠️ Resize the mask to the SOURCE's pixel dims first — a caller may hand us a mask sized
        // in POINTS while the image is in PIXELS (any asset whose DPI ≠ 72; a 150-DPI 2250×4000
        // PNG reports 1080×1920 points). The MLX sibling crashed on exactly this before the same
        // fix landed there. The mask is authoritative for WHERE, never for HOW BIG.
        var soft = binarizedMask(resize(mask, width: w, height: h))
        for _ in 0 ..< 3 { soft = boxBlur3(soft, width: w, height: h) }
        let n = w * h
        var out = [Float](repeating: 0, count: 3 * n)
        for c in 0 ..< 3 {
            for i in 0 ..< n {
                let s = soft[i]
                out[c * n + i] = fillFull[c * n + i] * s + sourceFull[c * n + i] * (1 - s)
            }
        }
        return try toCGImage(out, width: w, height: h)
    }

    /// 3×3 box mean with edge replication — mirrors the sibling's `boxBlur3NCHW`.
    ///
    /// ⚠️ vImage, not a scalar loop. MEASURED in-app 2026-08-01: the scalar version cost **12.5 s**
    /// of a 14.1 s run on a 2250×4000 source (three iterations × 9 taps × 9M px ≈ 243M scalar ops),
    /// while the run's actual model work was 1.6 s. The MLX sibling never showed this because its
    /// paste rides MLX's vectorized array ops. Host-side postprocess must be vectorized in a
    /// package whose accelerator work is measured in milliseconds — otherwise the compositor IS
    /// the runtime. (The CLI parity gate missed it entirely: it composites nothing, so this only
    /// ever appears under in-app validation at real resolutions.)
    static func boxBlur3(_ x: [Float], width w: Int, height h: Int) -> [Float] {
        var src = x
        var out = [Float](repeating: 0, count: w * h)
        let kernel = [Float](repeating: 1.0 / 9.0, count: 9)
        src.withUnsafeMutableBufferPointer { sp in
            out.withUnsafeMutableBufferPointer { op in
                var inBuf = vImage_Buffer(data: sp.baseAddress, height: vImagePixelCount(h),
                                          width: vImagePixelCount(w), rowBytes: w * 4)
                var outBuf = vImage_Buffer(data: op.baseAddress, height: vImagePixelCount(h),
                                           width: vImagePixelCount(w), rowBytes: w * 4)
                _ = kernel.withUnsafeBufferPointer { kp in
                    vImageConvolve_PlanarF(&inBuf, &outBuf, nil, 0, 0, kp.baseAddress!, 3, 3, 0,
                                           vImage_Flags(kvImageEdgeExtend))
                }
            }
        }
        return out
    }

    // MARK: primitives

    static func resize(_ image: CGImage, width: Int, height: Int) -> CGImage {
        guard image.width != width || image.height != height else { return image }
        guard let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return image }
        ctx.interpolationQuality = .high
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage() ?? image
    }

    /// CHW float32 in `[0,1]`.
    static func rgbCHW(_ image: CGImage) -> [Float] {
        let w = image.width, h = image.height
        var buffer = [UInt8](repeating: 0, count: w * h * 4)
        buffer.withUnsafeMutableBytes { raw in
            guard let ctx = CGContext(
                data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        let n = w * h
        var chw = [Float](repeating: 0, count: 3 * n)
        for i in 0 ..< n {
            chw[i] = Float(buffer[i * 4]) / 255
            chw[n + i] = Float(buffer[i * 4 + 1]) / 255
            chw[2 * n + i] = Float(buffer[i * 4 + 2]) / 255
        }
        return chw
    }

    /// Luma ≥ 0.5 → 1 (white = remove).
    static func binarizedMask(_ image: CGImage) -> [Float] {
        let rgb = rgbCHW(image)
        let n = image.width * image.height
        var mask = [Float](repeating: 0, count: n)
        for i in 0 ..< n {
            let luma = (rgb[i] + rgb[n + i] + rgb[2 * n + i]) / 3
            mask[i] = luma >= 0.5 ? 1 : 0
        }
        return mask
    }

    /// CHW `[0,1]` (clipped) → CGImage.
    public static func toCGImage(_ chw: [Float], width w: Int, height h: Int) throws -> CGImage {
        let n = w * h
        var rgba = [UInt8](repeating: 255, count: n * 4)
        for i in 0 ..< n {
            for c in 0 ..< 3 {
                let v = max(0, min(1, chw[c * n + i]))
                rgba[i * 4 + c] = UInt8((v * 255).rounded())
            }
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                               bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false,
                               intent: .defaultIntent)
        else { throw MoebiusCoreAIError.imageEncodeFailed }
        return cg
    }

    // MARK: seeded gaussian RNG

    /// SplitMix64 → Box–Muller. Deterministic per seed; NOT the MLX stream (see offsetNoise note).
    struct SplitMix64 {
        var state: UInt64
        init(seed: UInt64) { state = seed &+ 0x9E37_79B9_7F4A_7C15 }
        mutating func next() -> UInt64 {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func uniform() -> Double {
            Double(next() >> 11) * (1.0 / 9_007_199_254_740_992.0)
        }
    }

    static func gaussians(count: Int, rng: inout SplitMix64) -> [Float] {
        var out = [Float](repeating: 0, count: count)
        var i = 0
        while i < count {
            let u1 = max(rng.uniform(), 1e-12)
            let u2 = rng.uniform()
            let r = (-2 * Foundation.log(u1)).squareRoot()
            out[i] = Float(r * Foundation.cos(2 * Double.pi * u2))
            if i + 1 < count { out[i + 1] = Float(r * Foundation.sin(2 * Double.pi * u2)) }
            i += 2
        }
        return out
    }
}

public enum MoebiusCoreAIError: Error, CustomStringConvertible {
    case assetMissing(String)
    case functionMissing(String)
    case badNPY(String)
    case imageEncodeFailed

    public var description: String {
        switch self {
        case .assetMissing(let n): return "missing .aimodel: \(n)"
        case .functionMissing(let n): return "no 'main' function in \(n)"
        case .badNPY(let m): return "embedding table unreadable: \(m)"
        case .imageEncodeFailed: return "could not encode output image"
        }
    }
}
