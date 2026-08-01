import Foundation
import XCTest
@testable import MoebiusCoreAI

/// Host-math tests — everything here runs without assets or the CoreAI runtime.
final class MoebiusCoreAITests: XCTestCase {

    // MARK: DDIM — the schedule facts the port history says a plausible substitute gets wrong

    func testTimestepScheduleAndStrength() {
        var s = DDIMScheduler()
        s.setTimesteps(20)
        XCTAssertEqual(s.timesteps.first, 950)
        XCTAssertEqual(s.timesteps.last, 0)
        XCTAssertEqual(s.timesteps.count, 20)
        // strength 0.99 drops the LEADING timestep: 19 steps starting at 900, not 950.
        s.applyStrength(0.99)
        XCTAssertEqual(s.timesteps.count, 19)
        XCTAssertEqual(s.timesteps.first, 900)
    }

    func testScaledLinearBetasNotLinear() {
        // "scaled_linear" is linear in SQRT space then squared — alphasCumprod[0] must reflect
        // beta_start exactly, and the midpoint must differ from a linear ramp's.
        let s = DDIMScheduler()
        XCTAssertEqual(s.alphasCumprod[0], 1 - 0.00085, accuracy: 1e-7)
        let linearMidBeta: Float = (0.00085 + 0.012) / 2
        let sqrtMid = ((0.00085 as Float).squareRoot() + (0.012 as Float).squareRoot()) / 2
        let scaledMidBeta = sqrtMid * sqrtMid
        XCTAssertNotEqual(scaledMidBeta, linearMidBeta)
        XCTAssertEqual(scaledMidBeta, 0.0048094, accuracy: 1e-5)
    }

    func testDDIMStepMatchesClosedForm() {
        var s = DDIMScheduler()
        s.setTimesteps(20)
        s.applyStrength(0.99)
        let t = 900
        let sample: [Float] = [0.5]
        let eps: [Float] = [0.1]
        let out = s.step(modelOutput: eps, timestep: t, sample: sample)
        let aT = s.alphasCumprod[t], aPrev = s.alphasCumprod[t - 50]
        let x0 = (0.5 - 0.1 * (1 - aT).squareRoot()) / aT.squareRoot()
        let want = x0 * aPrev.squareRoot() + 0.1 * (1 - aPrev).squareRoot()
        XCTAssertEqual(out[0], want, accuracy: 1e-6)
    }

    // MARK: noise

    func testOffsetNoiseDeterministicPerSeed() {
        let a = MoebiusImageIO_CoreAI.offsetNoise(count: 4 * 64 * 64, channels: 4, seed: 7,
                                                  offset: 0.0357)
        let b = MoebiusImageIO_CoreAI.offsetNoise(count: 4 * 64 * 64, channels: 4, seed: 7,
                                                  offset: 0.0357)
        let c = MoebiusImageIO_CoreAI.offsetNoise(count: 4 * 64 * 64, channels: 4, seed: 8,
                                                  offset: 0.0357)
        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
        // Roughly standard normal: mean ~0, var ~1 (+ tiny offset contribution).
        let mean = a.reduce(0, +) / Float(a.count)
        let variance = a.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Float(a.count)
        XCTAssertEqual(mean, 0, accuracy: 0.05)
        XCTAssertEqual(variance, 1, accuracy: 0.1)
    }

    // MARK: npy reader

    func testReadNPYRoundTrip() throws {
        // Hand-assembled v1 .npy: [2,3] f4 C-order.
        var header = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"
        while (10 + header.count + 1) % 64 != 0 { header += " " }
        header += "\n"
        var data = Data([0x93]) + "NUMPY".data(using: .ascii)! + Data([1, 0])
        data += Data([UInt8(header.count & 0xFF), UInt8(header.count >> 8)])
        data += header.data(using: .ascii)!
        let values: [Float32] = [1, 2, 3, 4, 5, 6]
        values.withUnsafeBytes { data += Data($0) }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("npytest-\(UUID().uuidString).npy")
        try data.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let out = try MoebiusPipeline_CoreAI.readNPY(url, expectCount: 6)
        XCTAssertEqual(out, [1, 2, 3, 4, 5, 6])
    }

    // MARK: mask geometry

    func testLatentMaskIsNearestDownsample() {
        // A mask image with a white 64×64 top-left square at 512² should produce an 8×8 white
        // block at 64² under NEAREST (index i·8).
        let side = 512
        var rgba = [UInt8](repeating: 0, count: side * side * 4)
        for y in 0 ..< 64 {
            for x in 0 ..< 64 {
                let p = (y * side + x) * 4
                rgba[p] = 255; rgba[p + 1] = 255; rgba[p + 2] = 255; rgba[p + 3] = 255
            }
        }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        let mask = CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                           bytesPerRow: side * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                           bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                           provider: provider, decode: nil, shouldInterpolate: false,
                           intent: .defaultIntent)!
        let prepared = MoebiusImageIO_CoreAI.prepare(source: mask, mask: mask)
        let m = prepared.maskLatent
        XCTAssertEqual(m[0], 1)
        XCTAssertEqual(m[7 * 64 + 7], 1)
        XCTAssertEqual(m[8 * 64 + 8], 0)
        XCTAssertEqual(m[63 * 64 + 63], 0)
    }
}
