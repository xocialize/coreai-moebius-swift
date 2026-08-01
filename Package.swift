// swift-tools-version: 6.2

// coreai-moebius-swift — Moebius (0.22B diffusion inpainting) on Apple GPU via CoreAI/MPSGraph.
//
// Sibling to `mlx-moebius-swift` (same capability home, different backend), following the
// `coreai-realesrgan-swift` pattern. Why it exists, measured (moebius-m0 `coreai/RESULTS.md`,
// 2026-08-01): the CoreAI static-graph executable runs the identical fp16 UNet forward in
// **49.8 ms vs the MLX sibling's 117.6 ms** (2.4×; 19-step CFG-2 projection 0.95 s vs 2.23 s) at
// the same fp16 accuracy floor (68.3 dB vs the shared PyTorch golden).
//
// ⚠️ GPU, not ANE — stated plainly because this repo's history contains two retracted "ANE"
// claims: full-model ANE compilation is blocked on an ANECCompiler bug (two 64²-level
// transformers per graph break the input-channel-split pass, VALUE-dependently — filed as
// apple/coreai-models#138 with a validated repro). 17/18 components already compile for ANE;
// when the OS fixes #138, this package inherits the ANE by re-export, no code change.
//
// ⚠️ macOS 27+ ONLY (CoreAI.framework). The MLX sibling is the fallback below 27.
//
// 🔑 Unlike the realesrgan donor, assets are NOT vendored: the UNet alone is 452 MB. They
// materialize from `coreai-community/Moebius-CoreAI` through the engine's WeightSourcing
// contract, like any big-weight package.
import PackageDescription

let package = Package(
    name: "coreai-moebius-swift",
    platforms: [.macOS("27.0")],
    products: [
        .library(name: "MoebiusCoreAI", targets: ["MoebiusCoreAI"]),
        // The MLXEngine `imageInpaint` package over the core — registerable beside the MLX
        // sibling as a second backend behind the same capability. ⚠️ Depends on MLXToolKit ONLY
        // (the engine's dependency-free contract layer): this product carries NO MLX.
        .library(name: "CoreAIMoebius", targets: ["CoreAIMoebius"]),
        // Parity gate vs the shared PyTorch goldens (also the fresh-download verifier).
        .executable(name: "coreai-moebius-gate", targets: ["CoreAIMoebiusGate"]),
    ],
    dependencies: [
        .package(url: "https://github.com/xocialize/mlx-engine-swift", from: "0.40.0"),
    ],
    targets: [
        .target(
            name: "MoebiusCoreAI",
            linkerSettings: [.linkedFramework("CoreAI")]
        ),
        .testTarget(name: "MoebiusCoreAITests", dependencies: ["MoebiusCoreAI"]),
        .executableTarget(name: "CoreAIMoebiusGate", dependencies: ["MoebiusCoreAI"]),
        .target(name: "CoreAIMoebius",
                dependencies: [
                    "MoebiusCoreAI",
                    .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                ]),
        .testTarget(name: "CoreAIMoebiusTests",
                    dependencies: [
                        "CoreAIMoebius",
                        .product(name: "MLXServeConformance", package: "mlx-engine-swift"),
                    ]),
    ]
)
