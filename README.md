# coreai-moebius-swift

[Moebius](https://github.com/hustvl/Moebius) (0.22B diffusion inpainting, Places2) on Apple
silicon via **CoreAI** static graphs — sibling to
[`mlx-moebius-swift`](https://github.com/xocialize/mlx-moebius-swift), same capability, different
backend, **2.7× faster end-to-end**.

|  | this package (CoreAI) | mlx-moebius-swift |
|---|---|---|
| UNet forward (fp16) | **49.8 ms** | 117.6 ms |
| 19 steps + decode | **1.01 s** | 2.7 s |
| pipeline parity vs PyTorch oracle | cos 0.999964 | cos 0.999941 |

Assets materialize from
[coreai-community/Moebius-CoreAI](https://huggingface.co/coreai-community/Moebius-CoreAI)
(fp16 UNet with rank-safe λ rewrites + fp64-precomputed BatchNorm; fp32 VAE encoder — the SD-VAE
encoder exceeds fp16 range; fp16 decoder). Export scripts ship in the HF repo.

- `MoebiusCoreAI` — the pipeline core (three `.aimodel` graphs + host-side DDIM/CFG/conditioning)
- `CoreAIMoebius` — the MLXEngine `imageInpaint` package (`coreai-moebius-inpaint`), MLXToolKit
  only, registerable beside the MLX sibling
- `coreai-moebius-gate` — parity gate vs the shared PyTorch goldens (also the fresh-download
  verifier)

⚠️ macOS 27+ (CoreAI.framework). **GPU delegate** — full-model Neural Engine compilation is
blocked on [apple/coreai-models#138](https://github.com/apple/coreai-models/issues/138); these
assets inherit the ANE by re-export when the OS compiler fixes it.

MIT (weights MIT via hustvl; port code MIT).
