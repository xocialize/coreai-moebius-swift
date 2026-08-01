---
license: mit
tags:
- coreai
- image-inpainting
- image-to-image
- diffusion
- apple-silicon
- moebius
library_name: coreai
---

# Moebius-CoreAI

[Moebius](https://github.com/hustvl/Moebius) — the 0.22B lightweight diffusion inpainting model
(object removal / image completion, Places2 fine-tune) — as **CoreAI `.aimodel` assets** for
Apple silicon, exported from the original [hustvl checkpoints](https://huggingface.co/hustvl/Moebius)
(MIT weights).

To our knowledge the first diffusion pipeline in `coreai-community`.

| asset | role | dtype | size | PSNR vs PyTorch golden |
|---|---|---|---|---|
| `moebius-unet-fp16-b2.aimodel` | denoiser (CFG batch-2) | fp16 | 452 MB | **68.3 dB** |
| `moebius-vae-encoder-fp32-b2.aimodel` | VAE posterior mean | fp32 | 137 MB | **104.7 dB** |
| `moebius-vae-decoder-fp16-b1.aimodel` | VAE decoder | fp16 | 99 MB | **68.5 dB** |
| `embedding_table.npy` | 20×3072 category conditioning | fp32 | 246 KB | exact |

## Numbers (measured, M5 Max, macOS 27)

- **UNet forward (fp16, GPU delegate): 49.8 ms** — 19-step CFG-2 projection **0.95 s**
  (the MLX port of the same checkpoint: 117.6 ms / 2.23 s).
- **Accuracy**: rel 9.338e-04 vs the shared PyTorch golden — the same fp16 floor as the MLX port
  (9.257e-04). The export folds all 124 BatchNorms into fp64-precomputed per-channel scale/shift
  (the checkpoint carries subnormal `running_var` channels that do not survive a naive fp16 cast).
- The exported UNet carries exact, rank-safe rewrites of the LambdaNetworks attention (einsum →
  broadcast/batched matmul; the positional Conv3d folded to a per-slice Conv2d) — numerically
  gated at export (fp32 pre/post rel ≤ 5e-07).
- The VAE **encoder ships fp32**: the SD-VAE encoder exceeds fp16 activation range (45.6 dB and
  CPU-lane NaN at fp16 — the classic `sdxl-vae-fp16-fix` problem). One encode per image makes
  fp32's cost invisible next to the denoise loop.

## Placement — GPU today, honestly

These assets run on the **GPU delegate**. Full-model Neural Engine compilation is currently
blocked by an ANECCompiler bug we filed with a validated repro —
[apple/coreai-models#138](https://github.com/apple/coreai-models/issues/138) (two 64²-level
transformer instances per graph break the input-channel-split pass, value-dependently). 17/18
model components already compile for ANE individually; when the OS compiler fixes #138 these
assets inherit the ANE by re-export, no consumer change. Note the failure mode: an ANE request
that fails **silently falls back to GPU** — verify placement with the GPU-idle signature, never
by "it ran".

## Usage

Pipeline: encode `[image, masked_image]` (fp32, `[2,3,512,512]`, `[-1,1]`) → posterior mean ×
0.13025 → DDIM (`scaled_linear` betas 0.00085–0.012, 20 steps, strength 0.99 → 19 steps from
t=900, CFG 2.5, noise offset 0.0357) over the UNet (`sample` `[2,9,64,64]` fp16 =
noisy(4)+mask(1)+masked(4), `timestep` `[2]` fp32, `encoder_hidden_states` `[2,10,3072]` fp16 =
table rows [10..19; 0..9]) → decode `latents / 0.13025` (fp16, `[1,4,64,64]`) → `(x+1)/2`.

A ready-made Swift package that does exactly this — scheduler, conditioning, image I/O,
mask compositing, MLXEngine integration, tests —
[`xocialize/coreai-moebius-swift`](https://github.com/xocialize/coreai-moebius-swift).

```swift
import CoreAI
let model = try await AIModel(contentsOf: unetURL,
    options: SpecializationOptions(preferredComputeUnitKind: .gpu))
let fn = try model.loadFunction(named: "main")!
// first load pays E5RT specialization (~40 s for the UNet, OS-cached after)
```

## Reproducibility

`export_unet.py` and `export_vae.py` (in this repo) re-create every asset from the original
checkpoints: PyTorch → `torch.export` → `coreai-torch` `TorchConverter` → `.aimodel`, with every
graph rewrite numerically gated in-line. No opaque binaries.

## Provenance & license

Model: [hustvl/Moebius](https://github.com/hustvl/Moebius) (paper:
[arXiv:2606.19195](https://arxiv.org/abs/2606.19195)) — **MIT weights**, Apache-2.0 reference
code. VAE: the SD KL-f8 autoencoder distributed with PixelHacker (MIT). This repo redistributes
the weights in a converted container under MIT, with the conversion scripts included.
