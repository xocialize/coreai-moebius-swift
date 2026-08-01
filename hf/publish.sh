#!/bin/zsh
# Publish the Moebius CoreAI assets + model card to HF coreai-community.
# Prerequisites (same two that blocked realesrgan's first attempt — both 403 identically):
#   1. coreai-community membership APPROVED
#   2. The fine-grained token granted repo.write on coreai-community
set -e
cd "$(dirname "$0")"
REPO=coreai-community/Moebius-CoreAI
M0=../../moebius-m0/coreai
hf repo create $REPO --repo-type model || true
STAGE=$(mktemp -d)
cp README.md "$STAGE/"
cp "$M0/export_unet.py" "$M0/export_vae.py" "$STAGE/"
cp -R "$M0/exports/moebius-unet-fp16-b2.aimodel" "$STAGE/"
cp -R "$M0/exports/moebius-vae-encoder-fp32-b2.aimodel" "$STAGE/"
cp -R "$M0/exports/moebius-vae-decoder-fp16-b1.aimodel" "$STAGE/"
cp "$M0/exports/embedding_table.npy" "$STAGE/"
hf upload $REPO "$STAGE" . --repo-type model \
  --commit-message "Moebius 0.22B diffusion inpainting fp16/fp32 — first diffusion pipeline in coreai-community"
rm -rf "$STAGE"
echo "published: https://huggingface.co/$REPO"
