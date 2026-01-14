#!/usr/bin/env bash
set -e

############################
# User-configurable inputs
############################

PROMPT="A masterful portrait of a younger Native American man with striking hazel eyes that hint at a blend of strength and vulnerability. His dark brown hair, styled in loose braids with feathers woven in, captures the essence of his cultural heritage. Wearing a traditional beaded shirt with intricate patterns, his gaze is directed slightly upward and to the side, as if lost in contemplation. The soft, natural light from a nearby window casts gentle shadows on his high cheekbones and jawline, emphasizing the texture of his skin and the subtle wrinkles around his eyes. A slightly blurred background reveals the rustic, earth-toned colors of his home environment, adding depth and context to this timeless and evocative portrait."

EXP_NAME="inverse-cleanest_skip"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="runs/${EXP_NAME}_${TIMESTAMP}"

CKPT="/root/autodl-tmp/models/Tongyi-MAI/Z-Image-Turbo"
DTYPE="bf16"          # bf16 | fp16 | fp32
HEIGHT=1024
WIDTH=1024
STEPS=8
GUIDANCE=0.0
SEED=42

NOISE="uniform"        # normal | uniform
NOISE_SCALE=1.0
UNIFORM_LOW=-1.0
UNIFORM_HIGH=1.0

# Watermark
W_CHANNEL="0 1 2 3"
W_ALPHA="1 1 1 1"

SAVE_EVERY=1
DECODE_EVERY=1

ATTN_BACKEND="${ZIMAGE_ATTENTION:-_native_flash}"

############################
# Optional feature toggles
############################

COMPILE_FLAG="--compile"
DECODE_FLAG="--decode"

############################
# Launch inference
############################

python trajectory_experiment.py \
  --prompt "${PROMPT}" \
  --out "${OUT_DIR}" \
  --ckpt "${CKPT}" \
  --dtype "${DTYPE}" \
  ${COMPILE_FLAG} \
  --height ${HEIGHT} \
  --width ${WIDTH} \
  --steps ${STEPS} \
  --guidance ${GUIDANCE} \
  --seed ${SEED} \
  --noise "${NOISE}" \
  --noise-scale ${NOISE_SCALE} \
  --uniform-low ${UNIFORM_LOW} \
  --uniform-high ${UNIFORM_HIGH} \
  --save-every ${SAVE_EVERY} \
  ${DECODE_FLAG} \
  --decode-every ${DECODE_EVERY} \
  --attn-backend "${ATTN_BACKEND}" \
  --w_channel ${W_CHANNEL} \
  --w_alpha ${W_ALPHA}
