#!/usr/bin/env bash
set -e

############################
# User-configurable inputs
############################

PROMPT="A middle-aged Caucasian male with deep-set blue eyes and a neatly trimmed salt-and-pepper beard, his weathered face adorned with fine laugh lines around the eyes.\n\nHe gazes softly into the camera, a warm smile hinting at a lifetime of experiences, his gaze steady and reassuring.\n\nThe photograph is taken from a slightly elevated angle, giving viewers a direct connection with his expressive eyes. The soft, directional lighting from a nearby window casts gentle shadows across his face, highlighting the texture of his skin and the flecks of gold in his irises.\n\nHe wears a well worn, light blue denim shirt, one of his sleeves rolled up slightly, revealing a faded, tattoo on the inside of his forearm.\n\nThis intimate portrait captures the essence of quiet resilience and gentle wisdom."

EXP_NAME="orig_uniform"
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
  --attn-backend "${ATTN_BACKEND}"
