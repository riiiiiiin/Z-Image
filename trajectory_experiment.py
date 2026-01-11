"""Run Z-Image generation while recording the denoising trajectory.

Examples:
  python trajectory_experiment.py --prompt "a cat" --noise normal --out runs/cat_normal
  python trajectory_experiment.py --prompt "a cat" --noise uniform --uniform-low -1 --uniform-high 1 --out runs/cat_uniform

This script saves:
  - runs/.../latents/step_XXXX.pt
  - runs/.../images/step_XXXX_00.png (optional)
  - runs/.../images/final_00.png
  - runs/.../trajectory.json
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate
from zimage.noise import normal_noise_sampler, uniform_noise_sampler
from zimage.trajectory import TrajectoryRecorder


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", type=str, default="Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights.")
    p.add_argument("--out", type=str, required=True, help="Output directory for trajectory files")

    p.add_argument("--ckpt", type=str, default="/root/autodl-tmp/models/Tongyi-MAI/Z-Image-Turbo")
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--compile", action="store_true")

    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--guidance", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--noise", type=str, default="normal", choices=["normal", "uniform"])
    p.add_argument("--noise-scale", type=float, default=1.0)
    p.add_argument("--uniform-low", type=float, default=-1.0)
    p.add_argument("--uniform-high", type=float, default=1.0)

    p.add_argument("--save-every", type=int, default=1, help="Save latents every N steps")
    p.add_argument("--decode", action="store_true", help="Decode and save images along the trajectory")
    p.add_argument("--decode-every", type=int, default=1, help="Decode images every N steps")

    p.add_argument("--attn-backend", type=str, default=os.environ.get("ZIMAGE_ATTENTION", "_native_flash"))
    return p.parse_args()


def main():
    args = parse_args()

    device = select_device()
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model_path = ensure_model_weights(args.ckpt, verify=False)
    components = load_from_local_dir(model_path, device=device, dtype=dtype, compile=args.compile)

    AttentionBackend.print_available_backends()
    set_attention_backend(args.attn_backend)

    generator = torch.Generator(device).manual_seed(args.seed)

    if args.noise == "uniform":
        noise_sampler = uniform_noise_sampler(
            low=args.uniform_low,
            high=args.uniform_high,
            scale=args.noise_scale,
        )
    else:
        noise_sampler = normal_noise_sampler(scale=args.noise_scale)

    recorder = TrajectoryRecorder(
        args.out,
        save_latents=True,
        save_images=True,
        save_every=args.save_every,
        save_images_every=args.decode_every,
        keep_only_first_image=True,
    )
    recorder.set_run_config(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        steps=args.steps,
        guidance=args.guidance,
        seed=args.seed,
        noise=args.noise,
        noise_scale=args.noise_scale,
        uniform_low=args.uniform_low,
        uniform_high=args.uniform_high,
        device=str(device),
        dtype=str(args.dtype),
        attn_backend=args.attn_backend,
    )

    start = time.time()
    _ = generate(
        prompt=args.prompt,
        **components,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        generator=generator,
        noise_sampler=noise_sampler,
        callback=recorder,
        callback_steps=1,
        callback_decode=bool(args.decode),
        callback_decode_steps=args.decode_every,
        output_type="pil",
    )
    elapsed = time.time() - start
    recorder.set_run_config(elapsed_seconds=elapsed)
    recorder.flush()
    print(f"Saved trajectory to: {args.out} (elapsed {elapsed:.2f}s)")


if __name__ == "__main__":
    main()
