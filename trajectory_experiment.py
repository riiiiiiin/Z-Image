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
import regex as re

import torch
import numpy as np
import random

from utils import AttentionBackend, ensure_model_weights, load_from_local_dir, set_attention_backend
from zimage import generate, invert_images_to_init_latents
from zimage.noise import NormalNoiseSampler, UniformNoiseSampler
from zimage.trajectory import TrajectoryRecorder


def select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)
    
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
    
    # watermark
    p.add_argument('--w_seed', default=999999, type=int)
    p.add_argument('--w_channel', nargs='*', type=int, default=[0],
                    help="one or more channel indices, e.g. --w_channel 0 1 2")
    p.add_argument('--w_alpha', nargs='*', type=float, default=[1],
                    help="per-channel alpha values in same order as --w_channel, e.g. --w_alpha 0.03 0.01 0.01 "
                         "If omitted, a default_alpha will be used or broadcast.")
    p.add_argument('--w_pattern', default='rand')
    p.add_argument('--w_mask_shape', default='circle')
    p.add_argument('--w_radius', default=10, type=int)
    p.add_argument('--w_measurement', default='l1_complex')
    p.add_argument('--w_injection', default='complex')
    p.add_argument('--w_pattern_const', default=0, type=float)
    return p.parse_args()


def main():
    args = parse_args()
    
    pattern = re.compile(r"^w_(.+)$")
    watermark_args = {}
    for k, v in vars(args).items():
        m = pattern.match(k)
        if m and v is not None:
            watermark_args[k] = v
            
    print(watermark_args)

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

    set_random_seed(args.seed)
    generator = torch.Generator(device).manual_seed(args.seed)

    def get_noise_sampler(watermark_args):
        if args.noise == "uniform":
            noise_sampler = UniformNoiseSampler(
                low=args.uniform_low,
                high=args.uniform_high,
                scale=args.noise_scale,
                watermark_args=watermark_args,
            )
        else:
            noise_sampler = NormalNoiseSampler(scale=args.noise_scale, watermark_args=watermark_args)

        return noise_sampler
    
    clean_sampler = get_noise_sampler(None)
    watermark_sampler = get_noise_sampler(watermark_args)
    
    clean_path = os.path.join(args.out, "clean")
    watermark_path = os.path.join(args.out, "watermark")
    args.__delattr__("out")
    
    def _generate(path, sampler):
        recorder = TrajectoryRecorder(
            path,
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
        output = generate(
            prompt=args.prompt,
            **components,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=generator,
            noise_sampler=sampler,
            callback=recorder,
            callback_steps=1,
            callback_decode=bool(args.decode),
            callback_decode_steps=args.decode_every,
            output_type="pil",
        )
        elapsed = time.time() - start
        recorder.set_run_config(elapsed_seconds=elapsed)
        recorder.flush()
        print(f"Saved trajectory to: {path} (elapsed {elapsed:.2f}s)")
        
        return output[0]

    clean_image = _generate(clean_path, clean_sampler)
    watermarked_image = _generate(watermark_path, watermark_sampler)
    
    torch._dynamo.reset()
    def _invert(path, image):
        recorder = TrajectoryRecorder(
            path,
            save_latents=True,
            save_images=True,
            save_every=args.save_every,
            save_images_every=args.decode_every,
            keep_only_first_image=True,
        )
        start = time.time()
        output = invert_images_to_init_latents(
            **components,
            images = [image], 
            guidance_scale=args.guidance,
            num_inference_steps=args.steps,
            callback=recorder,
            callback_steps=1,
            callback_decode=bool(args.decode),
            callback_decode_steps=args.decode_every,
        )
        elapsed = time.time() - start
        recorder.set_run_config(elapsed_seconds=elapsed)
        recorder.flush()
        print(f"Saved trajectory to: {path} (elapsed {elapsed:.2f}s)")
    
    clean_init_latent = _invert(clean_path + "_invert", clean_image)
    watermark_init_latent = _invert(watermark_path + "_invert", watermarked_image)

if __name__ == "__main__":
    main()
