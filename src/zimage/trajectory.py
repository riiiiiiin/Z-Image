"""Trajectory recording helpers (callbacks) for Z-Image.

Designed to be plugged into `zimage.generate(callback=..., callback_decode=...)`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def _to_float(x: Any) -> Any:
    if isinstance(x, torch.Tensor) and x.numel() == 1:
        return float(x.detach().cpu().item())
    return x


@dataclass
class TrajectoryRecorder:
    """A callback that saves per-step latents (and optionally decoded images).

    Files written:
      - trajectory.json
      - latents/step_XXXX.pt
      - images/step_XXXX_YY.png (if callback_decode=True)
      - images/final_YY.png (if final images provided)

    Notes:
      - This is IO-heavy if you save every step. Consider save_every>1.
      - Latents are saved as float32 tensors on CPU.
    """

    output_dir: str | Path
    save_latents: bool = True
    save_images: bool = True
    save_every: int = 1
    save_images_every: int = 1
    keep_only_first_image: bool = True
    run_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.latents_dir = self.output_dir / "latents"
        self.images_dir = self.output_dir / "images"
        if self.save_latents:
            self.latents_dir.mkdir(parents=True, exist_ok=True)
        if self.save_images:
            self.images_dir.mkdir(parents=True, exist_ok=True)

        if self.save_every < 1:
            raise ValueError(f"save_every must be >= 1 (got {self.save_every})")
        if self.save_images_every < 1:
            raise ValueError(f"save_images_every must be >= 1 (got {self.save_images_every})")

        self.records: List[Dict[str, Any]] = []

    def set_run_config(self, **kwargs):
        self.run_config.update(kwargs)

    def __call__(self, info: Dict[str, Any]):
        stage = info.get("stage")

        if stage == "step_end":
            step_index = int(info.get("step_index", -1))
            if step_index < 0:
                return

            record: Dict[str, Any] = {
                "stage": "step_end",
                "step_index": step_index,
                "num_inference_steps": int(info.get("num_inference_steps", 0) or 0),
                "t": _to_float(info.get("t")),
                "t_norm": float(info.get("t_norm", 0.0) or 0.0),
                "height": int(info.get("height", 0) or 0),
                "width": int(info.get("width", 0) or 0),
            }

            if self.save_latents and (step_index % self.save_every == 0):
                latents: torch.Tensor = info["latents"].detach().float().cpu()
                latents_path = self.latents_dir / f"step_{step_index:04d}.pt"
                torch.save(latents, latents_path)
                record["latents_path"] = str(latents_path.relative_to(self.output_dir))

            images = info.get("images")
            if self.save_images and images is not None and (step_index % self.save_images_every == 0):
                if self.keep_only_first_image:
                    images = images[:1]
                saved = []
                for j, img in enumerate(images):
                    img_path = self.images_dir / f"step_{step_index:04d}_{j:02d}.png"
                    img.save(img_path)
                    saved.append(str(img_path.relative_to(self.output_dir)))
                record["image_paths"] = saved

            self.records.append(record)
            return

        if stage == "final":
            record: Dict[str, Any] = {
                "stage": "final",
                "output_type": info.get("output_type"),
            }

            if self.save_latents and "latents" in info:
                latents: torch.Tensor = info["latents"].detach().float().cpu()
                latents_path = self.latents_dir / "final.pt"
                torch.save(latents, latents_path)
                record["latents_path"] = str(latents_path.relative_to(self.output_dir))

            images = info.get("images")
            if self.save_images and images is not None:
                if self.keep_only_first_image:
                    images = images[:1]
                saved = []
                for j, img in enumerate(images):
                    img_path = self.images_dir / f"final_{j:02d}.png"
                    img.save(img_path)
                    saved.append(str(img_path.relative_to(self.output_dir)))
                record["image_paths"] = saved

            self.records.append(record)
            self.flush()
            return

    def flush(self):
        payload = {
            "run_config": self.run_config,
            "records": self.records,
        }
        path = self.output_dir / "trajectory.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
