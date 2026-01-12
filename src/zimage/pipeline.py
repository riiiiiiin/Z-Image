"""Z-Image Pipeline."""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger
import torch

from config import (
    BASE_IMAGE_SEQ_LEN,
    BASE_SHIFT,
    DEFAULT_CFG_TRUNCATION,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_WIDTH,
    MAX_IMAGE_SEQ_LEN,
    MAX_SHIFT,
)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = BASE_IMAGE_SEQ_LEN,
    max_seq_len: int = MAX_IMAGE_SEQ_LEN,
    base_shift: float = BASE_SHIFT,
    max_shift: float = MAX_SHIFT,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"The scheduler does not support custom timestep schedules.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"The scheduler does not support custom sigmas schedules.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def _decode_latents_to_pil(vae, latents: torch.Tensor):
    """Decode latents to a list of PIL Images (batch)."""
    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents = (latents.to(vae.dtype) / vae.config.scaling_factor) + shift_factor
    image = vae.decode(latents, return_dict=False)[0]

    from PIL import Image

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255).round().astype("uint8")
    return [Image.fromarray(img) for img in image]

def _encode_pil_to_latents(vae, images):
    """Encode PIL Images to latent vectors (batch)."""
    import numpy as np
    if not isinstance(images, (list, tuple)):
        images = [images]

    device = vae.device
    dtype = vae.dtype

    image = np.stack([np.array(img) for img in images], axis=0)
    image = torch.from_numpy(image).to(device=device, dtype=dtype)

    image = image / 255.0
    image = image * 2.0 - 1.0
    image = image.permute(0, 3, 1, 2)

    posterior = vae.encode(image).latent_dist
    latents = posterior.mode()

    shift_factor = getattr(vae.config, "shift_factor", 0.0) or 0.0
    latents = (latents - shift_factor) * vae.config.scaling_factor

    return latents

@torch.no_grad()
def _predict_single_step(t, timesteps, i, num_inference_steps, latents, transformer, prompt_embeds_list, negative_prompt_embeds_list, actual_batch_size, guidance_scale, do_classifier_free_guidance, cfg_truncation, cfg_normalization):
    # If current t is 0 and it's the last step, skip computation
    if t == 0 and i == len(timesteps) - 1:
        logger.debug(f"Step {i+1}/{num_inference_steps} | t: {t.item():.2f} | Skipping last step")
        return

    timestep = t.expand(latents.shape[0])
    timestep = (1000 - timestep) / 1000
    t_norm = timestep[0].item()

    current_guidance_scale = guidance_scale
    if do_classifier_free_guidance and cfg_truncation is not None and float(cfg_truncation) <= 1:
        if t_norm > cfg_truncation:
            current_guidance_scale = 0.0

    apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0

    if apply_cfg:
        latents_typed = latents.to(
            transformer.dtype if hasattr(transformer, "dtype") else next(transformer.parameters()).dtype
        )
        latent_model_input = latents_typed.repeat(2, 1, 1, 1)
        prompt_embeds_model_input = prompt_embeds_list + negative_prompt_embeds_list
        timestep_model_input = timestep.repeat(2)
    else:
        latent_model_input = latents.to(next(transformer.parameters()).dtype)
        prompt_embeds_model_input = prompt_embeds_list
        timestep_model_input = timestep

    latent_model_input = latent_model_input.unsqueeze(2)
    latent_model_input_list = list(latent_model_input.unbind(dim=0))

    model_out_list = transformer(
        latent_model_input_list,
        timestep_model_input,
        prompt_embeds_model_input,
    )[0]

    if apply_cfg:
        pos_out = model_out_list[:actual_batch_size]
        neg_out = model_out_list[actual_batch_size:]
        noise_pred = []
        for j in range(actual_batch_size):
            pos = pos_out[j].float()
            neg = neg_out[j].float()
            pred = pos + current_guidance_scale * (pos - neg)

            if cfg_normalization and float(cfg_normalization) > 0.0:
                ori_pos_norm = torch.linalg.vector_norm(pos)
                new_pos_norm = torch.linalg.vector_norm(pred)
                max_new_norm = ori_pos_norm * float(cfg_normalization)
                if new_pos_norm > max_new_norm:
                    pred = pred * (max_new_norm / new_pos_norm)
            noise_pred.append(pred)
        noise_pred = torch.stack(noise_pred, dim=0)
    else:
        noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

    noise_pred = -noise_pred.squeeze(2)
    return noise_pred.to(torch.float32), t_norm
    

@torch.no_grad()
def generate(
    transformer,
    vae,
    text_encoder,
    tokenizer,
    scheduler,
    prompt: Union[str, List[str]],
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    num_inference_steps: int = DEFAULT_INFERENCE_STEPS,
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: int = 1,
    generator: Optional[torch.Generator] = None,
    cfg_normalization: bool = False,
    cfg_truncation: float = DEFAULT_CFG_TRUNCATION,
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH,
    output_type: str = "pil",
    # ---- experiment / hooks ----
    initial_latents: Optional[torch.Tensor] = None,
    noise_sampler: Optional[
        Callable[[tuple, Optional[torch.Generator], Union[str, torch.device], torch.dtype], torch.Tensor]
    ] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    callback_steps: int = 1,
    callback_decode: bool = False,
    callback_decode_steps: int = 1,
):
    device = next(transformer.parameters()).device

    if hasattr(vae, "config") and hasattr(vae.config, "block_out_channels"):
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    else:
        vae_scale_factor = 8
    vae_scale = vae_scale_factor * 2

    if height % vae_scale != 0:
        raise ValueError(f"Height must be divisible by {vae_scale} (got {height}).")
    if width % vae_scale != 0:
        raise ValueError(f"Width must be divisible by {vae_scale} (got {width}).")

    if isinstance(prompt, str):
        batch_size = 1
        prompt = [prompt]
    else:
        batch_size = len(prompt)

    do_classifier_free_guidance = guidance_scale > 1.0
    logger.info(f"Generating image: {height}x{width}, steps={num_inference_steps}, cfg={guidance_scale}")

    formatted_prompts = []
    for p in prompt:
        messages = [{"role": "user", "content": p}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        formatted_prompts.append(formatted_prompt)

    text_inputs = tokenizer(
        formatted_prompts,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids.to(device)
    prompt_masks = text_inputs.attention_mask.to(device).bool()

    prompt_embeds = text_encoder(
        input_ids=text_input_ids,
        attention_mask=prompt_masks,
        output_hidden_states=True,
    ).hidden_states[-2]

    prompt_embeds_list = []
    for i in range(len(prompt_embeds)):
        prompt_embeds_list.append(prompt_embeds[i][prompt_masks[i]])

    negative_prompt_embeds_list = []
    if do_classifier_free_guidance:
        if negative_prompt is None:
            negative_prompt = ["" for _ in prompt]
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]

        neg_formatted = []
        for p in negative_prompt:
            messages = [{"role": "user", "content": p}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            neg_formatted.append(formatted_prompt)

        neg_inputs = tokenizer(
            neg_formatted,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        neg_input_ids = neg_inputs.input_ids.to(device)
        neg_masks = neg_inputs.attention_mask.to(device).bool()

        neg_embeds = text_encoder(
            input_ids=neg_input_ids,
            attention_mask=neg_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        for i in range(len(neg_embeds)):
            negative_prompt_embeds_list.append(neg_embeds[i][neg_masks[i]])

    if num_images_per_prompt > 1:
        prompt_embeds_list = [pe for pe in prompt_embeds_list for _ in range(num_images_per_prompt)]
        if do_classifier_free_guidance:
            negative_prompt_embeds_list = [
                npe for npe in negative_prompt_embeds_list for _ in range(num_images_per_prompt)
            ]

    height_latent = 2 * (int(height) // vae_scale)
    width_latent = 2 * (int(width) // vae_scale)
    shape = (batch_size * num_images_per_prompt, transformer.in_channels, height_latent, width_latent)

    if callback_steps < 1:
        raise ValueError(f"callback_steps must be >= 1 (got {callback_steps}).")
    if callback_decode_steps < 1:
        raise ValueError(f"callback_decode_steps must be >= 1 (got {callback_decode_steps}).")

    if initial_latents is not None:
        latents = initial_latents.to(device=device, dtype=torch.float32)
        if tuple(latents.shape) != tuple(shape):
            raise ValueError(
                f"initial_latents shape mismatch: expected {shape}, got {tuple(latents.shape)}. "
                "(Tip: height/width/vae_scale determine latent spatial size.)"
            )
    else:
        if noise_sampler is None:
            latents = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        else:
            latents = noise_sampler(shape, generator, device, torch.float32)

    actual_batch_size = batch_size * num_images_per_prompt
    image_seq_len = (shape[2] // 2) * (shape[3] // 2)

    mu = calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    scheduler_kwargs = {"mu": mu}
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=None,
        **scheduler_kwargs,
    )

    logger.info(f"Sampling loop start: {num_inference_steps} steps")

    from tqdm import tqdm

    # Denoising loop with progress bar
    for i, t in enumerate(tqdm(timesteps, desc="Denoising", total=len(timesteps))):
        noise_pred, t_norm = _predict_single_step(t, timesteps, i, num_inference_steps, latents, transformer, prompt_embeds_list, negative_prompt_embeds_list, actual_batch_size, guidance_scale, do_classifier_free_guidance, cfg_truncation, cfg_normalization)
        
        latents = scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
        assert latents.dtype == torch.float32

        if callback is not None and (i % callback_steps == 0 or i == len(timesteps) - 1):
            payload: Dict[str, Any] = {
                "stage": "step_end",
                "step_index": i,
                "num_inference_steps": num_inference_steps,
                "t": t,
                "t_norm": t_norm,
                "latents": latents,
                "noise_pred": noise_pred,
                "height": height,
                "width": width,
            }
            if callback_decode and (i % callback_decode_steps == 0 or i == len(timesteps) - 1):
                payload["images"] = _decode_latents_to_pil(vae, latents)
            callback(payload)

    if output_type == "latent":
        if callback is not None:
            callback({"stage": "final", "latents": latents, "output_type": "latent"})
        return latents

    if output_type == "pil":
        image = _decode_latents_to_pil(vae, latents)
        if callback is not None:
            callback({"stage": "final", "latents": latents, "images": image, "output_type": "pil"})
        return image

    raise ValueError(f"Unsupported output_type: {output_type}")

@torch.no_grad()
def _invert_single_step(
    scheduler,
    transformer,
    s_next: torch.Tensor,                # sample after forward step, shape (B,C,H,W)
    timestep: torch.Tensor,              # the same timestep value used during forward (element of scheduler.timesteps)
    prompt_embeds_list: List,
    negative_prompt_embeds_list: Optional[List] = None,
    guidance_scale: float = 1.0,
    cfg_normalization: Optional[float] = None,
    num_fixed_point_iters: int = 8,
    tol: Optional[float] = None,
    device: Optional[torch.device] = None,
):
    """
    Approximate inversion of a single forward step of FlowMatchEulerDiscreteScheduler:
        forward: s_next = s_prev + dt * model_output(s_prev, t)
    We solve for s_prev given s_next and timestep t using Picard iterations:
        s_prev^{n+1} = s_next - dt * model_output(s_prev^{n}, t)

    Returns: s_prev_estimate (same shape & dtype as s_next)
    """
    device = device or next(transformer.parameters()).device
    s_next = s_next.to(device)

    # locate index for given timestep in scheduler -- prefer exact match, fallback to nearest
    try:
        sigma_idx = scheduler.index_for_timestep(timestep, schedule_timesteps=scheduler.timesteps)
    except Exception:
        # fallback: find nearest
        sched = scheduler.timesteps.to(device)
        # ensure timestep is tensor on same device
        t_val = timestep.to(device) if torch.is_tensor(timestep) else torch.tensor(float(timestep), device=device)
        diffs = torch.abs(sched - t_val)
        sigma_idx = int(torch.argmin(diffs).item())

    # get dt consistent with scheduler.step (sigma_next - sigma)
    sigmas = scheduler.sigmas.to(device)
    # Ensure sigma_idx + 1 exists
    if sigma_idx + 1 >= sigmas.shape[0]:
        raise ValueError("Invalid sigma index for inversion (no sigma_next).")
    sigma = sigmas[sigma_idx]
    sigma_next = sigmas[sigma_idx + 1]
    dt = (sigma_next - sigma).to(torch.float32).to(device)

    # initialize estimate for s_prev as s_next (reasonable starting point)
    s_prev = s_next.clone()

    for it in range(num_fixed_point_iters):
        # compute model_output at current s_prev estimate
        model_out, _ = _predict_single_step(
            transformer,
            s_prev,
            timestep,
            prompt_embeds_list,
            negative_prompt_embeds_list,
            guidance_scale,
            cfg_normalization,
        )
        # Picard update: s_prev <- s_next - dt * model_out(s_prev)
        s_new = s_next - dt * model_out

        # check tolerance if provided
        if tol is not None:
            diff = torch.max(torch.abs(s_new - s_prev))
            s_prev = s_new
            if diff.item() <= tol:
                break
        else:
            s_prev = s_new

    # returned dtype: convert to scheduler/model dtype if necessary (keep float32 to be safe)
    return s_prev.to(s_next.dtype)

@torch.no_grad()
def invert_images_to_init_latents(
    transformer,
    vae,
    scheduler,
    images,  # accepts list/sequence of PIL images OR a batched image tensor depending on your _encode_pil_to_latents impl
    prompt_embeds_list: List,
    negative_prompt_embeds_list: Optional[List] = None,
    guidance_scale: float = 1.0,
    cfg_normalization: Optional[float] = None,
    num_inference_steps: Optional[int] = None,  # if provided, will call scheduler.set_timesteps(...) to match generation
    num_fixed_point_iters: int = 8,
    relax_alpha: Optional[float] = None,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    callback_steps: int = 1,
    callback_decode: bool = False,  # not used here, but kept for API parity
    callback_decode_steps: int = 1,
    device: Optional[torch.device] = None,
):
    """
    End-to-end flow:
      images -> encode via _encode_pil_to_latents(vae, images) -> treat as s_N (final latents)
      -> iterate scheduler.timesteps in reverse, calling invert_one_step_flowmatch_euler per step
      -> return estimated init_latents (s_0)

    Logging: uses callback(payload) similarly to pipeline.generate's callback.
    """
    device = device or next(transformer.parameters()).device

    # 1) (optional) set timesteps on scheduler to match generation config
    if num_inference_steps is not None:
        scheduler.set_timesteps(num_inference_steps, device=device)

    timesteps = scheduler.timesteps  # tensor of timesteps
    sigmas = scheduler.sigmas
    n_steps = len(timesteps)

    # 2) encode images -> final latents (s_N)
    # Expected: _encode_pil_to_latents returns latents in SAME representation as pipeline's `latents`
    pipeline_latents = _encode_pil_to_latents(vae, images).to(device).to(torch.float32)

    current_latents = pipeline_latents  # s_{N}

    # prepare some info for callback
    height = images[0].size[1] if hasattr(images[0], "size") else None
    width = images[0].size[0] if hasattr(images[0], "size") else None

    from tqdm import tqdm
    # reverse iterate timesteps: from last index down to 0
    # note: pipeline's forward loop iterated timesteps in increasing order (sigmas index increases)
    for idx in tqdm(range(n_steps - 1, -1, -1), desc="Inversion (reverse steps)", total=n_steps):
        t = timesteps[idx].to(device)

        if t == 0 and idx == n_steps - 1:
            # log skip
            if callback is not None and ( (n_steps - 1 - idx) % callback_steps == 0 or idx == 0):
                payload = {
                    "stage": "step_end",
                    "step_index": n_steps - 1 - idx,
                    "num_inference_steps": n_steps,
                    "t": t,
                    "t_norm": ((1000.0 - t.item()) / 1000.0) if torch.is_tensor(t) else None,
                    "latents": current_latents,
                    "noise_pred": None,
                    "height": height,
                    "width": width,
                }
                callback(payload)
            continue

        # perform single-step inversion (parallel batch)
        s_prev, last_model_out = _invert_single_step(
            scheduler=scheduler,
            transformer=transformer,
            s_next=current_latents,
            timestep=t,
            prompt_embeds_list=prompt_embeds_list,
            negative_prompt_embeds_list=negative_prompt_embeds_list,
            guidance_scale=guidance_scale,
            cfg_normalization=cfg_normalization,
            num_fixed_point_iters=num_fixed_point_iters,
            device=device,
            relax_alpha=relax_alpha,
        )

        # For logging: compute t_norm same way pipeline did
        timestep_expanded = t.expand(current_latents.shape[0])
        t_norm = float(((1000.0 - timestep_expanded[0].item()) / 1000.0))

        # prepare payload for callback (mimic generate's payload)
        if callback is not None and ((n_steps - 1 - idx) % callback_steps == 0 or idx == 0):
            payload: Dict[str, Any] = {
                "stage": "step_end",  # pipeline used "step_end"
                "step_index": n_steps - 1 - idx,
                "num_inference_steps": n_steps,
                "t": t,
                "t_norm": t_norm,
                "latents": s_prev,           # the estimated latents at this reversed step (s_prev)
                "noise_pred": last_model_out,  # model_output evaluated at final iterate
                "height": height,
                "width": width,
            }
            # optionally decode for debug if user wants (uses _decode_latents_to_pil if available)
            if callback_decode and ((n_steps - 1 - idx) % callback_decode_steps == 0 or idx == 0):
                try:
                    from .pipeline import _decode_latents_to_pil  # same helper used in generate
                    payload["images"] = _decode_latents_to_pil(vae, s_prev)
                except Exception:
                    # graceful: skip decode if unavailable
                    payload["images"] = None
            callback(payload)

        current_latents = s_prev  # set for next reverse step

    # After full reverse pass, current_latents approx equals s_0 (init latents)
    if callback is not None:
        callback({"stage": "final", "latents": current_latents, "output_type": "latent"})

    return current_latents