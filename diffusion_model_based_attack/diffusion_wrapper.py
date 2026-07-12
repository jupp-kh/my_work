from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


@dataclass(frozen=True)
class DiffusionLoadConfig:
    model_id: str
    device: torch.device
    dtype: torch.dtype
    scheduler: str = "ddim"
    use_safetensors: bool = True
    variant: str | None = None
    attention_slicing: bool = True
    vae_slicing: bool = True
    vae_tiling: bool = False
    gradient_checkpointing: bool = True
    checkpoint_unet: bool = True
    checkpoint_vae: bool = True


class FrozenStableDiffusion:
    """Stable Diffusion wrapper with frozen weights and differentiable latent input."""

    def __init__(self, pipeline: Any, config: DiffusionLoadConfig) -> None:
        self.pipeline = pipeline
        self.config = config
        self.device = config.device
        self.dtype = config.dtype
        self.vae_scale_factor = getattr(pipeline, "vae_scale_factor", 8)

    @classmethod
    def load(cls, config: DiffusionLoadConfig) -> "FrozenStableDiffusion":
        try:
            from diffusers import DDIMScheduler, StableDiffusionPipeline
        except ImportError as exc:
            raise ImportError(
                "diffusers is required. Install the packages in requirements.txt for this directory."
            ) from exc

        load_kwargs: dict[str, object] = {
            "torch_dtype": config.dtype,
            "use_safetensors": config.use_safetensors,
        }
        if config.variant:
            load_kwargs["variant"] = config.variant

        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                config.model_id,
                safety_checker=None,
                requires_safety_checker=False,
                **load_kwargs,
            )
        except TypeError:
            pipeline = StableDiffusionPipeline.from_pretrained(config.model_id, **load_kwargs)
            if hasattr(pipeline, "safety_checker"):
                pipeline.safety_checker = None

        if config.scheduler == "ddim":
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
        elif config.scheduler != "default":
            raise ValueError(f"Unsupported scheduler: {config.scheduler}")

        pipeline.to(config.device)
        for component_name in ("unet", "vae", "text_encoder"):
            component = getattr(pipeline, component_name, None)
            if component is not None:
                component.eval()
                component.requires_grad_(False)

        if config.gradient_checkpointing:
            for component_name in ("unet", "vae"):
                component = getattr(pipeline, component_name, None)
                if component is not None and hasattr(component, "enable_gradient_checkpointing"):
                    component.enable_gradient_checkpointing()

        if config.attention_slicing and hasattr(pipeline, "enable_attention_slicing"):
            pipeline.enable_attention_slicing()
        if config.vae_slicing and hasattr(pipeline, "enable_vae_slicing"):
            pipeline.enable_vae_slicing()
        if config.vae_tiling and hasattr(pipeline, "enable_vae_tiling"):
            pipeline.enable_vae_tiling()

        return cls(pipeline=pipeline, config=config)

    @property
    def trainable_parameter_count(self) -> int:
        total = 0
        for component_name in ("unet", "vae", "text_encoder"):
            component = getattr(self.pipeline, component_name, None)
            if component is not None:
                total += sum(parameter.numel() for parameter in component.parameters() if parameter.requires_grad)
        return total

    def encode_prompt(
        self,
        prompt: str,
        negative_prompt: str,
        batch_size: int,
        guidance_scale: float,
    ) -> torch.Tensor:
        do_classifier_free_guidance = guidance_scale > 1.0
        with torch.no_grad():
            if hasattr(self.pipeline, "encode_prompt"):
                result = self.pipeline.encode_prompt(
                    prompt=prompt,
                    device=self.device,
                    num_images_per_prompt=batch_size,
                    do_classifier_free_guidance=do_classifier_free_guidance,
                    negative_prompt=negative_prompt or None,
                )
                if isinstance(result, tuple):
                    prompt_embeds = result[0]
                    negative_prompt_embeds = result[1] if len(result) > 1 else None
                    if do_classifier_free_guidance:
                        if negative_prompt_embeds is None:
                            raise RuntimeError("Pipeline did not return negative prompt embeddings.")
                        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                else:
                    prompt_embeds = result
            else:
                prompt_embeds = self.pipeline._encode_prompt(
                    prompt,
                    self.device,
                    batch_size,
                    do_classifier_free_guidance,
                    negative_prompt=negative_prompt or None,
                )
        return prompt_embeds.detach().to(device=self.device, dtype=self.dtype)

    def initial_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        seed: int,
    ) -> torch.Tensor:
        if height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0:
            raise ValueError(f"height and width must be divisible by {self.vae_scale_factor}.")
        latent_channels = int(getattr(self.pipeline.unet.config, "in_channels", 4))
        shape = (
            batch_size,
            latent_channels,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        generator = torch.Generator(device=self.device).manual_seed(seed)
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
        sigma = float(getattr(self.pipeline.scheduler, "init_noise_sigma", 1.0))
        return latents * sigma

    def generate(
        self,
        initial_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        guidance_scale: float,
        num_inference_steps: int,
    ) -> torch.Tensor:
        scheduler = self.pipeline.scheduler
        scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = scheduler.timesteps
        do_classifier_free_guidance = guidance_scale > 1.0

        latents = initial_latents.to(device=self.device, dtype=self.dtype)
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self.dtype)

        accepts_eta = "eta" in inspect.signature(scheduler.step).parameters
        for timestep in timesteps:
            latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
            noise_pred = self._predict_noise(latent_model_input, timestep, prompt_embeds)

            if do_classifier_free_guidance:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

            step_kwargs = {"eta": 0.0} if accepts_eta else {}
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False, **step_kwargs)[0]

        return self.decode_latents(latents)

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        scaling_factor = float(getattr(self.pipeline.vae.config, "scaling_factor", 0.18215))
        scaled_latents = latents / scaling_factor
        if self.config.checkpoint_vae and torch.is_grad_enabled() and scaled_latents.requires_grad:
            decoded = checkpoint(self._decode_vae, scaled_latents, use_reentrant=False)
        else:
            decoded = self._decode_vae(scaled_latents)
        images = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)
        return images.float()

    def _predict_noise(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        def unet_forward(latents: torch.Tensor, embeds: torch.Tensor) -> torch.Tensor:
            return self.pipeline.unet(latents, timestep, encoder_hidden_states=embeds).sample

        if self.config.checkpoint_unet and torch.is_grad_enabled() and latent_model_input.requires_grad:
            return checkpoint(unet_forward, latent_model_input, prompt_embeds, use_reentrant=False)
        return unet_forward(latent_model_input, prompt_embeds)

    def _decode_vae(self, scaled_latents: torch.Tensor) -> torch.Tensor:
        return self.pipeline.vae.decode(scaled_latents, return_dict=False)[0]
