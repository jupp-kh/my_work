from __future__ import annotations

import inspect
from contextlib import nullcontext
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

try:
    from .text_embedding import PlaceholderEmbeddingManager, build_placeholder_tokens, expand_prompt
except ImportError:  # pragma: no cover - script execution fallback.
    from text_embedding import PlaceholderEmbeddingManager, build_placeholder_tokens, expand_prompt


def choose_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def resolve_dtype(mixed_precision: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if mixed_precision == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if mixed_precision == "no":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed precision setting: {mixed_precision}")


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


class SDXLTurboEmbeddingExperiment:
    def __init__(self, args: Any, device: torch.device, dtype: torch.dtype) -> None:
        try:
            from diffusers import StableDiffusionXLPipeline
        except ImportError as exc:  # pragma: no cover - exercised in real environment.
            raise ImportError("diffusers is required. Install requirements.txt in this directory.") from exc

        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "use_safetensors": not args.no_safetensors,
        }
        if args.diffusion_variant:
            load_kwargs["variant"] = args.diffusion_variant
        self.pipe = StableDiffusionXLPipeline.from_pretrained(args.model_id, **load_kwargs)
        self.pipe.to(device)
        self.device = device
        self.dtype = dtype
        self.args = args
        self.vae_scale_factor = int(getattr(self.pipe, "vae_scale_factor", 8))
        self.decode_dtype = torch.float32 if args.vae_float32 else dtype
        self.last_generation_diagnostics: dict[str, bool] = {}

        for component_name in ("unet", "vae", "text_encoder", "text_encoder_2"):
            component = getattr(self.pipe, component_name, None)
            if component is not None:
                component.eval()
                component.requires_grad_(False)

        if args.vae_float32:
            self.pipe.vae.to(device=device, dtype=torch.float32)

        if args.gradient_checkpointing:
            for component_name in ("unet", "vae"):
                component = getattr(self.pipe, component_name, None)
                if component is not None and hasattr(component, "enable_gradient_checkpointing"):
                    component.enable_gradient_checkpointing()
        if args.attention_slicing and hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        if args.vae_slicing and hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        if args.vae_tiling and hasattr(self.pipe, "enable_vae_tiling"):
            self.pipe.enable_vae_tiling()

        self.placeholder_tokens = build_placeholder_tokens(args.placeholder_token, args.num_vectors)
        expanded_prompt = expand_prompt(args.prompt, args.placeholder_token, self.placeholder_tokens)
        if expanded_prompt != args.prompt:
            print(f"Expanded placeholder prompt: {expanded_prompt}")
        self.prompt = expanded_prompt

        optimize_first = args.text_encoder_selection in {"first", "both"}
        optimize_second = args.text_encoder_selection in {"second", "both"}
        self.encoder_1 = PlaceholderEmbeddingManager(
            name="text_encoder",
            tokenizer=self.pipe.tokenizer,
            text_encoder=self.pipe.text_encoder,
            placeholder_tokens=self.placeholder_tokens,
            initializer_token=args.initializer_token,
            trainable=optimize_first,
            device=device,
        )
        self.encoder_2 = PlaceholderEmbeddingManager(
            name="text_encoder_2",
            tokenizer=self.pipe.tokenizer_2,
            text_encoder=self.pipe.text_encoder_2,
            placeholder_tokens=self.placeholder_tokens,
            initializer_token=args.initializer_token,
            trainable=optimize_second,
            device=device,
        )
        self.managers = [self.encoder_1, self.encoder_2]
        self.trainable_parameters = [parameter for manager in self.managers for parameter in manager.parameters()]
        if not self.trainable_parameters:
            raise ValueError("No placeholder embeddings selected for optimization.")

    @property
    def frozen_trainable_parameter_count(self) -> int:
        total = 0
        for component_name in ("unet", "vae", "text_encoder", "text_encoder_2"):
            component = getattr(self.pipe, component_name, None)
            if component is not None:
                total += sum(
                    parameter.numel()
                    for name, parameter in component.named_parameters()
                    if parameter.requires_grad and "token_embedding" not in name
                )
        return total

    def embedding_state(self) -> dict[str, Any]:
        return {
            "placeholder_token": self.args.placeholder_token,
            "placeholder_tokens": self.placeholder_tokens,
            "text_encoder_selection": self.args.text_encoder_selection,
            "encoder_1": self.encoder_1.state_dict(),
            "encoder_2": self.encoder_2.state_dict(),
        }

    def load_embedding_state(self, state: dict[str, Any]) -> None:
        if "encoder_1" in state and "embedding" in state["encoder_1"]:
            self.encoder_1.load_embedding(state["encoder_1"]["embedding"])
        if "encoder_2" in state and "embedding" in state["encoder_2"]:
            self.encoder_2.load_embedding(state["encoder_2"]["embedding"])

    def embedding_regularization(self) -> torch.Tensor:
        losses = [manager.regularization() for manager in self.managers if manager.parameter is not None]
        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def embedding_norm(self) -> float:
        norms = [manager.norm().detach() for manager in self.managers if manager.parameter is not None]
        if not norms:
            return 0.0
        return float(torch.linalg.vector_norm(torch.stack(norms)).cpu())

    def embedding_displacement(self) -> float:
        displacements = [manager.displacement().detach() for manager in self.managers if manager.parameter is not None]
        if not displacements:
            return 0.0
        return float(torch.linalg.vector_norm(torch.stack(displacements)).cpu())

    def encode_prompt(self, prompt: str, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_embeds_list = []
        pooled_prompt_embeds = None
        pairs = [
            (self.pipe.tokenizer, self.pipe.text_encoder),
            (self.pipe.tokenizer_2, self.pipe.text_encoder_2),
        ]
        for tokenizer, text_encoder in pairs:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(self.device)
            outputs = text_encoder(
                input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            prompt_embeds_list.append(outputs.hidden_states[-2])
            pooled_prompt_embeds = outputs[0]
        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        if pooled_prompt_embeds is None:
            raise RuntimeError("SDXL text encoder did not return pooled prompt embeddings.")
        prompt_embeds = prompt_embeds.repeat(batch_size, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat(batch_size, 1)
        return (
            prompt_embeds.to(device=self.device, dtype=self.dtype),
            pooled_prompt_embeds.to(device=self.device, dtype=self.dtype),
        )

    def add_time_ids(self, batch_size: int) -> torch.Tensor:
        original_size = (self.args.height, self.args.width)
        target_size = (self.args.height, self.args.width)
        crops_coords_top_left = (0, 0)
        projection_dim = int(getattr(self.pipe.text_encoder_2.config, "projection_dim", 1280))
        add_time_ids = self.pipe._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=self.dtype,
            text_encoder_projection_dim=projection_dim,
        )
        return add_time_ids.to(self.device).repeat(batch_size, 1)

    def initial_latents(self, num_samples: int, seed: int) -> torch.Tensor:
        if self.args.height % self.vae_scale_factor != 0 or self.args.width % self.vae_scale_factor != 0:
            raise ValueError(f"height and width must be divisible by {self.vae_scale_factor}.")
        generator = torch.Generator(device=self.device).manual_seed(seed)
        channels = int(getattr(self.pipe.unet.config, "in_channels", 4))
        shape = (
            num_samples,
            channels,
            self.args.height // self.vae_scale_factor,
            self.args.width // self.vae_scale_factor,
        )
        latents = torch.randn(shape, generator=generator, device=self.device, dtype=self.dtype)
        return latents * float(getattr(self.pipe.scheduler, "init_noise_sigma", 1.0))

    def generate(self, fixed_latents: torch.Tensor) -> torch.Tensor:
        self.last_generation_diagnostics = {}
        batch_size = fixed_latents.shape[0]
        prompt_embeds, pooled_prompt_embeds = self.encode_prompt(self.prompt, batch_size=batch_size)
        add_time_ids = self.add_time_ids(batch_size=batch_size)

        do_classifier_free_guidance = self.args.guidance_scale > 1.0
        if do_classifier_free_guidance:
            negative_prompt = self.args.negative_prompt or ""
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompt(
                negative_prompt,
                batch_size=batch_size,
            )
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

        scheduler = self.pipe.scheduler
        scheduler.set_timesteps(self.args.inference_steps, device=self.device)
        latents = fixed_latents.to(device=self.device, dtype=self.dtype)
        accepts_eta = "eta" in inspect.signature(scheduler.step).parameters
        added_cond_kwargs = {"text_embeds": pooled_prompt_embeds, "time_ids": add_time_ids}

        for timestep in scheduler.timesteps:
            latent_model_input = torch.cat([latents, latents], dim=0) if do_classifier_free_guidance else latents
            latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
            self.last_generation_diagnostics["latent_model_input_finite"] = bool(
                torch.isfinite(latent_model_input.detach()).all().cpu()
            )
            noise_pred = self.predict_noise(
                latent_model_input=latent_model_input,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                add_time_ids=add_time_ids,
            )
            self.last_generation_diagnostics["noise_pred_finite"] = bool(torch.isfinite(noise_pred.detach()).all().cpu())
            if do_classifier_free_guidance:
                noise_uncond, noise_text = noise_pred.chunk(2)
                noise_pred = noise_uncond + self.args.guidance_scale * (noise_text - noise_uncond)
            step_kwargs = {"eta": 0.0} if accepts_eta else {}
            latents = scheduler.step(noise_pred, timestep, latents, return_dict=False, **step_kwargs)[0]
            self.last_generation_diagnostics["latents_after_step_finite"] = bool(
                torch.isfinite(latents.detach()).all().cpu()
            )

        scaling_factor = float(getattr(self.pipe.vae.config, "scaling_factor", 0.18215))
        scaled_latents = (latents / scaling_factor).to(device=self.device, dtype=self.decode_dtype)
        self.last_generation_diagnostics["scaled_latents_finite"] = bool(torch.isfinite(scaled_latents.detach()).all().cpu())
        decoded = self.decode_latents(scaled_latents)
        self.last_generation_diagnostics["decoded_finite"] = bool(torch.isfinite(decoded.detach()).all().cpu())
        images = decoded.float() / 2.0 + 0.5
        self.last_generation_diagnostics["images_finite"] = bool(torch.isfinite(images.detach()).all().cpu())
        return images

    def predict_noise(
        self,
        *,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        add_time_ids: torch.Tensor,
    ) -> torch.Tensor:
        def unet_forward(
            latent_input: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            text_embeds: torch.Tensor,
            time_ids: torch.Tensor,
        ) -> torch.Tensor:
            return self.pipe.unet(
                latent_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                return_dict=False,
            )[0]

        if self.args.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(
                unet_forward,
                latent_model_input,
                prompt_embeds,
                pooled_prompt_embeds,
                add_time_ids,
                use_reentrant=False,
            )
        return unet_forward(latent_model_input, prompt_embeds, pooled_prompt_embeds, add_time_ids)

    def decode_latents(self, scaled_latents: torch.Tensor) -> torch.Tensor:
        def vae_forward(latents: torch.Tensor) -> torch.Tensor:
            return self.pipe.vae.decode(latents, return_dict=False)[0]

        should_checkpoint = (
            self.args.gradient_checkpointing
            and torch.is_grad_enabled()
            and bool(scaled_latents.requires_grad)
        )
        if self.decode_dtype == torch.float32 and self.device.type == "cuda":
            with torch.autocast(device_type="cuda", enabled=False):
                if should_checkpoint:
                    return checkpoint(vae_forward, scaled_latents, use_reentrant=False)
                return vae_forward(scaled_latents)
        if should_checkpoint:
            return checkpoint(vae_forward, scaled_latents, use_reentrant=False)
        return vae_forward(scaled_latents)
