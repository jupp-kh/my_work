from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from .image_utils import resize_tensor, save_tensor_image
except ImportError:
    from image_utils import resize_tensor, save_tensor_image  # type: ignore


PROMPT_TEMPLATE_VERSION = 3


@dataclass(frozen=True)
class PromptSearchConfig:
    base_prompt: str
    negative_prompt: str
    target_label: str
    num_prompts: int
    seed: int
    generator_model: str
    cache_dir: Path
    search_space: str = "broad"
    force_regenerate: bool = False


@dataclass(frozen=True)
class PromptScore:
    prompt: str
    score: float
    hash_l1: float
    image_mse: float | None


@dataclass(frozen=True)
class PromptSearchResult:
    selected_prompt: str
    selected_score: PromptScore
    prompt_cache: Path
    score_cache: Path
    scores: list[PromptScore]


def _stable_key(data: dict[str, Any]) -> str:
    encoded = json.dumps(data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _prompt_cache_path(config: PromptSearchConfig) -> Path:
    key = _stable_key(
        {
            "base_prompt": config.base_prompt,
            "negative_prompt": config.negative_prompt,
            "target_label": config.target_label,
            "num_prompts": config.num_prompts,
            "seed": config.seed,
            "generator_model": config.generator_model,
            "search_space": config.search_space,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        }
    )
    return config.cache_dir / f"prompts_{key}.json"


def _score_cache_path(
    prompt_cache_path: Path,
    diffusion_model: str,
    height: int,
    width: int,
    diffusion_steps: int,
    guidance_scale: float,
    hash_weight: float,
    image_weight: float,
) -> Path:
    key = _stable_key(
        {
            "prompt_cache": prompt_cache_path.name,
            "diffusion_model": diffusion_model,
            "height": height,
            "width": width,
            "diffusion_steps": diffusion_steps,
            "guidance_scale": guidance_scale,
            "hash_weight": hash_weight,
            "image_weight": image_weight,
        }
    )
    return prompt_cache_path.with_name(f"scores_{key}.json")


def generate_or_load_prompt_candidates(config: PromptSearchConfig) -> tuple[list[str], Path]:
    config.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _prompt_cache_path(config)
    if cache_path.exists() and not config.force_regenerate:
        with open(cache_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        prompts = [str(prompt) for prompt in data["prompts"]]
        if len(prompts) == config.num_prompts:
            return prompts, cache_path

    prompts = _generate_prompt_candidates(config)
    payload = {"config": _jsonable_config(config), "prompts": prompts}
    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return prompts, cache_path


def _jsonable_config(config: PromptSearchConfig) -> dict[str, Any]:
    data = asdict(config)
    data["cache_dir"] = str(config.cache_dir)
    return data


def _generate_prompt_candidates(config: PromptSearchConfig) -> list[str]:
    prompts = [config.base_prompt]
    if config.generator_model and config.generator_model != "templates":
        prompts.extend(_generate_with_light_text_model(config))
    prompts.extend(_template_prompts(config.base_prompt, config.num_prompts, config.seed, config.search_space))

    deduped: list[str] = []
    seen = set()
    for prompt in prompts:
        cleaned = " ".join(prompt.strip().split())
        if cleaned and cleaned.lower() not in seen:
            deduped.append(cleaned)
            seen.add(cleaned.lower())
        if len(deduped) >= config.num_prompts:
            break
    return deduped


def _generate_with_light_text_model(config: PromptSearchConfig) -> list[str]:
    try:
        from transformers import pipeline
    except ImportError:
        return []

    try:
        task = _generation_task_for_model(config.generator_model)
        generator = pipeline(task, model=config.generator_model, device=-1)
        request = _prompt_generation_request(config)
        outputs = generator(
            request,
            max_new_tokens=64,
            num_return_sequences=max(config.num_prompts - 1, 1),
            do_sample=True,
            temperature=0.8,
        )
    except Exception:
        return []

    prompts = []
    for output in outputs:
        text = str(output.get("generated_text", ""))
        prompts.extend(_extract_generated_prompt_candidates(text, request))
    return prompts


def _generation_task_for_model(model_id: str) -> str:
    lowered = model_id.lower()
    if any(name in lowered for name in ("t5", "flan", "bart", "pegasus")):
        return "text2text-generation"
    return "text-generation"


def _prompt_generation_request(config: PromptSearchConfig) -> str:
    if config.search_space == "broad":
        return (
            "Generate image prompts for a diffusion model.\n"
            "Return one prompt per line. Do not number the lines. Do not use a repeated prefix.\n"
            "Each prompt should describe a different everyday scene: food, airplane, dog, street, room, vehicle, landscape, sport, tool, or object.\n"
            "Do not stay near this base concept: "
            f"{config.base_prompt}\n"
            "Prompts:"
        )
    return (
        "Generate image prompts for a diffusion model.\n"
        "Return one prompt per line. Do not number the lines. Do not use a repeated prefix.\n"
        "Keep the same main idea but vary camera view, setting, lighting, and composition.\n"
        f"Base concept: {config.base_prompt}\n"
        "Prompts:"
    )


def _extract_generated_prompt_candidates(text: str, request: str) -> list[str]:
    generated = text[len(request) :] if text.startswith(request) else text
    generated = generated.split("Prompts:")[-1]
    candidates = []
    for line in generated.replace(";", "\n").splitlines():
        cleaned = line.strip().strip("\"'")
        cleaned = cleaned.lstrip("-*0123456789. )\t").strip()
        if cleaned:
            candidates.append(cleaned)
    if candidates:
        return candidates
    cleaned = generated.strip().strip("\"'")
    return [cleaned] if cleaned else []


def _template_prompts(base_prompt: str, num_prompts: int, seed: int, search_space: str) -> list[str]:
    if search_space == "prompt":
        return _prompt_near_template_prompts(base_prompt, num_prompts, seed)
    if search_space == "broad":
        return _broad_attack_template_prompts(base_prompt, num_prompts, seed)
    raise ValueError(f"Unsupported prompt search space: {search_space}")


def _prompt_near_template_prompts(base_prompt: str, num_prompts: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    variants = [
        "{prompt} in natural window light.",
        "Close-up view of {prompt}.",
        "Overhead view of {prompt} on a simple background.",
        "{prompt} with the main subject fully visible.",
        "{prompt} with balanced colors and a clean composition.",
        "{prompt} in a single coherent scene.",
    ]
    prompts = [base_prompt]
    while len(prompts) < num_prompts:
        template = rng.choice(variants)
        prompts.append(template.format(prompt=base_prompt))
    return prompts


def _broad_attack_template_prompts(base_prompt: str, num_prompts: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    scene_prompts = [
        "Bento lunch box filled with vegetables and rice on a wooden table.",
        "Passenger airplane parked on an airport runway at sunset.",
        "Friendly dog sitting on green grass in a city park.",
        "Red sports car parked on a wet downtown street.",
        "Cozy bedroom with a made bed and wooden nightstand.",
        "Busy street market with fruit stalls and people walking.",
        "Mountain lake with pine trees reflected in the water.",
        "Office desk with a laptop, notebook, coffee mug, and lamp.",
        "Bicycle leaning against a brick wall on a quiet street.",
        "Kitchen counter with vegetables, a cutting board, and bowls.",
        "Train station platform with a silver commuter train.",
        "Beach with umbrellas, towels, and blue ocean water.",
        "Bookshelf beside a comfortable reading chair.",
        "Soccer ball on a green field under stadium lights.",
        "Small camping tent beside a forest trail.",
        "Colorful toy robot on a child's playroom floor.",
        "Hot air balloons floating above farmland.",
        "Glass vase of fresh flowers on a dining table.",
        "Harbor with fishing boats and calm water.",
        "White ceramic teapot and cups on a patterned tablecloth.",
        "City bus stopped beside a sidewalk.",
        "Wooden toolbox with hand tools on a workshop bench.",
        "Grocery basket filled with fruit, bread, and bottled drinks.",
        "Snow-covered cabin with warm window light in a forest.",
        "Colorful sneakers on pavement.",
        "White coffee cup beside an open book.",
        "Wooden dining table set for breakfast.",
        "Motorcycle parked near a road at sunset.",
        "Living room with a sofa, rug, and houseplants.",
        "Yellow school bus in front of a school building.",
        "Fruit bowl with apples, bananas, oranges, and grapes.",
        "Sailboat on blue water near the shore.",
        "Garden path lined with flowers.",
        "Backpack and hiking boots near a trail map.",
        "Bakery counter filled with bread and pastries.",
        "Basketball hoop on an outdoor court.",
        "Hotel room with a bed, desk, and window.",
        "Camera on a table beside printed photographs.",
        "Farm tractor parked beside a field.",
        "Bowl of soup with bread on a kitchen table.",
    ]

    shuffled_prompts = scene_prompts[:]
    rng.shuffle(shuffled_prompts)
    prompts = [base_prompt]
    index = 0
    while len(prompts) < num_prompts:
        if index > 0 and index % len(shuffled_prompts) == 0:
            rng.shuffle(shuffled_prompts)
        prompt = shuffled_prompts[index % len(shuffled_prompts)]
        if index >= len(scene_prompts):
            prompt = prompt[:-1] + " in clear natural light."
        prompts.append(prompt)
        index += 1
    return prompts


def score_prompt_candidates(
    prompts: list[str],
    prompt_cache_path: Path,
    diffusion: Any,
    certphash: Any,
    target_hash: Any,
    target_image: torch.Tensor | None,
    negative_prompt: str,
    batch_size: int,
    height: int,
    width: int,
    seed: int,
    guidance_scale: float,
    diffusion_steps: int,
    diffusion_model: str,
    hash_weight: float,
    image_weight: float,
    run_dir: Path,
    force_rescore: bool = False,
) -> PromptSearchResult:
    score_cache = _score_cache_path(
        prompt_cache_path,
        diffusion_model,
        height,
        width,
        diffusion_steps,
        guidance_scale,
        hash_weight,
        image_weight,
    )
    if score_cache.exists() and not force_rescore:
        with open(score_cache, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        scores = [PromptScore(**item) for item in data["scores"]]
        selected = PromptScore(**data["selected_score"])
        return PromptSearchResult(
            selected_prompt=str(data["selected_prompt"]),
            selected_score=selected,
            prompt_cache=prompt_cache_path,
            score_cache=score_cache,
            scores=scores,
        )

    scores: list[PromptScore] = []
    best_image = None
    best_score: PromptScore | None = None
    for index, prompt in enumerate(prompts):
        with torch.no_grad():
            prompt_embeds = diffusion.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                batch_size=batch_size,
                guidance_scale=guidance_scale,
            )
            latents = diffusion.initial_latents(
                batch_size=batch_size,
                height=height,
                width=width,
                seed=seed,
            )
            images = diffusion.generate(
                initial_latents=latents,
                prompt_embeds=prompt_embeds,
                guidance_scale=guidance_scale,
                num_inference_steps=diffusion_steps,
            )
            quantized = certphash.quantized_hash(images)
            hash_l1 = float(certphash.l1_distance_per_sample(quantized, target_hash.quantized).min().cpu())
            image_mse = _image_mse(images, target_image)
            score = hash_weight * hash_l1
            if image_mse is not None:
                score += image_weight * image_mse * 144.0 * 255.0
            prompt_score = PromptScore(
                prompt=prompt,
                score=float(score),
                hash_l1=hash_l1,
                image_mse=image_mse,
            )
            scores.append(prompt_score)
            if best_score is None or prompt_score.score < best_score.score:
                best_score = prompt_score
                best_image = images.detach().cpu()

        print(
            f"prompt_search {index + 1:03d}/{len(prompts):03d} "
            f"score={prompt_score.score:.2f} hash_l1={prompt_score.hash_l1:.2f} "
            f"prompt={prompt_score.prompt}"
        )

    if best_score is None:
        raise RuntimeError("Prompt search produced no scores.")

    if best_image is not None:
        save_tensor_image(best_image, run_dir / "prompt_search_best.png")

    payload = {
        "selected_prompt": best_score.prompt,
        "selected_score": asdict(best_score),
        "scores": [asdict(item) for item in scores],
    }
    with open(score_cache, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    with open(run_dir / "prompt_search_results.json", "w", encoding="utf-8") as handle:
        json.dump(payload | {"prompt_cache": str(prompt_cache_path), "score_cache": str(score_cache)}, handle, indent=2, sort_keys=True)

    return PromptSearchResult(
        selected_prompt=best_score.prompt,
        selected_score=best_score,
        prompt_cache=prompt_cache_path,
        score_cache=score_cache,
        scores=scores,
    )


def _image_mse(images: torch.Tensor, target_image: torch.Tensor | None) -> float | None:
    if target_image is None:
        return None
    target = target_image.to(images.device, dtype=images.dtype)
    target = resize_tensor(target, images.shape[-2:])
    return float(F.mse_loss(images, target.expand_as(images)).detach().cpu())
