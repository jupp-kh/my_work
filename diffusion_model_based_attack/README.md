# Diffusion Model Based CertPhash Attack

This directory contains a modular latent/noise optimization pipeline. It freezes the pretrained diffusion model and the CertPhash model, then optimizes only the initial diffusion latent/noise tensor.

## What It Optimizes

The true CertPhash byte hash uses `round`, `relu`, integer conversion, byte packing, and base64 encoding, so gradients do not propagate through the final hash. The optimizer therefore uses a differentiable surrogate loss between continuous CertPhash logits and the target quantized hash bytes. It still measures success with the true rounded L1 hash distance.

Default success threshold:

```bash
--threshold 1800
```

## Install Extra Dependencies

From the repository root:

```bash
conda activate certphash
pip install -r my_work/diffusion_model_based_attack/requirements.txt
```

The current environment inspected during setup did not have CUDA available to PyTorch. CPU execution is allowed by the script but is expected to be very slow.

## Run

The default run now favors structure over raw resolution: `256x256`, `20` diffusion steps, guidance scale `5.0`, smaller latent updates, UNet/VAE checkpointing, and CUDA cache cleanup after each optimization step.

Target an image hash:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python my_work/diffusion_model_based_attack/optimize_latent.py \
  --target-image CertPhash/train_verify/data/coco100x100_val/000000000009.jpg \
  --prompt "a realistic overhead food photograph of a lunch box filled with colorful vegetables" \
  --optimization-steps 200 \
  --threshold 1800 \
  --batch-size 1
```

Target an explicit hash:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python my_work/diffusion_model_based_attack/optimize_latent.py \
  --target-hash "BASE64_OR_144_BYTE_VALUES" \
  --prompt "a realistic overhead food photograph of a lunch box filled with colorful vegetables" \
  --optimization-steps 200 \
  --threshold 1800
```

Preview the prompt before optimization. Preview mode only saves `initial_baseline.png`, `config.json`, and `summary.json`; it does not write TensorBoard logs.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python my_work/diffusion_model_based_attack/optimize_latent.py \
  --target-image CertPhash/train_verify/data/coco100x100_val/000000000009.jpg \
  --prompt "a realistic overhead food photograph of a lunch box filled with colorful vegetables" \
  --height 384 \
  --width 384 \
  --diffusion-steps 30 \
  --guidance-scale 7.5 \
  --preview-only
```

For more structure during optimization, keep a moderate resolution and strengthen the image prior:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python my_work/diffusion_model_based_attack/optimize_latent.py \
  --target-image CertPhash/train_verify/data/coco100x100_val/000000000009.jpg \
  --prompt "a realistic overhead food photograph of a lunch box filled with colorful vegetables" \
  --height 384 \
  --width 384 \
  --diffusion-steps 20 \
  --guidance-scale 7.5 \
  --learning-rate 2e-3 \
  --grad-clip 1.0 \
  --hash-scale 64 \
  --hash-loss mse \
  --latent-l2-weight 1e-4 \
  --image-anchor-weight 0.01 \
  --total-variation-weight 1e-4 \
  --optimization-steps 400 \
  --threshold 1800 \
  --batch-size 1
```

If CUDA still runs out of memory, use the smaller emergency profile. It may be less detailed, but should still produce more structure than the earlier noisy run:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python my_work/diffusion_model_based_attack/optimize_latent.py \
  --target-image CertPhash/train_verify/data/coco100x100_val/000000000009.jpg \
  --prompt "a realistic overhead food photograph of a lunch box filled with colorful vegetables" \
  --height 256 \
  --width 256 \
  --diffusion-steps 12 \
  --guidance-scale 3.0 \
  --learning-rate 5e-4 \
  --latent-l2-weight 1e-3 \
  --image-anchor-weight 0.08 \
  --optimization-steps 200 \
  --threshold 1800 \
  --batch-size 1 \
  --disable-tensorboard \
  --image-save-interval 50
```

For higher quality after the structured low-resolution run works, increase one knob at a time: first `--diffusion-steps`, then `--guidance-scale`, and only then `--height/--width`. Increasing model or image size is usually less helpful than using a better prompt, more denoising steps, lower learning rate, and stronger anchoring.

If the current L1 jumps far above the best value, enable rollback. This restores the best latent, reduces the learning rate, and clears Adam momentum:

Avoid `--rollback-patience 1 --rollback-min-regression 0` for normal runs. It can repeatedly restore the same best latent before the optimizer has enough room to explore.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python my_work/diffusion_model_based_attack/optimize_latent.py \
  --target-image CertPhash/train_verify/data/coco100x100_val/000000000009.jpg \
  --prompt "a realistic overhead food photograph of a lunch box filled with colorful vegetables" \
  --height 384 \
  --width 384 \
  --diffusion-steps 20 \
  --guidance-scale 7.5 \
  --learning-rate 1e-3 \
  --grad-clip 0.25 \
  --hash-scale 144 \
  --hash-loss mse \
  --latent-l2-weight 1e-4 \
  --image-anchor-weight 0.005 \
  --total-variation-weight 5e-5 \
  --rollback-patience 5 \
  --rollback-min-regression 400 \
  --rollback-lr-factor 0.5 \
  --optimization-steps 600 \
  --threshold 1800 \
  --batch-size 1
```

## Optional: Optimize Prompt Embeddings

The literal prompt text is discrete, so it cannot be optimized directly with gradients. The script can optionally optimize the continuous prompt embeddings initialized from your text prompt while still keeping the diffusion model frozen.

Use this as a second-stage option after a latent-only run. Prompt embeddings are optimized as float32 tensors for stability, then cast to the diffusion model dtype during generation. If a step produces NaN or inf, the run restores the best state, decays the learning rate, clears optimizer state, and continues.

Prompt-embedding parameters:

- `--optimize-prompt-embeds`: enables soft prompt optimization in addition to latent optimization.
- `--prompt-learning-rate`: learning rate for prompt embeddings only. Keep this smaller than latent LR.
- `--prompt-l2-weight`: keeps optimized prompt embeddings close to the original text prompt embedding.
- `--prompt-grad-clip`: caps prompt-embedding gradient norm to avoid prompt drift.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python my_work/diffusion_model_based_attack/optimize_latent.py \
  --target-image CertPhash/train_verify/data/coco100x100_val/000000000009.jpg \
  --prompt "a realistic overhead food photograph of a lunch box filled with colorful vegetables" \
  --height 384 \
  --width 384 \
  --diffusion-steps 20 \
  --guidance-scale 7.5 \
  --learning-rate 1e-3 \
  --optimize-prompt-embeds \
  --prompt-learning-rate 1e-5 \
  --prompt-l2-weight 1e-3 \
  --prompt-grad-clip 0.03 \
  --hash-scale 144 \
  --hash-loss mse \
  --latent-l2-weight 1e-4 \
  --image-anchor-weight 0.005 \
  --total-variation-weight 5e-5 \
  --optimization-steps 600 \
  --threshold 1800 \
  --batch-size 1
```

The prompt embedding update is intentionally gentle. Increase `--prompt-learning-rate` only if hash progress is flat, no NaN recovery messages appear, and image structure remains good. Avoid combining prompt-embedding optimization with a large cycling prompt pool at first; use `--prompt-search-mode best` or `--prompt-search-top-k 2` for a more stable test.

## Reading Flat Runs

The printed `l1` is the true rounded hash distance. `hash_loss` is the differentiable surrogate. `logit_l1` is the continuous, unrounded distance between CertPhash logits and the target hash bytes. If all three are flat, check `update_norm`, `image_delta`, and `logit_delta`. Near-zero `update_norm` means the optimizer is barely moving the latent. Near-zero `image_delta` means the diffusion output is barely changing. Near-zero `logit_delta` means the generated image may change, but CertPhash does not see a meaningful hash change.

## Resume Training

To continue from the best latent of a previous run, pass `--init-latents`. This starts a fresh optimizer from that latent, which is usually better than carrying old Adam momentum:

```bash
--init-latents my_work/diffusion_model_based_attack/results/RUN_NAME/best_latents.pt
```

If prompt embeddings were optimized in the previous run, also pass:

```bash
--init-prompt-embeds my_work/diffusion_model_based_attack/results/RUN_NAME/best_prompt_embeds.pt
--optimize-prompt-embeds
```

To initialize from a full checkpoint:

```bash
--resume-checkpoint my_work/diffusion_model_based_attack/results/RUN_NAME/checkpoints/checkpoint_step_000200.pt
```

Add `--resume-optimizer` only when you want to restore the saved optimizer state too. Resume runs must use the same height, width, batch size, prompt pool shape, and prompt-embedding setting as the source run.

If rollback keeps returning to the same early best value, resume near the best latent instead of exactly on it:

```bash
--init-latents my_work/diffusion_model_based_attack/results/RUN_NAME/best_latents.pt
--init-latents-noise-std 0.05
--rollback-noise-std 0.03
```

Increase these values for more exploration, or decrease them if the first few steps immediately become much worse.

## Optional: Cached Prompt Search

Prompt search generates a small candidate set, previews each candidate at low cost, scores candidates by true CertPhash L1 and optional target-image MSE, saves the candidate prompts and scores, then initializes training with the selected prompt. It reuses cached prompts/scores unless the search settings change or you pass `--prompt-search-force` / `--prompt-search-force-rescore`.

Prompt search reuses the training `--height`, `--width`, `--diffusion-steps`, and `--guidance-scale` so you do not need a separate set of generation parameters.

For attack-only runs, leave `--prompt-search-image-weight` at its default `0.0`. This ranks prompts only by hash distance, regardless of whether the generated image resembles the target.

Template prompt generation is the default and does not require a text model. By default, `--prompt-search-space broad` explores unrelated scene categories instead of staying near the initial prompt, for example food, airplanes, dogs, streets, interiors, vehicles, landscapes, and object scenes. The built-in templates use short plain prompts without a repeated prefix. Use `--prompt-search-space prompt` only when you want variants of the initial prompt.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python my_work/diffusion_model_based_attack/optimize_latent.py \
  --target-image CertPhash/train_verify/data/coco100x100_val/000000000009.jpg \
  --prompt "a realistic overhead food photograph of a lunch box filled with colorful vegetables" \
  --prompt-search \
  --prompt-search-num 12 \
  --prompt-search-space broad \
  --prompt-search-mode best \
  --prompt-search-hash-weight 1.0 \
  --height 384 \
  --width 384 \
  --diffusion-steps 20 \
  --guidance-scale 7.5 \
  --learning-rate 1e-3 \
  --hash-scale 144 \
  --hash-loss mse \
  --latent-l2-weight 1e-4 \
  --image-anchor-weight 0.005 \
  --total-variation-weight 5e-5 \
  --optimization-steps 600 \
  --threshold 1800 \
  --batch-size 1
```

To use a small local prompt-generation model, pass its Hugging Face model id. GPT-style models use text generation; T5/FLAN/BART-style models are automatically used as instruction/text-to-text generators:

```bash
--prompt-search-generator-model distilgpt2
```

For cleaner prompt generation, try an instruction model if it is already installed or cached in your environment:

```bash
--prompt-search-generator-model google/flan-t5-base
```

Prompt search writes reusable data under `results/prompt_cache/` and writes the chosen prompt to `selected_prompt.json` in the run directory.

To avoid sticking to one prompt, keep the top-k scored prompts and rotate or sample them during optimization. This is useful when the goal is only to get a hash attack:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python my_work/diffusion_model_based_attack/optimize_latent.py \
  --target-image CertPhash/train_verify/data/coco100x100_val/000000000009.jpg \
  --prompt "a realistic overhead food photograph of a lunch box filled with colorful vegetables" \
  --prompt-search \
  --prompt-search-num 16 \
  --prompt-search-space broad \
  --prompt-search-mode cycle \
  --prompt-search-top-k 4 \
  --prompt-search-hash-weight 1.0 \
  --prompt-search-image-weight 0.0 \
  --height 384 \
  --width 384 \
  --diffusion-steps 20 \
  --guidance-scale 7.5 \
  --learning-rate 1e-3 \
  --hash-scale 144 \
  --hash-loss mse \
  --image-anchor-weight 0 \
  --latent-l2-weight 0 \
  --total-variation-weight 0 \
  --optimization-steps 800 \
  --threshold 1800 \
  --batch-size 1
```

Use `--prompt-search-mode random` instead of `cycle` if you want more exploration. Use `cycle` first because it is deterministic and easier to compare across runs.

## TensorBoard

```bash
tensorboard --logdir my_work/results/tensorboard/diffusion_model_based_attack
```

## Outputs

Each run writes:

- `config.json`
- `metrics.jsonl`
- `summary.json`
- `best.png`
- `best_latents.pt`
- periodic sample grids in `samples/`
- periodic checkpoints in `checkpoints/`
- TensorBoard logs with losses, true L1 distances, learning rate, gradient norm, generated samples, and best image
