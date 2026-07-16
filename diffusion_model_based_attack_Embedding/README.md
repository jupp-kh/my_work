# SDXL-Turbo Placeholder Embedding CertPhash Attack

This experiment freezes `stabilityai/sdxl-turbo` and optimizes only learned placeholder-token embeddings, initialized from an existing token such as `image`.

The training loss uses the repository CertPhash model's continuous 144-dimensional output before `relu(round(...))` byte packing. Exact evaluation uses the original `compute_hash_coco` implementation from:

```text
/home/user/kharitay1/CertPhash/CertPhash/attack/utils/hashing.py
```

By default, success follows the requested hash L1 definition:

```python
success = best_l1_distance <= l1_threshold
```

where `best_l1_distance` is the exact summed L1 distance between the generated-image CertPhash byte hash and the target-image CertPhash byte hash. The script also records `exact_rgb_l1_distance`, computed from rounded `[0,255]` RGB values, and `mean_absolute_pixel_difference`.

## Install

```bash
pip install -r /home/user/kharitay1/CertPhash/my_work/diffusion_model_based_attack_Embedding/requirements.txt
```

## Run

```bash
python /home/user/kharitay1/CertPhash/my_work/diffusion_model_based_attack_Embedding/optimize_hash_embedding.py \
  --target-image /path/to/target.png \
  --prompt "a photograph of <hash-concept>" \
  --placeholder-token "<hash-concept>" \
  --initializer-token "image" \
  --num-vectors 1 \
  --steps 1000 \
  --learning-rate 1e-3 \
  --l1-threshold 1800 \
  --seed 0
```

Each execution creates a unique run under:

```text
/home/user/kharitay1/CertPhash/my_work/diffusion_model_based_attack_Embedding/result
```

At startup the script prints the exact TensorBoard command, for example:

```bash
tensorboard --logdir "<RUN_DIRECTORY>/tensorboard"
```

## Resume

Resume from a specific checkpoint:

```bash
python /home/user/kharitay1/CertPhash/my_work/diffusion_model_based_attack_Embedding/optimize_hash_embedding.py \
  --target-image /path/to/target.png \
  --runs-dir "/home/user/kharitay1/CertPhash/my_work/diffusion_model_based_attack_Embedding/result" \
  --resume-from "<RUN_DIRECTORY>/checkpoints/step_000500.pt"
```

Resume the newest checkpoint for a run:

```bash
python /home/user/kharitay1/CertPhash/my_work/diffusion_model_based_attack_Embedding/optimize_hash_embedding.py \
  --target-image /path/to/target.png \
  --run-dir "<RUN_DIRECTORY>" \
  --resume-from latest
```

## Notes

- The implementation is CertPhash-specific. `--hash-algorithm` is kept as a CLI compatibility flag and accepts only `certphash` or `coco_photodna`.
- Source is split into focused modules: `certphash_wrapper.py`, `diffusion_wrapper.py`, `text_embedding.py`, `image_utils.py`, `run_utils.py`, and `tensorboard_utils.py`.
- `--text-encoder-selection first|second|both` controls which SDXL text encoder placeholder rows are optimized.
- `--num-latent-samples` uses a fixed latent pool and optimizes one shared placeholder embedding across those samples.
- VAE decoding runs in float32 by default (`--vae-float32`) because SDXL VAE fp16 decode can produce NaNs on some GPUs.
- Gradient checkpointing is enabled by default and wraps the manual UNet/VAE calls to reduce differentiable SDXL memory use.
- If a step produces NaN/inf loss or gradients, the trainer restores the last best embedding, clears AdamW momentum, multiplies the learning rate by `--nonfinite-lr-factor`, and continues.
- Periodic `step_*.pt` checkpoints are saved only at `--checkpoint-interval`; improved results update `checkpoints/best.pt`.
- Image grids default to `--image-interval 50`; exact evaluation can run more often without forcing extra image files.
- Adaptive evaluation is enabled by default. When the last exact L1 is within `--adaptive-eval-close-delta` of the best L1, or when `best_l1 <= --adaptive-eval-best-l1-threshold`, the eval interval is divided by `--adaptive-eval-factor` and clamped by `--min-eval-interval`.
- Results are never mixed across fresh runs; resume continues inside the original run directory unless `--new-run-on-resume` is set.
