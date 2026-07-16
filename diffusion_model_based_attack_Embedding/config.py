from __future__ import annotations

from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path("/home/user/kharitay1/CertPhash")
CERTPHASH_ROOT = REPO_ROOT / "CertPhash"
CERTPHASH_ATTACK_ROOT = CERTPHASH_ROOT / "attack"

DEFAULT_RUNS_DIR = SCRIPT_DIR / "result"
DEFAULT_MODEL_ID = "stabilityai/sdxl-turbo"
DEFAULT_CERTPHASH_MODEL = CERTPHASH_ROOT / "train_verify" / "saved_models" / "coco_photodna_ep8" / "ckpt_best.pth"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_HASH_ALGORITHMS = ("certphash", "coco_photodna")

METRIC_FIELDS = [
    "step",
    "total_loss",
    "hash_surrogate_loss",
    "normalized_l1_loss",
    "embedding_regularization",
    "image_range_loss",
    "exact_l1_distance",
    "best_l1_distance",
    "exact_rgb_l1_distance",
    "mean_absolute_pixel_difference",
    "best_mean_absolute_pixel_difference",
    "l1_threshold",
    "l1_minus_threshold",
    "exact_hash_distance",
    "best_hash_distance",
    "bit_match_percentage",
    "learning_rate",
    "gradient_norm",
    "embedding_norm",
    "embedding_displacement",
    "runtime_seconds",
    "active_eval_interval",
    "best_step",
    "success",
    "skipped_step",
    "skip_reason",
]
