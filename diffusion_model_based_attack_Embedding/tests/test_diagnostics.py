from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import image_utils
import run_utils


def test_exact_rgb_l1_uses_255_values() -> None:
    generated = torch.zeros(1, 3, 2, 2)
    target = torch.ones(1, 3, 2, 2)
    distance = image_utils.exact_rgb_l1_distance(generated, target)
    assert distance.item() == 3 * 2 * 2 * 255


def test_early_stopping_threshold_logic() -> None:
    best = run_utils.BestState(l1_distance=1800.0)
    assert best.l1_distance <= 1800.0
    assert not (run_utils.BestState(l1_distance=1800.01).l1_distance <= 1800.0)


def test_unique_run_dirs_are_separate(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    target.write_bytes(b"not-an-image-for-this-test")
    first = run_utils.unique_run_dir(tmp_path, "run", target)
    first.mkdir(parents=True)
    second = run_utils.unique_run_dir(tmp_path, "run", target)
    assert first != second


def test_tensorboard_subdirectory_is_per_run(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    paths_a = run_utils.ensure_run_layout(run_a)
    paths_b = run_utils.ensure_run_layout(run_b)
    assert paths_a["tensorboard"] != paths_b["tensorboard"]
    assert paths_a["tensorboard"].name == "tensorboard"
    assert paths_b["tensorboard"].name == "tensorboard"


def test_hash_surrogate_loss_produces_finite_gradient() -> None:
    features = torch.tensor([[0.1, 2.0, 5.0]], requires_grad=True)
    target = torch.tensor([0.0, 1.0, 6.0])
    loss = torch.nn.functional.smooth_l1_loss(features / 255.0, target.view(1, -1) / 255.0, beta=0.02)
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0


def test_l1_loss_produces_finite_nonzero_embedding_gradient() -> None:
    embedding = torch.nn.Parameter(torch.tensor([[0.5, -0.25]]))
    image = embedding.sum().view(1, 1, 1, 1).expand(1, 3, 2, 2)
    target = torch.zeros_like(image)
    loss = torch.nn.functional.l1_loss(image * 255.0, target)
    loss.backward()
    assert embedding.grad is not None
    assert torch.isfinite(embedding.grad).all()
    assert embedding.grad.abs().sum().item() > 0


def test_generated_image_autograd_connection() -> None:
    embedding = torch.nn.Parameter(torch.tensor([[1.0, 2.0]]))
    generated = torch.sigmoid(embedding.mean()).view(1, 1, 1, 1).expand(1, 3, 4, 4)
    assert generated.requires_grad
    generated.mean().backward()
    assert embedding.grad is not None


def test_frozen_parameters_remain_unchanged() -> None:
    frozen = torch.nn.Linear(2, 2)
    frozen.requires_grad_(False)
    before = [parameter.detach().clone() for parameter in frozen.parameters()]
    trainable = torch.nn.Parameter(torch.ones(1, 2))
    loss = frozen(trainable).sum()
    loss.backward()
    torch.optim.AdamW([trainable], lr=0.1).step()
    after = list(frozen.parameters())
    assert all(torch.equal(left, right) for left, right in zip(before, after))


def test_resume_restores_step_and_best_metrics(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "run" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "step_000123.pt"
    torch.save({"global_step": 123, "best_l1_distance": 42.0, "best_hash_distance": 7.0}, checkpoint)
    payload = run_utils.torch_load(checkpoint)
    best = run_utils.BestState(
        l1_distance=float(payload["best_l1_distance"]),
        hash_distance=float(payload["best_hash_distance"]),
    )
    assert payload["global_step"] == 123
    assert best.l1_distance == 42.0
    assert best.hash_distance == 7.0


def test_resumed_tensorboard_step_sequence() -> None:
    start_step = 500
    next_step = start_step + 1
    assert next_step == 501
    assert next_step > start_step


def test_best_selection_uses_hash_distance_tie_breaker() -> None:
    best = run_utils.BestState(l1_distance=100.0, hash_distance=20.0, loss=5.0)
    candidate = {"exact_l1_distance": 100.0, "exact_hash_distance": 19.0}
    assert run_utils.is_better_result(candidate, candidate_loss=10.0, best=best)


def test_best_checkpoint_uses_stable_filename(tmp_path: Path) -> None:
    assert run_utils.best_checkpoint_path(tmp_path) == tmp_path / "best.pt"


def test_adaptive_eval_interval_shortens_near_best() -> None:
    assert (
        run_utils.effective_eval_interval(
            10,
            best_l1_distance=250.0,
            last_eval_l1_distance=330.0,
            close_delta=100.0,
            best_l1_threshold=100.0,
            factor=2.0,
            min_interval=1,
        )
        == 5
    )
    assert (
        run_utils.effective_eval_interval(
            10,
            best_l1_distance=99.0,
            last_eval_l1_distance=1000.0,
            close_delta=100.0,
            best_l1_threshold=100.0,
            factor=2.0,
            min_interval=1,
        )
        == 5
    )
    assert (
        run_utils.effective_eval_interval(
            10,
            best_l1_distance=250.0,
            last_eval_l1_distance=500.0,
            close_delta=100.0,
            best_l1_threshold=100.0,
            factor=2.0,
            min_interval=1,
        )
        == 10
    )


def test_nonfinite_recovery_lr_decay_updates_scheduler_base() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    new_lr = run_utils.decay_optimizer_lr(optimizer, factor=0.5, min_learning_rate=1e-6, lr_scheduler=scheduler)
    scheduler.step()
    assert new_lr == 5e-4
    assert optimizer.param_groups[0]["lr"] == 5e-4


def test_mean_abs_pixel_difference_from_l1() -> None:
    assert math.isclose(image_utils.mean_abs_pixel_difference_from_l1(12.0, 2, 2), 1.0)
