"""
Correctness checks for diloco.py. Runs on random tokens, so no dataset is needed.

    python -m pytest tests/ -v

The key fact: with ONE worker, outer_lr = 1 and no momentum, the outer step is
    theta <- theta - 1 * (theta - theta_worker) = theta_worker
so DiLoCo must reproduce plain AdamW exactly, for any H.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diloco import TrainConfig, run  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tiny(**kw):
    base = dict(
        n_layer=2,
        n_head=2,
        n_embd=64,
        block_size=32,
        batch_size=4,
        steps=20,
        warmup=5,
        eval_interval=1,
        eval_iters=2,
        dtype="fp32",
        device=DEVICE,
        seed=0,
    )
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    # 1000-token vocabulary is enough; ids must stay < vocab_size (50304)
    return (
        rng.integers(0, 1000, 50_000).astype(np.uint16),
        rng.integers(0, 1000, 5_000).astype(np.uint16),
    )


def val_losses(rows):
    return np.array([r["val_loss"] for r in rows])


def train_once(cfg, data):
    return run(cfg, *data, write=False, verbose=False)


@pytest.mark.parametrize("H", [1, 5])
def test_diloco_k1_equals_adamw(data, H):
    base = train_once(tiny(mode="baseline"), data)
    dil = train_once(
        tiny(
            mode="diloco",
            k=1,
            H=H,
            outer_lr=1.0,
            outer_momentum=0.0,
            nesterov=False,
            eval_interval=H,
        ),
        data,
    )
    b = val_losses(base)[::H]  # baseline logs every step; DiLoCo every H steps
    d = val_losses(dil)
    assert len(b) == len(d)
    np.testing.assert_allclose(d, b, rtol=0, atol=1e-4)


def test_training_reduces_loss(data):
    rows = train_once(tiny(mode="diloco", k=2, H=5, steps=40, lr=3e-3), data)
    v = val_losses(rows)
    assert v[-1] < v[0] - 0.5, f"loss did not drop: {v[0]:.3f} -> {v[-1]:.3f}"


def test_outer_lr_zero_freezes_model(data):
    # outer_lr = 0 means the global model never moves, whatever the workers do
    rows = train_once(
        tiny(
            mode="diloco",
            k=2,
            H=5,
            outer_lr=0.0,
            outer_momentum=0.0,
            nesterov=False,
            eval_interval=5,
        ),
        data,
    )
    v = val_losses(rows)
    np.testing.assert_allclose(v, v[0], atol=1e-6)


def test_workers_see_different_data(data):
    # with k=2 the run must differ from k=1 (otherwise sharding is broken)
    a = train_once(tiny(mode="diloco", k=1, H=5, eval_interval=5), data)
    b = train_once(tiny(mode="diloco", k=2, H=5, eval_interval=5), data)
    assert not np.allclose(val_losses(a)[1:], val_losses(b)[1:])
