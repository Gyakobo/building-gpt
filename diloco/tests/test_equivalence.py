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
    base = dict(n_layer=2, n_head=2, n_embd=64, block_size=32, batch_size=4,
                steps=20, warmup=5, eval_interval=1, eval_iters=2,
                dtype="fp32", device=DEVICE, seed=0)
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    # 1000-token vocabulary is enough; ids must stay < vocab_size (50304)
    return (rng.integers(0, 1000, 50_000).astype(np.uint16),
            rng.integers(0, 1000, 5_000).astype(np.uint16))


def val_losses(rows):
    return np.array([r["val_loss"] for r in rows])


def train_once(cfg, data):
    return run(cfg, *data, write=False, verbose=False)


@pytest.mark.parametrize("H", [1, 5])
def test_diloco_k1_equals_adamw(data, H):
    base = train_once(tiny(mode="baseline"), data)
    dil = train_once(tiny(mode="diloco", k=1, H=H, outer_lr=1.0,
                          outer_momentum=0.0, nesterov=False, eval_interval=H), data)
    b = val_losses(base)[:: H]           # baseline logs every step; DiLoCo every H steps
    d = val_losses(dil)
    assert len(b) == len(d)
    np.testing.assert_allclose(d, b, rtol=0, atol=1e-4)


def test_training_reduces_loss(data):
    rows = train_once(tiny(mode="diloco", k=2, H=5, steps=40, lr=3e-3), data)
    v = val_losses(rows)
    assert v[-1] < v[0] - 0.5, f"loss did not drop: {v[0]:.3f} -> {v[-1]:.3f}"


def test_outer_lr_zero_freezes_model(data):
    # outer_lr = 0 means the global model never moves, whatever the workers do
    rows = train_once(tiny(mode="diloco", k=2, H=5, outer_lr=0.0, outer_momentum=0.0,
                           nesterov=False, eval_interval=5), data)
    v = val_losses(rows)
    np.testing.assert_allclose(v, v[0], atol=1e-6)


def test_workers_see_different_data(data):
    # with k=2 the run must differ from k=1 (otherwise sharding is broken)
    a = train_once(tiny(mode="diloco", k=1, H=5, eval_interval=5), data)
    b = train_once(tiny(mode="diloco", k=2, H=5, eval_interval=5), data)
    assert not np.allclose(val_losses(a)[1:], val_losses(b)[1:])


def test_length_shards_partition_by_story_length():
    from diloco import EOT, length_shards
    rng = np.random.default_rng(1)
    stories = [np.append(rng.integers(0, 1000, n), EOT) for n in rng.integers(5, 200, 300)]
    data = np.concatenate(stories).astype(np.uint16)
    parts = length_shards(data, 3)
    assert sum(len(p) for p in parts) == len(data)                      # every token used once
    assert np.array_equal(np.sort(np.concatenate(parts)), np.sort(data))
    mean_len = [len(p) / max(1, int((p == EOT).sum())) for p in parts]
    assert mean_len[0] < mean_len[1] < mean_len[2]                      # short -> long stories
    assert max(len(p) for p in parts) < 1.5 * min(len(p) for p in parts)  # balanced sizes


def test_length_mode_trains(data):
    from diloco import EOT
    train, val = data
    train = train.copy(); train[::50] = EOT                              # fake story boundaries
    rows = train_once(tiny(mode="diloco", k=2, H=5, shard_mode="length", eval_interval=5,
                           data_dir="/nonexistent"), (train, val))
    assert np.isfinite(val_losses(rows)).all()


def test_all_warm_start_equals_adamw(data):
    # k=1 and warm_start_steps == steps: the whole run is ordinary AdamW, so it must match the baseline
    base = train_once(tiny(mode="baseline"), data)
    warm = train_once(tiny(mode="diloco", k=1, H=5, warm_start_steps=20), data)
    np.testing.assert_allclose(val_losses(warm), val_losses(base), rtol=0, atol=1e-6)


def test_warm_start_then_diloco(data):
    # warm start for 10 steps, then 2 DiLoCo rounds of H=5: rounds are counted on top of the warm steps
    rows = train_once(tiny(mode="diloco", k=2, H=5, warm_start_steps=10, eval_interval=5), data)
    assert [r["step"] for r in rows] == [0, 5, 10, 15, 20]
    assert [r["comm_rounds"] for r in rows] == [0, 5, 10, 11, 12]
    assert np.isfinite(val_losses(rows)).all()


def test_validation_set_independent_of_batch_size(data):
    # every run must be scored on the same validation sequences, whatever its training batch size:
    # with the same initialization, the step-0 validation loss must match exactly
    a = train_once(tiny(mode="baseline", batch_size=4, steps=2, eval_iters=3), data)
    b = train_once(tiny(mode="baseline", batch_size=16, steps=2, eval_iters=3), data)
    assert a[0]["val_loss"] == b[0]["val_loss"]
