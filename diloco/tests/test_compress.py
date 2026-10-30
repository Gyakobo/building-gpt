"""
Tests for compress.py and the compressed DiLoCo path. CPU-only, no dataset needed.

    python -m pytest tests/test_compress.py -v
"""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compress import (BLOCK, CHUNK, ErrorFeedback, compress_all, make_compressor,  # noqa: E402
                      pack_bits, unpack_bits)

ALL = ["fp32", "fp16", "int8", "int4", "int2", "sign",
       "topk10", "topk5", "topk1", "topk0.1", "randk1", "topk1_int4"]


def rand(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g)


# ---------------------------------------------------------------------------
# bit packing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bits", [1, 2, 4])
@pytest.mark.parametrize("n", [1, 7, 128, 1001])
def test_pack_unpack_roundtrip(bits, n):
    q = torch.randint(0, 2 ** bits, (n,))
    packed = pack_bits(q, bits)
    assert packed.dtype == torch.uint8
    assert packed.numel() == math.ceil(n * bits / 8)
    assert torch.equal(unpack_bits(packed, bits, n), q)


# ---------------------------------------------------------------------------
# every compressor: shape, dtype, finiteness, determinism
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("shape", [(1,), (383,), (384,), (384, 1536), (50304, 8), (3, 5, 7)])
def test_roundtrip_shape_and_finite(name, shape):
    c = make_compressor(name)
    x = rand(*shape)
    x_hat, nbytes = c.roundtrip(x, gen_seed=42)
    assert x_hat.shape == x.shape and x_hat.dtype == torch.float32
    assert torch.isfinite(x_hat).all()
    assert nbytes > 0


@pytest.mark.parametrize("name", ALL)
def test_zero_tensor_stays_zero(name):
    x = torch.zeros(4096)
    x_hat, _ = make_compressor(name).roundtrip(x, gen_seed=1)
    assert torch.equal(x_hat, x)


@pytest.mark.parametrize("name", ALL)
def test_deterministic(name):
    x = rand(8192)
    a, _ = make_compressor(name).roundtrip(x, gen_seed=5)
    b, _ = make_compressor(name).roundtrip(x, gen_seed=5)
    assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# accuracy of each quantizer
# ---------------------------------------------------------------------------
def test_fp32_is_exact():
    x = rand(1000)
    assert torch.equal(make_compressor("fp32").roundtrip(x)[0], x)


def test_fp16_error():
    x = rand(10_000)
    x_hat = make_compressor("fp16").roundtrip(x)[0]
    assert ((x_hat - x).abs() <= x.abs() * 2 ** -10 + 1e-7).all()


def test_int8_error_within_half_step():
    x = rand(10_000)
    x_hat = make_compressor("int8").roundtrip(x)[0]
    step = x.abs().max() / 127
    assert (x_hat - x).abs().max() <= step / 2 + 1e-6


def test_int4_error_within_half_step_per_block():
    x = rand(BLOCK * 20)
    x_hat = make_compressor("int4").roundtrip(x)[0]
    step = x.view(-1, BLOCK).abs().amax(1, keepdim=True) / 7
    # fp16 scale storage adds a tiny relative error on top of rounding
    assert ((x_hat - x).view(-1, BLOCK).abs() <= step / 2 * 1.01 + 1e-6).all()


def test_int4_uses_15_levels_per_block():
    x = rand(BLOCK)
    x_hat = make_compressor("int4").roundtrip(x)[0]
    assert len(torch.unique(x_hat)) <= 15


@pytest.mark.parametrize("name,levels", [("int2", 4), ("sign", 2)])
def test_low_bit_level_count(name, levels):
    x = rand(BLOCK)                        # one block -> one scale
    x_hat = make_compressor(name).roundtrip(x)[0]
    assert len(torch.unique(x_hat)) <= levels


def test_sign_keeps_sign_and_block_mean_abs():
    x = rand(BLOCK * 4)
    x_hat = make_compressor("sign").roundtrip(x)[0]
    assert torch.equal(torch.sign(x_hat), torch.where(x >= 0, 1.0, -1.0))
    mean_abs = x.view(-1, BLOCK).abs().mean(1)
    np.testing.assert_allclose(x_hat.view(-1, BLOCK).abs().mean(1).numpy(),
                               mean_abs.half().float().numpy(), rtol=1e-6)


def test_quantization_error_ordering():
    # more bits -> smaller error, on the same tensor
    x = rand(BLOCK * 64)
    err = {n: float((make_compressor(n).roundtrip(x)[0] - x).norm() / x.norm())
           for n in ["fp16", "int8", "int4", "int2", "sign"]}
    assert err["fp16"] < err["int8"] < err["int4"] < err["int2"] < err["sign"] < 1.0


# ---------------------------------------------------------------------------
# sparsifiers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("density", [0.1, 0.05, 0.01, 0.001])
def test_topk_keeps_largest_per_chunk(density):
    x = rand(CHUNK * 3)
    x_hat = make_compressor(f"topk{density * 100:g}").roundtrip(x)[0]
    k = max(1, round(density * CHUNK))
    for xc, hc in zip(x.view(-1, CHUNK), x_hat.view(-1, CHUNK)):
        kept = hc != 0
        assert int(kept.sum()) == k
        assert torch.equal(hc[kept], xc[kept])                              # values sent exactly
        assert xc[kept].abs().min() >= xc[~kept].abs().max()                # and they are the largest


def test_topk_small_tensor_is_its_own_chunk():
    x = rand(384)                          # a bias / LayerNorm vector
    x_hat = make_compressor("topk1").roundtrip(x)[0]
    assert int((x_hat != 0).sum()) == round(0.01 * 384)


def test_topk_int4_matches_support_of_topk():
    x = rand(CHUNK * 2)
    a = make_compressor("topk1").roundtrip(x)[0]
    b = make_compressor("topk1_int4").roundtrip(x)[0]
    assert torch.equal(a != 0, b != 0) or (b != 0).sum() <= (a != 0).sum()  # int4 may round tiny values to 0
    rel = float((b - a).norm() / a.norm())
    assert rel < 0.15


def test_randk_mask_depends_only_on_seed():
    x, y = rand(CHUNK * 2, seed=1), rand(CHUNK * 2, seed=2)
    c = make_compressor("randk1")
    a, _ = c.roundtrip(x, gen_seed=7)
    b, _ = c.roundtrip(y, gen_seed=7)
    d, _ = c.roundtrip(x, gen_seed=8)
    assert torch.equal(a != 0, b != 0)          # same seed -> same positions, whatever the values
    assert not torch.equal(a != 0, d != 0)      # different seed -> different positions
    assert torch.equal(a[a != 0], x[a != 0])    # kept values are exact (no rescaling)


def test_randk_needs_seed():
    with pytest.raises(AssertionError):
        make_compressor("randk1").roundtrip(rand(CHUNK))


# ---------------------------------------------------------------------------
# byte accounting: must match the table in the spec
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,bytes_per_param", [
    ("fp32", 4.0), ("fp16", 2.0),
    ("int4", 0.5 + 2 / BLOCK), ("int2", 0.25 + 2 / BLOCK), ("sign", 0.125 + 2 / BLOCK),
    ("topk10", 0.1 * 6), ("topk5", 0.05 * 6), ("topk1", 0.01 * 6), ("topk0.1", 0.001 * 6),
    ("randk1", 0.01 * 4), ("topk1_int4", 0.01 * 2.5 + 2 / CHUNK),
])
def test_bytes_per_param(name, bytes_per_param):
    n = CHUNK * 64                          # divisible by BLOCK and CHUNK, so no padding
    _, nbytes = make_compressor(name).roundtrip(rand(n), gen_seed=0)
    k_round = {"topk10": 410 / 409.6, "topk5": 205 / 204.8, "topk1": 41 / 40.96,
               "topk0.1": 4 / 4.096, "randk1": 41 / 40.96, "topk1_int4": 41 / 40.96}
    expected = bytes_per_param * n * k_round.get(name, 1.0)    # k is rounded to an integer per chunk
    if name == "topk1_int4":   # per chunk: 41 int16 indices + one fp16 scale; all values packed 2 per byte
        expected = 64 * (41 * 2 + 2) + math.ceil(64 * 41 / 2)
    assert nbytes == pytest.approx(expected, rel=1e-9)


def test_int8_bytes():
    n = 10_000
    assert make_compressor("int8").roundtrip(rand(n))[1] == n + 4


# ---------------------------------------------------------------------------
# error feedback
# ---------------------------------------------------------------------------
def test_ef_lossless_keeps_buffer_zero():
    params = [rand(256), rand(16, 16, seed=1)]
    ef = ErrorFeedback(params)
    for r in range(3):
        deltas = [rand(256, seed=10 + r), rand(16, 16, seed=20 + r)]
        sent, _ = ef.compress(make_compressor("fp32"), deltas)
        for s, d in zip(sent, deltas):
            assert torch.equal(s, d)
    assert ef.norm() == 0.0


@pytest.mark.parametrize("name", ["topk1", "int2", "sign", "randk1"])
def test_ef_telescopes(name):
    # classic EF (beta = 1): everything ever sent + what is still in the buffer = everything produced
    c = make_compressor(name)
    params = [torch.zeros(CHUNK * 2)]
    ef = ErrorFeedback(params)
    total_delta, total_sent = torch.zeros(CHUNK * 2), torch.zeros(CHUNK * 2)
    for r in range(20):
        d = rand(CHUNK * 2, seed=100 + r)
        sent, _ = ef.compress(c, [d], seeds=[r])
        total_delta += d
        total_sent += sent[0]
    torch.testing.assert_close(total_sent + ef.buf[0], total_delta, rtol=0, atol=1e-4)


def test_ef_recovers_small_coordinates_over_time():
    # a coordinate too small for top-k in one round gets sent once its residual has built up
    x = torch.full((CHUNK,), 1.0)
    x[0] = 0.05
    ef = ErrorFeedback([x])
    c = make_compressor("topk1")
    ever_sent = False
    # 41 of 4096 slots per round: each 1.0 coordinate's residual reaches ~100 before it is sent,
    # so the 0.05 coordinate needs ~2,000 rounds to catch up; without EF it would never be sent
    for r in range(4000):
        sent, _ = ef.compress(c, [x.clone()])
        ever_sent |= bool(sent[0][0] != 0)
    assert ever_sent


def test_ef_beta_decays_buffer():
    x = rand(CHUNK)
    ef = ErrorFeedback([x], beta=0.5)
    ef.compress(make_compressor("topk1"), [x])
    e1 = ef.buf[0].clone()
    sent, _ = ef.compress(make_compressor("topk1"), [torch.zeros_like(x)])
    torch.testing.assert_close(sent[0] + ef.buf[0], 0.5 * e1)


def test_compress_all_matches_per_tensor_roundtrip():
    c = make_compressor("int4")
    ds = [rand(500), rand(32, 32, seed=1)]
    out, nbytes = compress_all(c, ds)
    for o, d in zip(out, ds):
        assert torch.equal(o, c.roundtrip(d)[0])
    assert nbytes == sum(c.roundtrip(d)[1] for d in ds)


def test_unknown_compressor():
    with pytest.raises(ValueError):
        make_compressor("int3")


# ---------------------------------------------------------------------------
# integration with diloco.py
# ---------------------------------------------------------------------------
from diloco import TrainConfig, run  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tiny(**kw):
    base = dict(mode="diloco", k=2, H=5, n_layer=2, n_head=2, n_embd=64, block_size=32,
                batch_size=4, steps=20, warmup=5, eval_interval=5, eval_iters=2,
                dtype="fp32", device=DEVICE, seed=0)
    base.update(kw)
    return TrainConfig(**base)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    return (rng.integers(0, 1000, 50_000).astype(np.uint16),
            rng.integers(0, 1000, 5_000).astype(np.uint16))


def train(cfg, data):
    return run(cfg, *data, write=False, verbose=False)


def vals(rows):
    return np.array([r["val_loss"] for r in rows])


def test_fp32_compressor_matches_original_path(data):
    # averaging decoded fp32 deltas must reproduce the original weight-averaging code
    a = train(tiny(), data)
    b = train(tiny(compressor="fp32"), data)
    np.testing.assert_allclose(vals(b), vals(a), rtol=0, atol=1e-5)
    c = train(tiny(compressor="fp32", error_feedback=True), data)
    np.testing.assert_allclose(vals(c), vals(a), rtol=0, atol=1e-5)


def test_uncompressed_run_names_unchanged():
    assert TrainConfig(k=4, H=100, steps=4000).run_name() == "diloco_k4_H100_olr0.7_m0.9_nest_steps4000_s0"
    assert (TrainConfig(k=4, H=100, steps=14000, compressor="topk1", error_feedback=True).run_name()
            == "diloco_k4_H100_olr0.7_m0.9_nest_ctopk1_ef_steps14000_s0")
    assert "_ef0.95" in TrainConfig(compressor="topk1", error_feedback=True, ef_beta=0.95).run_name()


def test_bytes_logged_per_round(data):
    cfg = tiny(compressor="fp32")
    rows = train(cfg, data)
    from diloco import make_model
    n_params = sum(p.numel() for p in make_model(cfg).parameters())   # tied weights counted once
    rounds = cfg.steps // cfg.H
    assert rows[-1]["uplink_bytes"] == rounds * cfg.k * 4 * n_params
    assert rows[-1]["cos_sim"] == pytest.approx(1.0, abs=1e-6)
    assert rows[-1]["rel_err"] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("name,ef", [("int4", False), ("sign", True), ("topk1", True),
                                     ("randk1", True), ("topk1_int4", True)])
def test_compressed_runs_train(data, name, ef):
    rows = train(tiny(compressor=name, error_feedback=ef, steps=40, lr=3e-3), data)
    v = vals(rows)
    assert np.isfinite(v).all()
    assert v[-1] < v[0] - 0.3
    assert 0 < rows[-1]["cos_sim"] <= 1.0 + 1e-6
    if ef:
        assert rows[-1]["ef_norm"] > 0


def test_lossy_compression_changes_result(data):
    a = train(tiny(compressor="fp32"), data)
    b = train(tiny(compressor="topk1"), data)
    assert not np.allclose(vals(a)[1:], vals(b)[1:])


def test_final_large_eval(data, tmp_path):
    import json
    cfg = tiny(compressor="int8", final_eval_iters=6, out_dir=str(tmp_path))
    run(cfg, *data, verbose=False)
    s = json.load(open(tmp_path / (cfg.run_name() + ".json")))
    assert s["final_val_loss_large"] is not None and math.isfinite(s["final_val_loss_large"])
    assert s["total_uplink_bytes"] > 0
    # the CSV has the comm columns even though the first row (step 0) has none
    header = open(tmp_path / (cfg.run_name() + ".csv")).readline()
    assert "uplink_bytes" in header and "cos_sim" in header
