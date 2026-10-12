"""
DiLoCo (Douillard et al., 2023) on a small GPT, with k workers simulated one
after another on a single GPU. Also runs the plain-AdamW baselines.

DiLoCo run (4 workers, sync every 100 steps):
    python diloco.py --mode diloco --k 4 --H 100 --outer_lr 0.7

Sync baseline (one model, 4x the batch, same total tokens as the run above):
    python diloco.py --mode baseline --grad_accum 4

Solo baseline (one worker's batch, no collaboration):
    python diloco.py --mode baseline

Every run writes results/<name>.csv (one row per evaluation) and
results/<name>.json (config + final numbers). A run whose .json already exists
is skipped, so an interrupted sweep can just be restarted.
"""

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields

import numpy as np
import torch

from model import GPT, GPTConfig


@dataclass
class TrainConfig:
    mode: str = "diloco"  # "diloco" or "baseline"
    # --- DiLoCo ---
    k: int = 4  # number of workers
    H: int = 100  # inner steps between syncs
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    nesterov: bool = True
    # --- inner optimizer (AdamW), shared by workers and baselines ---
    steps: int = 3000  # inner steps per worker (= total steps for a baseline)
    batch_size: int = 32  # sequences per micro-batch
    grad_accum: int = 1  # micro-batches per step (baseline: 4 -> the k=4 sync baseline)
    lr: float = 6e-4
    min_lr_ratio: float = 0.1  # cosine decays to lr * min_lr_ratio
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # --- model ---
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    # --- bookkeeping ---
    eval_interval: int = 100  # inner steps between evaluations
    eval_iters: int = 20  # validation batches per evaluation
    seed: int = 0
    dtype: str = "bf16"  # "bf16" (experiments) or "fp32" (tests)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: str = "data"
    out_dir: str = "results"
    name: str = ""  # auto-generated from the config if empty
    force: bool = False  # re-run even if results exist

    def run_name(self):
        if self.name:
            return self.name
        if self.mode == "diloco":
            n = (
                f"diloco_k{self.k}_H{self.H}_olr{self.outer_lr}_m{self.outer_momentum}"
                f"{'_nest' if self.nesterov else ''}"
            )
        else:
            n = f"baseline_bs{self.batch_size}x{self.grad_accum}"
        return f"{n}_steps{self.steps}_s{self.seed}"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class Shard:
    """Random (x, y) batches from data[start:end]. Each worker owns one shard."""

    def __init__(self, data, start, end, cfg, seed):
        self.data, self.start, self.end, self.cfg = data, start, end, cfg
        self.gen = torch.Generator().manual_seed(seed)

    def next(self):
        B, T = self.cfg.batch_size, self.cfg.block_size
        ix = torch.randint(self.start, self.end - T - 1, (B,), generator=self.gen)
        return to_xy(self.data, ix, T, self.cfg.device)


def to_xy(data, ix, T, device):
    x = torch.stack(
        [torch.from_numpy(data[i : i + T].astype(np.int64)) for i in ix.tolist()]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + 1 + T].astype(np.int64))
            for i in ix.tolist()
        ]
    )
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    return x.to(device), y.to(device)


def load_data(cfg):
    train = np.memmap(
        os.path.join(cfg.data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val = np.memmap(os.path.join(cfg.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train, val


# ---------------------------------------------------------------------------
# training pieces
# ---------------------------------------------------------------------------
def make_model(cfg):
    mcfg = GPTConfig(
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
    )
    return GPT(mcfg).to(cfg.device)


def make_inner_opt(model, cfg):
    # weight decay on matrices/embeddings only, not on biases and LayerNorms (nanoGPT convention)
    params = list(model.parameters())
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        fused=cfg.device.startswith("cuda"),
    )


def lr_at(step, cfg):
    """Linear warmup, then cosine decay to lr * min_lr_ratio over cfg.steps."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + 0.5 * (cfg.lr - min_lr) * (
        1 + math.cos(math.pi * min(1.0, progress))
    )


def autocast(cfg):
    if cfg.dtype == "bf16":
        return torch.autocast(
            device_type=cfg.device.split(":")[0], dtype=torch.bfloat16
        )
    return torch.autocast(device_type=cfg.device.split(":")[0], enabled=False)


def inner_step(model, opt, shard, step, cfg):
    """One AdamW step (with gradient accumulation). Returns the mean training loss."""
    for g in opt.param_groups:
        g["lr"] = lr_at(step, cfg)
    total = 0.0
    for _ in range(cfg.grad_accum):
        x, y = shard.next()
        with autocast(cfg):
            _, loss = model(x, y)
        (loss / cfg.grad_accum).backward()
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return total / cfg.grad_accum


@torch.no_grad()
def evaluate(model, val, val_ix, cfg):
    """Mean loss on a FIXED set of validation batches, so every evaluation is comparable."""
    model.eval()
    losses = []
    for ix in val_ix:
        x, y = to_xy(val, ix, cfg.block_size, cfg.device)
        with autocast(cfg):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def fixed_val_batches(val, cfg):
    g = torch.Generator().manual_seed(1234)  # same batches for every run
    return [
        torch.randint(0, len(val) - cfg.block_size - 1, (cfg.batch_size,), generator=g)
        for _ in range(cfg.eval_iters)
    ]


# ---------------------------------------------------------------------------
# DiLoCo
# ---------------------------------------------------------------------------
def run_diloco(cfg, train, val, log):
    torch.manual_seed(cfg.seed)
    global_model = make_model(cfg)

    # ONE physical worker model, reused by all k workers. This works because every worker
    # restarts from the global weights each round; the only thing a worker keeps between
    # rounds is its AdamW state, so we keep k optimizers that all point at the same tensors.
    worker = make_model(cfg)
    inner_opts = [make_inner_opt(worker, cfg) for _ in range(cfg.k)]
    outer_opt = torch.optim.SGD(
        global_model.parameters(),
        lr=cfg.outer_lr,
        momentum=cfg.outer_momentum,
        nesterov=cfg.nesterov and cfg.outer_momentum > 0,
    )

    # each worker gets its own contiguous, non-overlapping slice of the training data
    bounds = np.linspace(0, len(train), cfg.k + 1).astype(np.int64)
    shards = [
        Shard(train, bounds[i], bounds[i + 1], cfg, seed=cfg.seed * 1000 + i)
        for i in range(cfg.k)
    ]

    g_params = list(global_model.parameters())
    w_params = list(worker.parameters())
    avg = [
        torch.zeros_like(p) for p in g_params
    ]  # running mean of the workers' weights

    assert cfg.steps % cfg.H == 0, "steps must be divisible by H"
    T = cfg.steps // cfg.H
    steps_done, train_losses = 0, []
    log(global_model, 0, 0, None)

    for t in range(T):
        torch._foreach_zero_(avg)
        for i in range(cfg.k):
            with torch.no_grad():
                torch._foreach_copy_(w_params, g_params)  # worker starts from global
            for h in range(cfg.H):
                train_losses.append(
                    inner_step(worker, inner_opts[i], shards[i], steps_done + h, cfg)
                )
            with torch.no_grad():
                torch._foreach_add_(avg, w_params, alpha=1.0 / cfg.k)
        steps_done += cfg.H

        # outer step: pseudo-gradient = how far the average worker moved, written into .grad
        with torch.no_grad():
            for p, a in zip(g_params, avg):
                p.grad = p - a
        outer_opt.step()
        outer_opt.zero_grad(set_to_none=True)

        if steps_done % cfg.eval_interval < cfg.H or t == T - 1:
            if log(global_model, steps_done, t + 1, float(np.mean(train_losses))):
                return
            train_losses = []


# ---------------------------------------------------------------------------
# baseline: plain AdamW on one model
# ---------------------------------------------------------------------------
def run_baseline(cfg, train, val, log):
    torch.manual_seed(cfg.seed)
    model = make_model(cfg)
    opt = make_inner_opt(model, cfg)
    shard = Shard(
        train, 0, len(train), cfg, seed=cfg.seed * 1000
    )  # same stream as DiLoCo worker 0
    train_losses = []
    log(model, 0, 0, None)
    for step in range(cfg.steps):
        train_losses.append(inner_step(model, opt, shard, step, cfg))
        if (step + 1) % cfg.eval_interval == 0 or step == cfg.steps - 1:
            # a data-parallel version of this baseline would communicate every step
            if log(model, step + 1, step + 1, float(np.mean(train_losses))):
                return
            train_losses = []


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run(cfg, train=None, val=None, write=True, verbose=True):
    """Train one config. Returns the list of logged rows."""
    if cfg.dtype == "fp32":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    name = cfg.run_name()
    csv_path = os.path.join(cfg.out_dir, name + ".csv")
    json_path = os.path.join(cfg.out_dir, name + ".json")
    if write and os.path.exists(json_path) and not cfg.force:
        print(f"[skip] {name} already done")
        return None
    if train is None:
        train, val = load_data(cfg)

    val_ix = fixed_val_batches(val, cfg)
    workers = cfg.k if cfg.mode == "diloco" else 1
    tokens_per_step = cfg.batch_size * cfg.grad_accum * cfg.block_size * workers
    rows, t0 = [], time.time()
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def log(model, step, comm_rounds, train_loss):
        """Evaluate and record one row. Returns True if the run diverged (stop early)."""
        val_loss = evaluate(model, val, val_ix, cfg)
        elapsed = time.time() - t0
        row = dict(
            step=step,
            comm_rounds=comm_rounds,
            tokens=step * tokens_per_step,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=lr_at(max(step - 1, 0), cfg),
            elapsed_s=round(elapsed, 1),
        )
        rows.append(row)
        if verbose:
            tps = row["tokens"] / elapsed if elapsed > 0 else 0
            tl = f"{train_loss:.4f}" if train_loss is not None else "  -   "
            print(
                f"step {step:5d} | rounds {comm_rounds:5d} | train {tl} | val {val_loss:.4f} "
                f"| {tps / 1e3:6.1f}k tok/s | {elapsed / 60:5.1f} min",
                flush=True,
            )
        return not math.isfinite(val_loss)

    if verbose:
        n = make_model(cfg).num_params()
        print(
            f"run: {name}\nparams: {n / 1e6:.1f}M | device: {cfg.device} | dtype: {cfg.dtype}"
        )

    if cfg.mode == "diloco":
        run_diloco(cfg, train, val, log)
    elif cfg.mode == "baseline":
        run_baseline(cfg, train, val, log)
    else:
        raise ValueError(f"unknown mode {cfg.mode}")

    final = rows[-1]
    summary = dict(
        config=asdict(cfg),
        name=name,
        final_val_loss=final["val_loss"],
        diverged=not math.isfinite(final["val_loss"]),
        steps_done=final["step"],
        comm_rounds=final["comm_rounds"],
        tokens=final["tokens"],
        minutes=round((time.time() - t0) / 60, 2),
        peak_mem_gb=(
            round(torch.cuda.max_memory_allocated() / 1e9, 2)
            if cfg.device.startswith("cuda")
            else None
        ),
    )
    if verbose:
        print(
            f"done: final val loss {summary['final_val_loss']:.4f} | "
            f"{summary['minutes']} min | peak GPU mem {summary['peak_mem_gb']} GB"
        )
    if write:
        os.makedirs(cfg.out_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
    return rows


def parse_args():
    p = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        if f.type in (bool, "bool"):
            p.add_argument(
                f"--{f.name}",
                type=lambda s: s.lower() in ("1", "true", "yes"),
                default=f.default,
            )
        else:
            t = {"int": int, "float": float, "str": str}.get(f.type, f.type)
            p.add_argument(f"--{f.name}", type=t, default=f.default)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    run(parse_args())
