"""
DiLoCo (Douillard et al., 2023) on a small GPT, with k workers simulated one
after another on a single GPU. Also runs the plain-AdamW baselines.

DiLoCo run (4 workers, sync every 100 steps):
    python diloco.py --mode diloco --k 4 --H 100 --outer_lr 0.7

Sync baseline (one model, 4x the batch, same total tokens as the run above):
    python diloco.py --mode baseline --batch_size 128

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
from dataclasses import asdict, dataclass, fields, replace

import numpy as np
import torch

from model import GPT, GPTConfig


@dataclass
class TrainConfig:
    mode: str = "diloco"        # "diloco" or "baseline"
    # --- DiLoCo ---
    k: int = 4                  # number of workers
    shard_mode: str = "random"  # "random": contiguous slices of the data (i.i.d.-like);
                                # "length": worker i gets the i-th length quantile of stories (non-i.i.d.)
    H: int = 100                # inner steps between syncs
    warm_start_steps: int = 0   # first train this many steps synchronously (batch 32*k), then switch to DiLoCo
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    nesterov: bool = True
    # --- inner optimizer (AdamW), shared by workers and baselines ---
    steps: int = 3000           # inner steps per worker (= total steps for a baseline)
    batch_size: int = 32        # sequences per optimizer step (sync baseline: 32 * k = 128)
    micro_batch: int = 16       # sequences per forward/backward pass; only affects memory, not results
    lr: float = 6e-4
    min_lr_ratio: float = 0.1   # cosine decays to lr * min_lr_ratio
    warmup: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # --- model ---
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    # --- bookkeeping ---
    eval_interval: int = 100    # inner steps between evaluations
    eval_iters: int = 20        # validation batches per evaluation
    seed: int = 0
    dtype: str = "bf16"         # "bf16" (experiments) or "fp32" (tests)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir: str = "data"
    out_dir: str = "results"
    name: str = ""              # auto-generated from the config if empty
    force: bool = False         # re-run even if results exist

    def run_name(self):
        if self.name:
            return self.name
        if self.mode == "diloco":
            n = (f"diloco_k{self.k}_H{self.H}_olr{self.outer_lr}_m{self.outer_momentum}"
                 f"{'_nest' if self.nesterov else ''}"
                 f"{'' if self.shard_mode == 'random' else '_' + self.shard_mode}"
                 f"{f'_warm{self.warm_start_steps}' if self.warm_start_steps else ''}")
        else:
            n = f"baseline_bs{self.batch_size}"
        if (self.n_layer, self.n_head, self.n_embd) != (6, 6, 384):   # non-default model size
            n += f"_L{self.n_layer}d{self.n_embd}"
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
    x = torch.stack([torch.from_numpy(data[i: i + T].astype(np.int64)) for i in ix.tolist()])
    y = torch.stack([torch.from_numpy(data[i + 1: i + 1 + T].astype(np.int64)) for i in ix.tolist()])
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


EOT = 50256   # GPT-2 end-of-text token, written between stories by data/prepare.py


def length_shards(train, k, cache_dir=None):
    """Non-i.i.d. split: sort stories by length and give worker i the i-th quantile.

    Worker 0 gets the shortest stories, worker k-1 the longest; each worker gets about the
    same number of tokens. Returns k arrays. Cached to disk so it is built only once.
    """
    paths = [os.path.join(cache_dir, f"train_len{k}_{i}.bin") for i in range(k)] if cache_dir else None
    if paths and all(os.path.exists(p) for p in paths):
        return [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]

    ends = np.flatnonzero(np.asarray(train) == EOT) + 1          # one past each story's EOT
    if len(ends) == 0 or ends[-1] != len(train):
        ends = np.append(ends, len(train))
    starts = np.concatenate([[0], ends[:-1]])
    lengths = ends - starts
    order = np.argsort(lengths, kind="stable")                    # shortest first
    cum = np.cumsum(lengths[order])
    # cut points so every worker gets ~len(train)/k tokens
    cuts = np.searchsorted(cum, np.arange(1, k) * cum[-1] / k)
    group = np.empty(len(lengths), dtype=np.uint8)
    for i, idx in enumerate(np.split(order, cuts)):
        group[idx] = i
    token_group = np.repeat(group, lengths)
    data = np.asarray(train)
    shards = [data[token_group == i] for i in range(k)]
    if paths:
        for p, sh in zip(paths, shards):
            sh.tofile(p)
        return [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
    return shards


def load_data(cfg):
    train = np.memmap(os.path.join(cfg.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(cfg.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train, val


# ---------------------------------------------------------------------------
# training pieces
# ---------------------------------------------------------------------------
def make_model(cfg):
    mcfg = GPTConfig(block_size=cfg.block_size, n_layer=cfg.n_layer,
                     n_head=cfg.n_head, n_embd=cfg.n_embd)
    return GPT(mcfg).to(cfg.device)


def make_inner_opt(model, cfg):
    # weight decay on matrices/embeddings only, not on biases and LayerNorms (nanoGPT convention)
    params = list(model.parameters())
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), fused=cfg.device.startswith("cuda"),
    )


def lr_at(step, cfg):
    """Linear warmup, then cosine decay to lr * min_lr_ratio over cfg.steps."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    min_lr = cfg.lr * cfg.min_lr_ratio
    return min_lr + 0.5 * (cfg.lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def autocast(cfg):
    if cfg.dtype == "bf16":
        return torch.autocast(device_type=cfg.device.split(":")[0], dtype=torch.bfloat16)
    return torch.autocast(device_type=cfg.device.split(":")[0], enabled=False)


def inner_step(model, opt, shard, step, cfg):
    """One AdamW step (with gradient accumulation). Returns the mean training loss."""
    for g in opt.param_groups:
        g["lr"] = lr_at(step, cfg)
    x, y = shard.next()
    n_micro = max(1, x.size(0) // cfg.micro_batch)
    total = 0.0
    # split the batch into chunks so the (tokens x 50k vocab) loss tensor fits in 8 GB;
    # gradients add up across chunks, so the update is the same as one big batch
    for xm, ym in zip(x.chunk(n_micro), y.chunk(n_micro)):
        with autocast(cfg):
            _, loss = model(xm, ym)
        (loss / n_micro).backward()
        total += loss.item()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return total / n_micro


@torch.no_grad()
def evaluate(model, val, val_ix, cfg):
    """Mean loss on a FIXED set of validation batches, so every evaluation is comparable."""
    model.eval()
    losses = []
    for ix in val_ix:
        x, y = to_xy(val, ix, cfg.block_size, cfg.device)
        for xm, ym in zip(x.chunk(max(1, x.size(0) // cfg.micro_batch)),
                          y.chunk(max(1, y.size(0) // cfg.micro_batch))):
            with autocast(cfg):
                _, loss = model(xm, ym)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


EVAL_BATCH = 32   # sequences per validation batch -- fixed, so every run sees the same validation set


def fixed_val_batches(val, cfg):
    # Same 20 x 32 validation sequences for EVERY run, whatever its training batch size.
    # (An earlier version used cfg.batch_size here, which gave the large-batch sync baselines
    # a larger validation set than everything else.)
    g = torch.Generator().manual_seed(1234)
    return [torch.randint(0, len(val) - cfg.block_size - 1, (EVAL_BATCH,), generator=g)
            for _ in range(cfg.eval_iters)]


# ---------------------------------------------------------------------------
# DiLoCo
# ---------------------------------------------------------------------------
def copy_opt_state(src_opt, src_model, dst_opts, dst_model):
    """Copy AdamW state from src_opt (on src_model's params) into each optimizer in dst_opts
    (on dst_model's params). Parameters are matched by position, which is safe because both
    models are built by make_model and both optimizers by make_inner_opt."""
    pairs = list(zip(src_model.parameters(), dst_model.parameters()))
    for opt in dst_opts:
        for ps, pd in pairs:
            st = src_opt.state.get(ps)
            if st:
                opt.state[pd] = {k: v.clone() if torch.is_tensor(v) else v for k, v in st.items()}


def run_diloco(cfg, train, val, log):
    torch.manual_seed(cfg.seed)
    global_model = make_model(cfg)

    # ONE physical worker model, reused by all k workers. This works because every worker
    # restarts from the global weights each round; the only thing a worker keeps between
    # rounds is its AdamW state, so we keep k optimizers that all point at the same tensors.
    worker = make_model(cfg)
    inner_opts = [make_inner_opt(worker, cfg) for _ in range(cfg.k)]
    outer_opt = torch.optim.SGD(global_model.parameters(), lr=cfg.outer_lr,
                                momentum=cfg.outer_momentum,
                                nesterov=cfg.nesterov and cfg.outer_momentum > 0)

    if cfg.shard_mode == "length":
        # non-i.i.d.: each worker trains on a different range of story lengths
        cache = cfg.data_dir if os.path.isdir(cfg.data_dir) else None
        parts = length_shards(train, cfg.k, cache_dir=cache)
        shards = [Shard(parts[i], 0, len(parts[i]), cfg, seed=cfg.seed * 1000 + i)
                  for i in range(cfg.k)]
    else:
        # each worker gets its own contiguous, non-overlapping slice of the training data
        bounds = np.linspace(0, len(train), cfg.k + 1).astype(np.int64)
        shards = [Shard(train, bounds[i], bounds[i + 1], cfg, seed=cfg.seed * 1000 + i)
                  for i in range(cfg.k)]

    g_params = list(global_model.parameters())
    w_params = list(worker.parameters())
    avg = [torch.zeros_like(p) for p in g_params]   # running mean of the workers' weights

    W = cfg.warm_start_steps
    assert 0 <= W <= cfg.steps and (cfg.steps - W) % cfg.H == 0, "steps - warm_start_steps must be divisible by H"
    T = (cfg.steps - W) // cfg.H
    steps_done, train_losses = 0, []
    log(global_model, 0, 0, None)

    if W:
        # Warm start: ordinary synchronous training of the global model with the combined batch of
        # all k workers (same tokens per step as DiLoCo), following the same learning-rate schedule.
        sync_cfg = replace(cfg, batch_size=cfg.batch_size * cfg.k)
        sync_opt = make_inner_opt(global_model, cfg)
        sync_shard = Shard(train, 0, len(train), sync_cfg, seed=cfg.seed * 1000)
        for step in range(W):
            train_losses.append(inner_step(global_model, sync_opt, sync_shard, step, sync_cfg))
            if (step + 1) % cfg.eval_interval == 0 or step == W - 1:
                if log(global_model, step + 1, step + 1, float(np.mean(train_losses))):
                    return
                train_losses = []
        steps_done = W
        # every worker continues from the warm-start optimizer state (Adam moments and step count)
        copy_opt_state(sync_opt, global_model, inner_opts, worker)
        del sync_opt

    for t in range(T):
        torch._foreach_zero_(avg)
        for i in range(cfg.k):
            with torch.no_grad():
                torch._foreach_copy_(w_params, g_params)          # worker starts from global
            for h in range(cfg.H):
                train_losses.append(inner_step(worker, inner_opts[i], shards[i], steps_done + h, cfg))
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
            if log(global_model, steps_done, W + t + 1, float(np.mean(train_losses))):
                return
            train_losses = []


# ---------------------------------------------------------------------------
# baseline: plain AdamW on one model
# ---------------------------------------------------------------------------
def run_baseline(cfg, train, val, log):
    torch.manual_seed(cfg.seed)
    model = make_model(cfg)
    opt = make_inner_opt(model, cfg)
    shard = Shard(train, 0, len(train), cfg, seed=cfg.seed * 1000)   # samples from the whole training set
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
    tokens_per_step = cfg.batch_size * cfg.block_size * workers
    rows, t0 = [], time.time()
    if cfg.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    def log(model, step, comm_rounds, train_loss):
        """Evaluate and record one row. Returns True if the run diverged (stop early)."""
        val_loss = evaluate(model, val, val_ix, cfg)
        elapsed = time.time() - t0
        row = dict(step=step, comm_rounds=comm_rounds, tokens=step * tokens_per_step,
                   train_loss=train_loss, val_loss=val_loss, lr=lr_at(max(step - 1, 0), cfg),
                   elapsed_s=round(elapsed, 1))
        rows.append(row)
        if verbose:
            tps = row["tokens"] / elapsed if elapsed > 0 else 0
            tl = f"{train_loss:.4f}" if train_loss is not None else "  -   "
            print(f"step {step:5d} | rounds {comm_rounds:5d} | train {tl} | val {val_loss:.4f} "
                  f"| {tps / 1e3:6.1f}k tok/s | {elapsed / 60:5.1f} min", flush=True)
        return not math.isfinite(val_loss)

    if verbose:
        n = make_model(cfg).num_params()
        print(f"run: {name}\nparams: {n / 1e6:.1f}M | device: {cfg.device} | dtype: {cfg.dtype}")

    if cfg.mode == "diloco":
        run_diloco(cfg, train, val, log)
    elif cfg.mode == "baseline":
        run_baseline(cfg, train, val, log)
    else:
        raise ValueError(f"unknown mode {cfg.mode}")

    final = rows[-1]
    summary = dict(config=asdict(cfg), name=name,
                   final_val_loss=final["val_loss"],
                   diverged=not math.isfinite(final["val_loss"]),
                   steps_done=final["step"], comm_rounds=final["comm_rounds"],
                   tokens=final["tokens"], minutes=round((time.time() - t0) / 60, 2),
                   peak_mem_gb=(round(torch.cuda.max_memory_allocated() / 1e9, 2)
                                if cfg.device.startswith("cuda") else None))
    if verbose:
        print(f"done: final val loss {summary['final_val_loss']:.4f} | "
              f"{summary['minutes']} min | peak GPU mem {summary['peak_mem_gb']} GB")
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
            # works as a bare flag (--force) or with a value (--nesterov false)
            p.add_argument(f"--{f.name}", type=lambda s: s.lower() in ("1", "true", "yes"),
                           nargs="?", const=True, default=f.default)
        else:
            t = {"int": int, "float": float, "str": str}.get(f.type, f.type)
            p.add_argument(f"--{f.name}", type=t, default=f.default)
    return TrainConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    run(parse_args())
