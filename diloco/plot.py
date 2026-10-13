"""
Turn results/ into the paper's figures and tables.

    python plot.py

Writes into figures/:
  heatmap.pdf / .png          final val loss over H x outer_lr (k=4)          -> Figure 2
  comm.pdf / .png             val loss vs. communication rounds               -> Figure 3
  workers.pdf / .png          final val loss vs. number of workers k          -> Figure 4 (once those runs exist)
  results_table.tex           headline table, mean +- std over seeds          -> \\input{} in the paper
  grid_table.tex              every grid run                                  -> Appendix B
Figures that don't have their runs yet are skipped.
"""
import csv
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS, FIGS = "results", "figures"
GRID_STEPS = 4000   # must match configs/grid.yaml

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------------
def load_runs():
    runs = []
    for jp in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        with open(jp) as f:
            s = json.load(f)
        if s["name"] == "timing" or s["config"].get("name") == "timing":
            continue
        with open(jp[:-5] + ".csv") as f:
            s["curve"] = [{k: (float(v) if v not in ("", None) else None) for k, v in row.items()}
                          for row in csv.DictReader(f)]
        runs.append(s)
    return runs


def c(run, key):
    return run["config"][key]


def is_diloco(r, k=4, steps=GRID_STEPS, momentum=0.9, nesterov=True):
    return (c(r, "mode") == "diloco" and c(r, "k") == k and c(r, "steps") == steps
            and c(r, "outer_momentum") == momentum and c(r, "nesterov") == nesterov)


def baseline(runs, batch_size, steps=GRID_STEPS):
    return [r for r in runs if c(r, "mode") == "baseline" and c(r, "batch_size") == batch_size
            and c(r, "steps") == steps]


def stats(rs):
    v = [r["final_val_loss"] for r in rs if not r["diverged"]]
    if not v:
        return float("nan"), float("nan"), len(rs)
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")), len(v)


def fmt(mean, std):
    if math.isnan(mean):
        return "diverged"
    return f"{mean:.3f}" if math.isnan(std) else f"{mean:.3f} $\\pm$ {std:.3f}"


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png")


# ---------------------------------------------------------------------------
def grid_cells(runs):
    """{(H, outer_lr): [runs]} for the main k=4 grid."""
    cells = defaultdict(list)
    for r in runs:
        if is_diloco(r):
            cells[(c(r, "H"), c(r, "outer_lr"))].append(r)
    return cells


def plot_heatmap(runs):
    cells = grid_cells(runs)
    if not cells:
        print("  heatmap: no grid runs yet, skipped")
        return None
    Hs = sorted({h for h, _ in cells})
    lrs = sorted({lr for _, lr in cells})
    Z = np.full((len(Hs), len(lrs)), np.nan)
    for (h, lr), rs in cells.items():
        Z[Hs.index(h), lrs.index(lr)] = stats(rs)[0]

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.grid(False)
    cmap = matplotlib.colormaps["Blues_r"].copy()   # one hue: darker = lower (better) loss
    cmap.set_bad("#d9d8d4")
    finite = Z[np.isfinite(Z)]
    # cap the scale at the 90th percentile so one very bad cell doesn't wash out the rest;
    # the printed numbers are always exact
    vmin, vmax = finite.min(), np.quantile(finite, 0.9)
    vmax = vmax + 0.15 * (vmax - vmin + 1e-9)
    im = ax.imshow(np.ma.masked_invalid(Z), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(len(Hs)):
        for j in range(len(lrs)):
            done = (Hs[i], lrs[j]) in cells
            txt = "div." if done and np.isnan(Z[i, j]) else ("" if not done else f"{Z[i, j]:.3f}")
            dark = done and np.isfinite(Z[i, j]) and Z[i, j] < vmin + 0.55 * (vmax - vmin)
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color="white" if dark else INK)
    best = np.unravel_index(np.nanargmin(Z), Z.shape)
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
    if 500 in Hs and 0.7 in lrs:   # original DiLoCo setting
        ax.add_patch(plt.Rectangle((lrs.index(0.7) - 0.5, Hs.index(500) - 0.5), 1, 1,
                                   fill=False, ec=INK, lw=1.0, ls="--"))
    ax.set_xticks(range(len(lrs)), [str(x) for x in lrs])
    ax.set_yticks(range(len(Hs)), [str(x) for x in Hs])
    ax.set_xlabel("outer learning rate")
    ax.set_ylabel("inner steps between syncs $H$")
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("final validation loss (scale capped)" if finite.max() > vmax else "final validation loss")
    cb.outline.set_visible(False)
    save(fig, "heatmap")
    return {"best": (Hs[best[0]], lrs[best[1]]), "best_loss": float(Z[best])}


def plot_comm(runs, best_lr_per_H):
    """Val loss vs. communication rounds: best outer_lr for each H, plus baselines."""
    sync = baseline(runs, 128)
    solo = baseline(runs, 32)
    if not best_lr_per_H:
        print("  comm: no grid runs yet, skipped")
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    Hs = sorted(best_lr_per_H)
    shades = matplotlib.colormaps["Blues"](np.linspace(0.45, 0.95, len(Hs)))  # one hue, ordered by H
    for H, col in zip(Hs, shades):
        r = best_lr_per_H[H]
        pts = [(p["comm_rounds"], p["val_loss"]) for p in r["curve"] if p["comm_rounds"] > 0]
        x, y = zip(*pts)
        ax.plot(x, y, color=col, lw=1.8, marker="o", ms=3,
                label=f"DiLoCo $H$={H} (outer lr {c(r, 'outer_lr')})")
    if sync:
        pts = [(p["comm_rounds"], p["val_loss"]) for p in sync[0]["curve"] if p["comm_rounds"] > 0]
        x, y = zip(*pts)
        ax.plot(x, y, color=MUTED, lw=1.6, ls="--", label="sync baseline (every step)")
    if solo:
        ax.axhline(solo[0]["final_val_loss"], color=MUTED, lw=1.2, ls=":",
                   label="solo baseline (final, no communication)")
    ax.set_xscale("log")
    ax.set_xlabel("communication rounds (log scale)")
    ax.set_ylabel("validation loss")
    lo = min(r["final_val_loss"] for r in best_lr_per_H.values() if not r["diverged"])
    ax.set_ylim(lo - 0.1, lo + 1.2)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "comm")


WORKER_CONFIGS = [(25, 0.7), (100, 0.7), (500, 0.7)]   # must match configs/workers.yaml


def plot_workers(runs):
    """Final val loss vs. k for a few (H, outer_lr) configs, plus the sync baseline at each k."""
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    drawn = 0
    for H, lr in WORKER_CONFIGS:
        by_k = defaultdict(list)
        for r in runs:
            if (c(r, "mode") == "diloco" and c(r, "H") == H and c(r, "outer_lr") == lr
                    and c(r, "outer_momentum") == 0.9 and c(r, "nesterov") and c(r, "steps") == GRID_STEPS):
                by_k[c(r, "k")].append(r)
        if len(by_k) < 2:
            continue
        ks = sorted(by_k)
        m = [stats(by_k[k])[0] for k in ks]
        ax.plot(ks, m, color=SERIES[drawn % len(SERIES)], lw=1.8, marker="o", ms=5,
                label=f"DiLoCo $H$={H}, outer lr {lr}")
        drawn += 1
    if not drawn:
        plt.close(fig)
        print("  workers: needs runs at k=2 and k=8 too, skipped")
        return
    sync = {k: baseline(runs, 32 * k) for k in (2, 4, 8)}
    ks = [k for k in (2, 4, 8) if sync[k]]
    if len(ks) >= 2:
        ax.plot(ks, [stats(sync[k])[0] for k in ks], color=MUTED, lw=1.4, ls="--", marker="s", ms=4,
                label="sync baseline (batch $32k$)")
    solo = baseline(runs, 32)
    if solo:
        ax.axhline(stats(solo)[0], color=MUTED, lw=1.2, ls=":", label="solo baseline")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8], ["2", "4", "8"])
    ax.minorticks_off()
    ax.set_xlabel("number of workers $k$")
    ax.set_ylabel("final validation loss")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "workers")


def write_ablation(runs):
    rows = []
    for m, nest, label in [(0.9, True, "Nesterov, momentum 0.9 (default)"),
                           (0.9, False, "plain momentum 0.9"),
                           (0.0, False, "no momentum")]:
        for lr in (0.7, 1.0):
            rs = [r for r in runs if is_diloco(r, momentum=m, nesterov=nest)
                  and c(r, "H") == 100 and c(r, "outer_lr") == lr and c(r, "seed") == 0]
            if rs:
                rows.append((label, str(lr), fmt(*stats(rs)[:2])))
    if len(rows) <= 2:
        print("  ablation: needs configs/ablation.yaml runs, skipped")
        return
    with open(os.path.join(FIGS, "ablation_table.tex"), "w") as f:
        f.write("\\begin{tabular}{lcc}\n\\toprule\nOuter optimizer & Outer lr & Val loss \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  wrote figures/ablation_table.tex")


# ---------------------------------------------------------------------------
def write_tables(runs, cells, best):
    os.makedirs(FIGS, exist_ok=True)
    rows = []
    sync, solo = baseline(runs, 128), baseline(runs, 32)
    if sync:
        rows.append(("Sync baseline", "1", "--", "4", f"{sync[0]['comm_rounds']:,}", fmt(*stats(sync)[:2])))
    if solo:
        rows.append(("Solo baseline", "--", "--", "1", "0", fmt(*stats(solo)[:2])))
    if (500, 0.7) in cells:
        rs = cells[(500, 0.7)]
        rows.append(("DiLoCo (original settings)", "500", "0.7", "4", f"{rs[0]['comm_rounds']:,}",
                     fmt(*stats(rs)[:2])))
    if best and best["best"] != (500, 0.7):
        rs = cells[best["best"]]
        rows.append(("DiLoCo (best found)", str(best["best"][0]), str(best["best"][1]), "4",
                     f"{rs[0]['comm_rounds']:,}", fmt(*stats(rs)[:2])))
    with open(os.path.join(FIGS, "results_table.tex"), "w") as f:
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n"
                "Method & $H$ & Outer lr & $k$ & Comm.\\ rounds & Val loss \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(r).replace(",", "{,}") + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  wrote figures/results_table.tex")

    with open(os.path.join(FIGS, "grid_table.tex"), "w") as f:
        f.write("\\begin{tabular}{cccc}\n\\toprule\n$H$ & Outer lr & Comm.\\ rounds & Val loss \\\\\n\\midrule\n")
        for (H, lr) in sorted(cells):
            rs = cells[(H, lr)]
            f.write(f"{H} & {lr} & {rs[0]['comm_rounds']} & {fmt(*stats(rs)[:2])} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("  wrote figures/grid_table.tex")
    return rows


def main():
    runs = load_runs()
    print(f"{len(runs)} finished runs in {RESULTS}/")
    best = plot_heatmap(runs)
    cells = grid_cells(runs)

    best_lr_per_H = {}
    for (H, lr), rs in cells.items():
        r = rs[0]
        if r["diverged"]:
            continue
        if H not in best_lr_per_H or r["final_val_loss"] < best_lr_per_H[H]["final_val_loss"]:
            best_lr_per_H[H] = r
    plot_comm(runs, best_lr_per_H)

    plot_workers(runs)
    rows = write_tables(runs, cells, best)
    write_ablation(runs)

    print("\nsummary")
    for r in rows:
        val = r[5].replace(" $\\pm$ ", " +- ")
        print(f"  {r[0]:28s} H={r[1]:>4s} lr={r[2]:>4s} k={r[3]} rounds={r[4]:>7s}  val {val}")
    if best:
        print(f"  best grid cell: H={best['best'][0]}, outer_lr={best['best'][1]} -> {best['best_loss']:.4f}")


if __name__ == "__main__":
    main()
