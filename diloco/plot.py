"""
Turn results/ into every figure and table in the paper.

    python plot.py

Writes into figures/ (PDF for the paper, PNG for a quick look):
  heatmap          final val loss over H x outer lr (k=4, 4000 steps)            Fig. 1
  lr_optimum       where the best outer lr sits: fixed steps / fixed rounds / k   Fig. 2
  comm             val loss vs. communication rounds (4000 steps)                 Fig. 3
  workers          gap closed vs. number of workers k                             Fig. 4
  long_budget      gap closed over a 20-tokens-per-parameter run + warm starts    Fig. 5
and LaTeX tables (results, workers, ablation, noniid, scale, warm, fixed_rounds,
lr_by_k, grid), plus numbers.json with every number quoted in the text.

"Gap closed" = (L_solo - L) / (L_solo - L_sync): 0% matches one worker training alone,
100% matches synchronous training on the same tokens.
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
GRID_STEPS = 4000
GRID_HS = [25, 50, 100, 250, 500]
GRID_LRS = [0.1, 0.3, 0.5, 0.7, 1.0]

# config fields added after the first sweep; older runs simply don't have them
DEFAULTS = dict(shard_mode="random", warm_start_steps=0, n_layer=6, n_head=6, n_embd=384,
                outer_momentum=0.9, nesterov=True, seed=0)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3 (validated palette)
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "legend.frameon": False, "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})
NUM = {}   # every number the paper quotes, dumped to numbers.json


def H_COLOR(H):
    """Same light-to-dark blue for a given H in every figure (sequential: H is ordered)."""
    i = GRID_HS.index(H) if H in GRID_HS else len(GRID_HS) - 1
    return matplotlib.colormaps["Blues"](np.linspace(0.45, 0.95, len(GRID_HS)))[i]


# ---------------------------------------------------------------------------
# loading and selecting runs
# ---------------------------------------------------------------------------
def load_runs():
    runs = []
    for jp in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        with open(jp) as f:
            s = json.load(f)
        if s["name"].startswith("timing"):
            continue
        with open(jp[:-5] + ".csv") as f:
            s["curve"] = [{k: (float(v) if v not in ("", None) else None) for k, v in row.items()}
                          for row in csv.DictReader(f)]
        runs.append(s)
    return runs


def c(run, key):
    return run["config"].get(key, DEFAULTS.get(key))


def select(runs, **want):
    """Runs whose config matches every key in `want` (unspecified extras take their defaults).
    Pass seed=None to accept any seed."""
    filt = dict(DEFAULTS)
    filt.update(want)
    out = []
    for r in runs:
        ok = True
        for k, v in filt.items():
            if v is None:
                continue
            if k in ("outer_momentum", "nesterov") and c(r, "mode") == "baseline":
                continue
            rv = c(r, k)
            if isinstance(v, float) or isinstance(rv, float):
                ok = ok and rv is not None and abs(float(rv) - float(v)) < 1e-9
            else:
                ok = ok and rv == v
            if not ok:
                break
        if ok:
            out.append(r)
    return out


def diloco(runs, **kw):
    return select(runs, mode="diloco", **kw)


def baseline(runs, batch_size, steps=GRID_STEPS, **kw):
    return select(runs, mode="baseline", batch_size=batch_size, steps=steps, **kw)


def finals(rs):
    return [r["final_val_loss"] for r in rs if not r["diverged"]]


def mean(rs):
    v = finals(rs)
    return float(np.mean(v)) if v else float("nan")


def sd(rs):
    v = finals(rs)
    return float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")


def fmt(m, s=float("nan")):
    if math.isnan(m):
        return "diverged"
    return f"{m:.3f}" if math.isnan(s) else f"{m:.3f} $\\pm$ {s:.4f}"


def gap(loss, solo, sync):
    if solo == sync:                     # e.g. at step 0, where every run has the same weights
        return 0.0
    return 100.0 * (solo - loss) / (solo - sync)


def curve(r):
    return {int(p["step"]): p["val_loss"] for p in r["curve"]}


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(os.path.join(FIGS, name + ".pdf"))
    fig.savefig(os.path.join(FIGS, name + ".png"), dpi=200)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png")


def write_table(name, header, rows, cols):
    with open(os.path.join(FIGS, name + ".tex"), "w") as f:
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n")
        for row in rows:
            if row == "MIDRULE":
                f.write("\\midrule\n")
            else:
                f.write(" & ".join(str(x) for x in row) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print(f"  wrote figures/{name}.tex")


# the k=4, 4000-step reference baselines (3 seeds each)
def ref_baselines(runs):
    return mean(baseline(runs, 32, seed=None)), mean(baseline(runs, 128, seed=None))


# ---------------------------------------------------------------------------
# Fig. 1 heatmap
# ---------------------------------------------------------------------------
def grid_cells(runs):
    cells = {}
    for H in GRID_HS:
        for lr in GRID_LRS:
            rs = diloco(runs, k=4, H=H, outer_lr=lr, steps=GRID_STEPS, seed=0)
            if rs:
                cells[(H, lr)] = rs
    return cells


def plot_heatmap(runs, cells):
    Z = np.full((len(GRID_HS), len(GRID_LRS)), np.nan)
    for (H, lr), rs in cells.items():
        Z[GRID_HS.index(H), GRID_LRS.index(lr)] = mean(rs)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.grid(False)
    cmap = matplotlib.colormaps["Blues_r"].copy()
    cmap.set_bad("#d9d8d4")
    finite = Z[np.isfinite(Z)]
    vmin, vmax = finite.min(), np.quantile(finite, 0.9)
    vmax = vmax + 0.15 * (vmax - vmin)
    im = ax.imshow(np.ma.masked_invalid(Z), cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    for i in range(len(GRID_HS)):
        for j in range(len(GRID_LRS)):
            if np.isfinite(Z[i, j]):
                dark = Z[i, j] < vmin + 0.55 * (vmax - vmin)
                ax.text(j, i, f"{Z[i, j]:.3f}", ha="center", va="center", fontsize=7.5,
                        color="white" if dark else INK)
    best = np.unravel_index(np.nanargmin(Z), Z.shape)
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
    ax.add_patch(plt.Rectangle((GRID_LRS.index(0.7) - 0.5, GRID_HS.index(500) - 0.5), 1, 1,
                               fill=False, ec=INK, lw=1.0, ls="--"))
    ax.set_xticks(range(len(GRID_LRS)), [str(x) for x in GRID_LRS])
    ax.set_yticks(range(len(GRID_HS)), [str(x) for x in GRID_HS])
    ax.set_xlabel("outer learning rate")
    ax.set_ylabel("inner steps between syncs $H$")
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label("final validation loss (scale capped)")
    cb.outline.set_visible(False)
    save(fig, "heatmap")
    NUM["grid"] = {f"H{H}_lr{lr}": round(float(Z[GRID_HS.index(H), GRID_LRS.index(lr)]), 4)
                   for (H, lr) in cells}


# ---------------------------------------------------------------------------
# Fig. 2 where is the best outer lr?
# ---------------------------------------------------------------------------
def lr_curves(runs):
    """Three families of (label, color, {lr: (mean, sd, n)})."""
    a = []
    for H in (25, 100, 250, 500):
        pts = {lr: diloco(runs, k=4, H=H, outer_lr=lr, steps=GRID_STEPS, seed=None)
               for lr in (0.3, 0.5, 0.7, 1.0)}
        a.append((f"$H$={H}", H_COLOR(H), pts))
    b = []
    for H, st in ((25, 600), (100, 2400), (250, 6000), (500, 12000)):
        pts = {lr: diloco(runs, k=4, H=H, outer_lr=lr, steps=st, seed=None) for lr in (0.5, 0.7, 1.0)}
        b.append((f"$H$={H}", H_COLOR(H), pts))
    cc = []
    for i, k in enumerate((2, 4, 8)):
        pts = {lr: diloco(runs, k=k, H=100, outer_lr=lr, steps=GRID_STEPS, seed=0)
               for lr in (0.3, 0.5, 0.7, 1.0, 1.3)}
        cc.append((f"$k$={k}", SERIES[i], pts))
    return a, b, cc


def plot_lr_optimum(runs):
    panels = lr_curves(runs)
    titles = ["(a) fixed 4000 steps, $k$=4\n(fewer rounds at larger $H$)",
              "(b) fixed 24 rounds, $k$=4\n(same rounds at every $H$)",
              "(c) $H$=100, 4000 steps\nby number of workers"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.0), sharey=True)
    best_lrs = {}
    for ax, title, fam, key in zip(axes, titles, panels, ("fixed_steps", "fixed_rounds", "by_k")):
        best_lrs[key] = {}
        for label, col, pts in fam:
            lrs = [lr for lr in sorted(pts) if pts[lr]]
            m = np.array([mean(pts[lr]) for lr in lrs])
            s = np.array([sd(pts[lr]) if len(pts[lr]) > 1 else 0.0 for lr in lrs])
            d = m - m.min()
            ax.errorbar(lrs, d, yerr=np.where(np.isnan(s), 0, s), color=col, lw=1.8, marker="o",
                        ms=4, capsize=2, label=label)
            ib = int(np.argmin(m))
            ax.plot([lrs[ib]], [0], marker="o", ms=9, mfc="none", mec=INK, mew=1.2)
            best_lrs[key][label.replace("$", "")] = lrs[ib]
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.set_xlabel("outer learning rate")
        ax.legend(fontsize=7, loc="upper right" if key != "by_k" else "upper center")
    axes[0].set_ylabel("loss above the best outer lr")
    axes[0].set_ylim(-0.01, 0.13)
    save(fig, "lr_optimum")
    NUM["best_lr"] = best_lrs


# ---------------------------------------------------------------------------
# Fig. 3 outer-lr preference vs number of outer rounds
# ---------------------------------------------------------------------------
def lr_pref_points(runs):
    """(H, T, L(1.0) - L(0.7), n_seeds_min) for every (H, steps) where both lrs exist (k=4)."""
    pts = []
    for H in GRID_HS:
        for st in sorted({c(r, "steps") for r in diloco(runs, k=4, H=H, seed=None)}):
            a = {c(r, "seed"): r for r in diloco(runs, k=4, H=H, outer_lr=0.7, steps=st, seed=None)}
            b = {c(r, "seed"): r for r in diloco(runs, k=4, H=H, outer_lr=1.0, steps=st, seed=None)}
            seeds = sorted(set(a) & set(b))            # compare like with like: matched seeds only
            if seeds and st % H == 0:
                d = mean([b[x] for x in seeds]) - mean([a[x] for x in seeds])
                pts.append((H, st // H, d, len(seeds)))
    return pts


def plot_lr_rounds(runs):
    pts = lr_pref_points(runs)
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for H in GRID_HS:
        q = sorted((T, d, n) for h, T, d, n in pts if h == H)
        if not q:
            continue
        LO = -0.024                                             # clip off-scale points, label them
        ys = [max(d, LO) for _, d, _ in q]
        ax.plot([T for T, _, _ in q], ys, color=H_COLOR(H), lw=1.4 if H == 100 else 0,
                marker="o", ms=6, label=f"$H$={H}")
        for (T, d, n), y in zip(q, ys):
            if n > 1:
                ax.plot([T], [y], marker="o", ms=10, mfc="none", mec=H_COLOR(H), mew=1.0)
            if d < LO:
                ax.annotate(f"{d:+.2f}", (T, y), xytext=(T * 1.12, y - 0.001), fontsize=7, va="center")
    ax.axhline(0, color=MUTED, lw=1.0)
    ax.set_xscale("log")
    ax.set_xticks([4, 8, 16, 24, 40, 80, 160], ["4", "8", "16", "24", "40", "80", "160"])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 0.045)
    ax.plot([], [], marker="o", ms=10, mfc="none", mec=MUTED, lw=0, label="mean of 2-3 seeds")
    ax.set_xlabel("number of outer rounds $T$ (log scale)")
    ax.set_ylabel("$L$(outer lr 1.0) $-$ $L$(outer lr 0.7)")
    ax.text(40, -0.02, "below 0: lr 1.0 is better", fontsize=7, color=MUTED, ha="center")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "lr_rounds")
    NUM["lr_pref"] = {f"H{H}_T{T}": {"diff": round(d, 4), "n": n} for H, T, d, n in pts}


def write_k1_table(runs):
    solo_rs = baseline(runs, 32, seed=None)
    solo = mean(solo_rs)
    rows, NUM["k1"] = [], {"solo": round(solo, 4)}
    for H in (25, 100, 500):
        vals = {lr: mean(diloco(runs, k=1, H=H, outer_lr=lr, steps=GRID_STEPS)) for lr in (0.4, 0.7)}
        best = min(vals.values())
        rows.append([H, f"{vals[0.4]:.3f}", f"{vals[0.7]:.3f}", f"{100 * (best / solo - 1):+.1f}\\%"])
        NUM["k1"][f"H{H}"] = {**{str(k): round(v, 4) for k, v in vals.items()}, "best_rel": round(100 * (best / solo - 1), 2)}
    rows.append("MIDRULE")
    rows.append(["AdamW (solo baseline)", "\\multicolumn{2}{c}{" + fmt(solo, sd(solo_rs)) + "}", "--"])
    write_table("k1_table", "$H$ & Outer lr 0.4 & Outer lr 0.7 & Best vs.\\ AdamW", rows, "cccc")


# ---------------------------------------------------------------------------
# Fig. 4 loss vs communication rounds
# ---------------------------------------------------------------------------
def plot_comm(runs, cells):
    solo, _ = ref_baselines(runs)
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    for H in GRID_HS:
        cand = [(mean(cells[(H, lr)]), lr) for lr in GRID_LRS if (H, lr) in cells]
        lr = min(cand)[1]
        r = cells[(H, lr)][0]
        pts = [(p["comm_rounds"], p["val_loss"]) for p in r["curve"] if p["comm_rounds"] > 0]
        x, y = zip(*pts)
        ax.plot(x, y, color=H_COLOR(H), lw=1.8, marker="o", ms=2.5, label=f"DiLoCo $H$={H} (lr {lr})")
    sync = baseline(runs, 128, seed=0)[0]
    pts = [(p["comm_rounds"], p["val_loss"]) for p in sync["curve"] if p["comm_rounds"] > 0]
    x, y = zip(*pts)
    ax.plot(x, y, color=MUTED, lw=1.6, ls="--", label="sync baseline (every step)")
    ax.axhline(baseline(runs, 32, seed=0)[0]["final_val_loss"], color=MUTED, lw=1.2, ls=":",
               label="solo baseline (final)")
    ax.set_xscale("log")
    ax.set_xlabel("communication rounds (log scale)")
    ax.set_ylabel("validation loss")
    ax.set_ylim(1.68, 2.9)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "comm")


# ---------------------------------------------------------------------------
# Fig. 4 workers
# ---------------------------------------------------------------------------
def worker_numbers(runs):
    solo = mean(baseline(runs, 32, seed=None))
    out = {}
    for k in (2, 4, 8):
        sync = mean(baseline(runs, 32 * k, seed=None))
        for H in (25, 100, 500):
            rs = diloco(runs, k=k, H=H, outer_lr=0.7, steps=GRID_STEPS, seed=None)
            if rs:
                out[(k, H)] = (mean(rs), gap(mean(rs), solo, sync), sync)
    return solo, out


def plot_workers(runs):
    solo, w = worker_numbers(runs)
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for H in (25, 100, 500):
        ks = [k for k in (2, 4, 8) if (k, H) in w]
        ax.plot(ks, [w[(k, H)][1] for k in ks], color=H_COLOR(H), lw=1.8, marker="o", ms=5,
                label=f"$H$={H}")
    ax.set_xscale("log", base=2)
    ax.set_xticks([2, 4, 8], ["2", "4", "8"])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_xlabel("number of workers $k$")
    ax.set_ylabel("gap closed (%)")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    save(fig, "workers")
    rows = []
    for k in (2, 4, 8):
        cells = [f"{w[(k, 25)][2]:.3f}"] + [f"{w[(k, H)][0]:.3f} ({w[(k, H)][1]:.0f}\\%)" for H in (25, 100, 500)]
        rows.append([k] + cells)
    write_table("workers_table", "$k$ & Sync baseline & DiLoCo $H{=}25$ & DiLoCo $H{=}100$ & DiLoCo $H{=}500$",
                rows, "ccccc")
    NUM["workers"] = {f"k{k}_H{H}": {"loss": round(v[0], 4), "gap": round(v[1], 1)} for (k, H), v in w.items()}
    NUM["workers_sync"] = {f"k{k}": round(w[(k, 25)][2], 4) for k in (2, 4, 8)}


# ---------------------------------------------------------------------------
# Fig. 5 long budget + warm start
# ---------------------------------------------------------------------------
LONG = 18000


def plot_long(runs):
    sync = baseline(runs, 128, steps=LONG)[0]
    solo = baseline(runs, 32, steps=LONG)[0]
    cs, co = curve(sync), curve(solo)
    S, O = sync["final_val_loss"], solo["final_val_loss"]
    entries = []   # (label, color, ls, run)
    for H in (25, 100, 500):
        entries.append((f"$H$={H}", H_COLOR(H), "-", diloco(runs, k=4, H=H, outer_lr=0.7, steps=LONG)[0]))
    for W, ls in ((2000, "--"), (6000, ":")):
        entries.append((f"$H$=500 after {W} sync steps", H_COLOR(500), ls,
                        diloco(runs, k=4, H=500, outer_lr=0.7, steps=LONG, warm_start_steps=W)[0]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 3.2), gridspec_kw={"width_ratios": [1.5, 1]})
    for label, col, ls, r in entries:
        cd = curve(r)
        steps = [s for s in sorted(cd) if s >= 1000 and s in cs and s in co]
        a1.plot(steps, [gap(cd[s], co[s], cs[s]) for s in steps], color=col, ls=ls, lw=1.8, label=label)
    a1.axhline(0, color=MUTED, lw=1.0)
    a1.axhline(100, color=MUTED, lw=1.0, ls="--")
    a1.text(18300, 102, "sync", va="bottom", fontsize=7, color=MUTED)
    a1.text(18300, 2, "solo", va="bottom", fontsize=7, color=MUTED)
    a1.set_xlim(0, 20000)
    a1.set_ylim(-75, 110)
    a1.set_xlabel("inner steps per worker")
    a1.set_ylabel("gap closed (%)")
    a1.set_title("(a) during a 20-tokens-per-parameter run", fontsize=8.5, loc="left")
    a1.legend(fontsize=6.5, loc="lower right", bbox_to_anchor=(0.9, 0.0))
    pts = []
    for label, col, ls, r in entries:
        g = gap(r["final_val_loss"], O, S)
        pts.append((r["comm_rounds"], g, label))
        mk = "o" if ls == "-" else ("s" if ls == "--" else "D")
        a2.plot([r["comm_rounds"]], [g], marker=mk, ms=7, color=col, ls="none")
        short = label.replace("$", "").replace(" after ", "\n+ ").replace(" sync steps", " sync")
        a2.annotate(short, (r["comm_rounds"], g), textcoords="offset points", xytext=(7, -3), fontsize=6.8,
                    color=INK)
    a2.axhline(0, color=MUTED, lw=1.0)
    a2.set_xscale("log")
    a2.set_xlim(20, 80000)
    a2.set_ylim(-15, 85)
    a2.set_xlabel("total communication rounds (log scale)")
    a2.set_ylabel("final gap closed (%)")
    a2.set_title("(b) final result vs. communication used", fontsize=8.5, loc="left")
    save(fig, "long_budget")
    NUM["long"] = {"sync": round(S, 4), "solo": round(O, 4),
                   **{lab.replace("$", ""): {"loss": round(r["final_val_loss"], 4),
                                             "gap": round(gap(r["final_val_loss"], O, S), 1),
                                             "rounds": int(r["comm_rounds"])} for lab, _, _, r in entries}}
    # how the H=500 cold run fades: gap at its evaluation points
    r500 = entries[2][3]
    cd = curve(r500)
    NUM["long_H500_trace"] = {s: round(gap(cd[s], co[s], cs[s]), 1) for s in sorted(cd) if s in cs and s % 2000 == 0}
    rows = [[lab.replace(" after ", ", after ").replace(" sync steps", " sync steps"), f"{r['comm_rounds']:,}".replace(",", "{,}"),
             f"{r['final_val_loss']:.3f}", f"{gap(r['final_val_loss'], O, S):.0f}\\%"] for lab, _, _, r in entries]
    rows += ["MIDRULE", ["Sync baseline", "18{,}000", f"{S:.3f}", "100\\%"], ["Solo baseline", "0", f"{O:.3f}", "0\\%"]]
    write_table("warm_table", "Run & Comm.\\ rounds & Val loss & Gap closed", rows, "lccc")


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def write_results_table(runs):
    solo_rs, sync_rs = baseline(runs, 32, seed=None), baseline(runs, 128, seed=None)
    solo, sync = mean(solo_rs), mean(sync_rs)
    rows = [["Sync baseline", "1", "--", "4", "4{,}000", fmt(sync, sd(sync_rs)), "100\\%"],
            ["Solo baseline", "--", "--", "1", "0", fmt(solo, sd(solo_rs)), "0\\%"]]
    NUM["ref"] = {"sync": round(sync, 4), "sync_sd": round(sd(sync_rs), 4),
                  "solo": round(solo, 4), "solo_sd": round(sd(solo_rs), 4)}
    for H in (25, 100, 500):
        rs = diloco(runs, k=4, H=H, outer_lr=0.7, steps=GRID_STEPS, seed=None)
        m = mean(rs)
        rows.append(["DiLoCo" + (" (original $H$)" if H == 500 else ""), str(H), "0.7", "4",
                     f"{GRID_STEPS // H:,}".replace(",", "{,}"), fmt(m, sd(rs)), f"{gap(m, solo, sync):.0f}\\%"])
        NUM[f"headline_H{H}"] = {"mean": round(m, 4), "sd": round(sd(rs), 4), "gap": round(gap(m, solo, sync), 1), "n": len(rs)}
    write_table("results_table", "Method & $H$ & Outer lr & $k$ & Comm.\\ rounds & Val loss & Gap closed",
                rows, "lcccccc")


def write_grid_table(cells):
    rows = [[H, lr, GRID_STEPS // H, fmt(mean(cells[(H, lr)]))] for (H, lr) in sorted(cells)]
    write_table("grid_table", "$H$ & Outer lr & Comm.\\ rounds & Val loss", rows, "cccc")


def write_fixed_rounds_table(runs):
    rows, NUM["fixed_rounds"] = [], {}
    for H, st in ((25, 600), (100, 2400), (250, 6000), (500, 12000)):
        S = mean(baseline(runs, 128, steps=st))
        O = mean(baseline(runs, 32, steps=st))
        cells = []
        for lr in (0.5, 0.7, 1.0):
            rs = diloco(runs, k=4, H=H, outer_lr=lr, steps=st, seed=None)
            m = mean(rs)
            cells.append(f"{fmt(m, sd(rs))} ({gap(m, O, S):.0f}\\%)")
            NUM["fixed_rounds"][f"H{H}_lr{lr}"] = {"mean": round(m, 4), "sd": round(sd(rs), 4) if len(rs) > 1 else None,
                                                   "n": len(rs), "gap": round(gap(m, O, S), 1)}
        NUM["fixed_rounds"][f"H{H}_base"] = {"sync": round(S, 4), "solo": round(O, 4)}
        rows.append([H, f"{st:,}".replace(",", "{,}"), f"{S:.3f}", f"{O:.3f}"] + cells)
    write_table("fixed_rounds_table",
                "$H$ & Steps & Sync & Solo & Outer lr 0.5 & Outer lr 0.7 & Outer lr 1.0", rows, "ccccccc")


def write_lr_by_k_table(runs):
    rows, NUM["lr_by_k"] = [], {}
    lrs = (0.3, 0.5, 0.7, 1.0, 1.3)
    for k in (2, 4, 8):
        vals = [mean(diloco(runs, k=k, H=100, outer_lr=lr, steps=GRID_STEPS, seed=0)) for lr in lrs]
        b = int(np.nanargmin(vals))
        rows.append([k] + [f"\\textbf{{{v:.3f}}}" if i == b else f"{v:.3f}" for i, v in enumerate(vals)])
        NUM["lr_by_k"][f"k{k}"] = {str(lr): round(v, 4) for lr, v in zip(lrs, vals)}
    write_table("lr_by_k_table", "$k$ & " + " & ".join(f"lr {lr}" for lr in lrs), rows, "cccccc")


def write_ablation(runs):
    rows, NUM["ablation"] = [], {}
    solo, sync = ref_baselines(runs)
    for m, nest, label in [(0.9, True, "Nesterov, momentum 0.9 (default)"),
                           (0.9, False, "Plain momentum 0.9"), (0.0, False, "No momentum")]:
        for lr in (0.7, 1.0):
            rs = diloco(runs, k=4, H=100, outer_lr=lr, steps=GRID_STEPS, seed=0,
                        outer_momentum=m, nesterov=nest)
            if rs:
                rows.append([label, lr, f"{mean(rs):.3f}", f"{gap(mean(rs), solo, sync):.0f}\\%"])
                NUM["ablation"][f"{label}_lr{lr}"] = round(mean(rs), 4)
    write_table("ablation_table", "Outer optimizer & Outer lr & Val loss & Gap closed", rows, "lccc")


def write_noniid_table(runs):
    rows, NUM["noniid"] = [], {}
    solo, sync = ref_baselines(runs)
    for H in (25, 100, 500):
        for lr in (0.7, 1.0):
            a = mean(diloco(runs, k=4, H=H, outer_lr=lr, steps=GRID_STEPS, seed=0))
            b = mean(diloco(runs, k=4, H=H, outer_lr=lr, steps=GRID_STEPS, shard_mode="length"))
            rows.append(["4{,}000", H, lr, f"{a:.3f} ({gap(a, solo, sync):.0f}\\%)",
                         f"{b:.3f} ({gap(b, solo, sync):.0f}\\%)", f"{b - a:+.3f}"])
            NUM["noniid"][f"short_H{H}_lr{lr}"] = {"random": round(a, 4), "length": round(b, 4), "diff": round(b - a, 4),
                                                   "gap_random": round(gap(a, solo, sync), 1), "gap_length": round(gap(b, solo, sync), 1)}
    rows.append("MIDRULE")
    S, O = mean(baseline(runs, 128, steps=LONG)), mean(baseline(runs, 32, steps=LONG))
    for H in (25, 100):
        a = mean(diloco(runs, k=4, H=H, outer_lr=0.7, steps=LONG))
        b = mean(diloco(runs, k=4, H=H, outer_lr=0.7, steps=LONG, shard_mode="length"))
        rows.append(["18{,}000", H, 0.7, f"{a:.3f} ({gap(a, O, S):.0f}\\%)", f"{b:.3f} ({gap(b, O, S):.0f}\\%)", f"{b - a:+.3f}"])
        NUM["noniid"][f"long_H{H}"] = {"random": round(a, 4), "length": round(b, 4), "diff": round(b - a, 4),
                                       "gap_random": round(gap(a, O, S), 1), "gap_length": round(gap(b, O, S), 1)}
    write_table("noniid_table", "Steps & $H$ & Outer lr & Same data (gap) & Different data (gap) & Cost",
                rows, "cccccc")


def write_scale_table(runs):
    rows, NUM["scale"] = [], {}
    for size, kw in (("30M", {}), ("124M", dict(n_layer=12, n_head=12, n_embd=768))):
        S = mean(baseline(runs, 64, **kw))
        O = mean(baseline(runs, 32, seed=None if not kw else 0, **kw))
        cells = []
        for H in (100, 500):
            m = mean(diloco(runs, k=2, H=H, outer_lr=0.7, steps=GRID_STEPS, **kw))
            cells.append(f"{m:.3f} ({gap(m, O, S):.0f}\\%)")
            NUM["scale"][f"{size}_H{H}"] = {"loss": round(m, 4), "gap": round(gap(m, O, S), 1)}
        NUM["scale"][f"{size}_base"] = {"sync": round(S, 4), "solo": round(O, 4)}
        rows.append([size, f"{S:.3f}", f"{O:.3f}"] + cells)
    write_table("scale_table", "Model & Sync ($k{=}2$) & Solo & DiLoCo $H{=}100$ & DiLoCo $H{=}500$", rows, "ccccc")
    # gap-closed trace during the 124M H=100 run
    kw = dict(n_layer=12, n_head=12, n_embd=768)
    cs, co = curve(baseline(runs, 64, **kw)[0]), curve(baseline(runs, 32, **kw)[0])
    cd = curve(diloco(runs, k=2, H=100, outer_lr=0.7, steps=GRID_STEPS, **kw)[0])
    NUM["scale_trace_124M_H100"] = {s: round(gap(cd[s], co[s], cs[s]), 1) for s in (1000, 2000, 3000, 4000)}


def main():
    runs = load_runs()
    print(f"{len(runs)} runs in {RESULTS}/")
    os.makedirs(FIGS, exist_ok=True)
    cells = grid_cells(runs)
    plot_heatmap(runs, cells)
    plot_lr_optimum(runs)
    plot_lr_rounds(runs)
    plot_comm(runs, cells)
    plot_workers(runs)
    plot_long(runs)
    write_results_table(runs)
    write_grid_table(cells)
    write_fixed_rounds_table(runs)
    write_lr_by_k_table(runs)
    write_ablation(runs)
    write_noniid_table(runs)
    write_scale_table(runs)
    write_k1_table(runs)
    with open(os.path.join(FIGS, "numbers.json"), "w") as f:
        json.dump(NUM, f, indent=1, default=str)
    print("  wrote figures/numbers.json")


if __name__ == "__main__":
    main()
