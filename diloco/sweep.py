"""
Run every config in one or more sweep files, one run at a time.

    python sweep.py configs/baselines.yaml configs/grid.yaml
    python sweep.py configs/grid.yaml --dry_run      # list the runs + time estimate, run nothing

A sweep file has a `base` (settings shared by every run) plus either
  - `sweep`: lists of values; every combination becomes one run, and/or
  - `runs`:  an explicit list of runs (each merged on top of `base`).

Each run is a separate `python diloco.py ...` process, so one crash (or out-of-memory)
doesn't kill the sweep. Finished runs are skipped, so stopping with Ctrl+C and
re-running the same command picks up where it left off.
"""
import argparse
import itertools
import subprocess
import sys
import time

import yaml

# measured on the RTX 3070 Ti: ~83k tokens/s with batch 32 x 256 tokens per worker step
SEC_PER_WORKER_STEP = 32 * 256 / 83_000


def expand(path):
    with open(path) as f:
        spec = yaml.safe_load(f)
    base = spec.get("base", {}) or {}
    runs = []
    if "sweep" in spec:
        keys = list(spec["sweep"].keys())
        for values in itertools.product(*[spec["sweep"][k] for k in keys]):
            runs.append({**base, **dict(zip(keys, values))})
    for r in spec.get("runs", []) or []:
        runs.append({**base, **r})
    return runs


def to_args(run):
    args = []
    for k, v in run.items():
        args += [f"--{k}", str(v)]
    return args


def est_minutes(run):
    workers = run.get("k", 4) if run.get("mode", "diloco") == "diloco" else 1
    batch_scale = run.get("batch_size", 32) / 32
    return run.get("steps", 3000) * workers * batch_scale * SEC_PER_WORKER_STEP / 60


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configs", nargs="+")
    p.add_argument("--dry_run", action="store_true")
    a = p.parse_args()

    runs = [r for c in a.configs for r in expand(c)]
    total = sum(est_minutes(r) for r in runs)
    print(f"{len(runs)} runs, about {total / 60:.1f} GPU-hours if none are already done\n")
    for i, r in enumerate(runs, 1):
        print(f"  {i:3d}. ~{est_minutes(r):4.0f} min  {r}")
    if a.dry_run:
        return

    failed, t0 = [], time.time()
    for i, r in enumerate(runs, 1):
        print(f"\n===== [{i}/{len(runs)}] {r}  (sweep time so far {(time.time() - t0) / 3600:.1f} h)",
              flush=True)
        code = subprocess.call([sys.executable, "diloco.py", *to_args(r)])
        if code != 0:
            print(f"!!!!! run failed with exit code {code}, continuing", flush=True)
            failed.append(r)
    print(f"\nsweep finished in {(time.time() - t0) / 3600:.1f} h, {len(failed)} failed")
    for r in failed:
        print("  failed:", r)


if __name__ == "__main__":
    main()