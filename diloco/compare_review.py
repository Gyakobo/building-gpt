"""
Compare a reviewer's runs (results_review/) with the released logs (results/).

    python sweep.py configs/review_quick.yaml
    python compare_review.py

GPU kernels in bfloat16 are not bit-for-bit deterministic across machines, so final losses
can differ slightly. Seed-to-seed standard deviations in the paper are up to 0.0045; we flag
differences above 0.015. The qualitative checks at the bottom are what the paper's claims rest on.
"""
import glob
import json
import os
import sys

TOL = 0.015


def load(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*.json")):
        with open(p) as f:
            out[os.path.basename(p)[:-5]] = json.load(f)["final_val_loss"]
    return out


def main(review="results_review", released="results"):
    rev, rel = load(review), load(released)
    if not rev:
        sys.exit(f"no runs in {review}/ -- run: python sweep.py configs/review_quick.yaml")
    print(f"{'run':55s} {'released':>9s} {'yours':>9s} {'diff':>8s}")
    bad = 0
    for name in sorted(rev):
        if name not in rel:
            print(f"{name:55s} {'--':>9s} {rev[name]:9.4f}")
            continue
        d = rev[name] - rel[name]
        flag = "  <-- larger than expected" if abs(d) > TOL else ""
        bad += bool(flag)
        print(f"{name:55s} {rel[name]:9.4f} {rev[name]:9.4f} {d:+8.4f}{flag}")

    def g(n):
        return rev.get(n)
    checks = [
        ("loss grows with H (H=25 < H=100 < H=500, k=4, lr 0.7)",
         ["diloco_k4_H25_olr0.7_m0.9_nest_steps4000_s0", "diloco_k4_H100_olr0.7_m0.9_nest_steps4000_s0",
          "diloco_k4_H500_olr0.7_m0.9_nest_steps4000_s0"], lambda a, b, c: a < b < c),
        ("4 rounds (H=100): outer lr 1.0 beats 0.7",
         ["diloco_k4_H100_olr1.0_m0.9_nest_steps400_s0", "diloco_k4_H100_olr0.7_m0.9_nest_steps400_s0"],
         lambda a, b: a < b),
        ("8 rounds (H=100): outer lr 1.0 beats 0.7",
         ["diloco_k4_H100_olr1.0_m0.9_nest_steps800_s0", "diloco_k4_H100_olr0.7_m0.9_nest_steps800_s0"],
         lambda a, b: a < b),
        ("40 rounds (H=100): outer lr 0.7 beats 1.0",
         ["diloco_k4_H100_olr0.7_m0.9_nest_steps4000_s0", "diloco_k4_H100_olr1.0_m0.9_nest_steps4000_s0"],
         lambda a, b: a < b),
        ("one worker at H=25 beats AdamW (solo baseline)",
         ["diloco_k1_H25_olr0.4_m0.9_nest_steps4000_s0", "baseline_bs32_steps4000_s0"], lambda a, b: a < b),
        ("one worker at H=100 does not beat AdamW",
         ["diloco_k1_H100_olr0.7_m0.9_nest_steps4000_s0", "baseline_bs32_steps4000_s0"], lambda a, b: a > b),
    ]
    print()
    for text, names, ok in checks:
        vals = [g(n) for n in names]
        status = "missing runs" if None in vals else ("OK" if ok(*vals) else "NOT REPRODUCED")
        print(f"[{status:>14s}] {text}")
    print(f"\n{bad} run(s) differ from the released logs by more than {TOL}")


if __name__ == "__main__":
    main(*sys.argv[1:])
