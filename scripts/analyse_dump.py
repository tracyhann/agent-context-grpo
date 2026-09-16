#!/usr/bin/env python3
"""Summarise a run's per-sample estimator dump (outputs/ccpo_samples.csv).

    scripts/analyse_dump.py experiments/<exp> [--from-step N]

Answers the questions the README quotes numbers for, against BOTH reference
estimators rather than the one that used to be mislabelled:

  * how closely does A_CC track GiGPO's step credit, and G2PO's?
  * how often does it disagree with them on the SIGN -- the only thing the policy
    gradient sees at first order
  * does the shrinkage ever fire, and where
  * how much of the batch the step term can act on, by bucket size and by level
"""
import argparse
import csv
import os
import sys

import numpy as np


def load(path, from_step=0):
    cols = {}
    with open(path) as fh:
        r = csv.DictReader(fh)
        for name in r.fieldnames or []:
            cols[name] = []
        for row in r:
            # the file is appended to across steps; a torn final line is normal
            try:
                if int(float(row["step"])) < from_step:
                    continue
            except (TypeError, ValueError):
                continue
            for k in cols:
                cols[k].append(row.get(k))
    out = {}
    for k, v in cols.items():
        try:
            out[k] = np.array([float(x) if x not in (None, "") else np.nan for x in v])
        except ValueError:
            out[k] = np.array(v, dtype=object)
    return out


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def sign_disagree(a, b):
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 1e-12) & (np.abs(b) > 1e-12)
    return float("nan") if m.sum() == 0 else float((np.sign(a[m]) != np.sign(b[m])).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp")
    ap.add_argument("--from-step", type=int, default=0)
    a = ap.parse_args()
    path = os.path.join(a.exp, "outputs", "ccpo_samples.csv")
    if not os.path.exists(path):
        sys.exit(f"no dump at {path} (CCPO arms only)")
    d = load(path, a.from_step)
    n = len(d.get("step", []))
    if not n:
        sys.exit("dump is empty")
    steps = d["step"]
    print(f"{os.path.basename(a.exp.rstrip('/'))}: {n} samples over steps "
          f"{int(np.nanmin(steps))}-{int(np.nanmax(steps))}")

    cc = d["adv_cc"]
    print("\nagreement with the reference estimators")
    for name, key in (("GiGPO step credit", "adv_gigpo"), ("G2PO step credit", "adv_g2po")):
        if key not in d:
            continue
        ref = d[key]
        print(f"  {name:<20} corr {corr(cc, ref):+.4f}   sign disagreement "
              f"{sign_disagree(cc, ref) * 100:5.1f}%")
    print("  (corr ~1.0 with a reference means no experiment can separate the two arms)")

    lam = d["lam"]
    fin = np.isfinite(lam)
    print("\nshrinkage")
    print(f"  lambda      mean {np.nanmean(lam):.4f}   >0 on {100 * np.nanmean(lam[fin] > 0):.1f}% "
          f"of samples   >0.5 on {100 * np.nanmean(lam[fin] > 0.5):.1f}%")
    if np.nanmean(lam[fin] > 0) < 1e-6:
        print("  lambda is identically zero: A_CC is the uniform baseline, and the")
        print("  estimator cannot differ from it however long the arm trains.")
    eff = np.abs(d["effect"])
    print(f"  |effect|    mean {np.nanmean(eff):.4f}   p90 {np.nanpercentile(eff, 90):.4f}")
    print(f"  mean A_CC   {np.nanmean(cc):+.4f}  (a group-relative advantage should sit near 0)")

    print("\nsupport")
    J = d["J"]
    ne = d["n_eff"]
    print(f"  reference trajectories J   mean {np.nanmean(J):.2f}  "
          f"p10 {np.nanpercentile(J, 10):.0f}  p90 {np.nanpercentile(J, 90):.0f}")
    print(f"  effective neighbours n_eff mean {np.nanmean(ne):.2f}  "
          f"ratio n_eff/J {np.nanmean(ne / np.maximum(J, 1e-9)):.3f}")
    if "level" in d:
        for lv in sorted(set(int(x) for x in d["level"] if np.isfinite(x))):
            m = d["level"] == lv
            print(f"  level {lv}: {100 * m.mean():5.1f}% of credited samples, "
                  f"lambda {np.nanmean(lam[m]):.3f}, |effect| {np.nanmean(eff[m]):.4f}")

    if "G" in d and "target" in d:
        print("\ntarget")
        print(f"  corr(target, return-to-go) {corr(d['target'], d['G']):+.4f}"
              "   (1.0 means target=return; below it the nextnode target is in force)")


if __name__ == "__main__":
    main()
