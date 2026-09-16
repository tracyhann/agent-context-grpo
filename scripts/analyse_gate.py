#!/usr/bin/env python3
"""Compare candidate anchor-state gates on a grouping dump, offline.

    scripts/analyse_gate.py experiments/<exp>/outputs/grouping.jsonl

The gate decides which occurrences share a bucket, and therefore which get any
step credit at all. That is a different quantity from how well phi predicts
WITHIN a bucket, which H-H bounds at 6.1% of target variance -- so a gate change
is not capped by that ceiling and is worth measuring separately.

Three numbers per candidate gate:

  dead            share of occurrences in buckets with FEWER THAN TWO DISTINCT
                  TRAJECTORIES. Leave-one-out has nothing to compare these against,
                  so they receive A_CC = 0 under every estimator. Counting bare
                  occurrences instead of distinct trajectories understates this
                  badly and disagrees with the trainer's own metric.
  n_traj          mean number of distinct trajectories per usable bucket.
  ICC             between-trajectory share of within-bucket target variance,
                  bias-corrected against a within-bucket label permutation at
                  matched group sizes. This is the part leave-one-out can exploit,
                  so it is the quantity a better gate should RAISE. A gate that
                  merges unrelated states will lower it while lowering singletons,
                  which is why singleton_frac alone is not a sufficient criterion.
"""
import argparse
import json
import sys
from collections import defaultdict

import numpy as np


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue          # torn final line while the run is writing
    return rows


def visited_signature(rows):
    """Set of DISTINCT observations the trajectory has seen up to and including t.

    In a deterministic environment entered from a fixed start, the set of states
    visited so far is a sufficient statistic for the current state -- the
    principled node identity both GiGPO and G2PO are missing, since they use the
    current observation string alone. Order-invariant on purpose: it keeps
    refined buckets populated rather than shattering them by path.
    """
    by_traj = defaultdict(list)
    for i, r in enumerate(rows):
        by_traj[(r["step"], r["traj_uid"])].append((r["t"], i))
    sig = [None] * len(rows)
    for _, items in by_traj.items():
        items.sort()
        seen = set()
        for _, i in items:
            seen.add(rows[i]["obs"])
            sig[i] = hash(frozenset(seen)) & 0xFFFFFFFF
    return sig


def band(x, edges):
    return int(np.searchsorted(edges, x))


def icc_one(groups):
    allv = np.concatenate([np.asarray(g, float) for g in groups])
    N, k = len(allv), len(groups)
    if k < 2 or N - k < 1:
        return np.nan
    gm = allv.mean()
    ns = np.array([len(g) for g in groups])
    msb = sum(len(g) * (np.mean(g) - gm) ** 2 for g in groups) / (k - 1)
    msw = sum(((np.asarray(g, float) - np.mean(g)) ** 2).sum() for g in groups) / (N - k)
    n0 = (N - (ns ** 2).sum() / N) / (k - 1)
    if n0 <= 0:
        return np.nan
    v = (msb - msw) / n0
    return v / (v + msw) if (v + msw) > 1e-12 else np.nan


def evaluate(rows, keyfn, name, n_perm=120, seed=0):
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[keyfn(r, i)].append(i)
    n = len(rows)
    # "Usable" means what LEAVE-ONE-OUT needs: at least two DISTINCT trajectories,
    # so an occurrence has some other trajectory to be compared against. Counting
    # raw occurrences instead understates the dead weight badly, because several
    # occurrences of one trajectory in a bucket still give LOO nothing -- that is
    # the difference between 0.05 and the 0.35 the trainer reports.
    ntraj = {k: len({rows[i]["traj_uid"] for i in v}) for k, v in buckets.items()}
    singles = sum(len(v) for k, v in buckets.items() if ntraj[k] < 2)
    usable = np.array([ntraj[k] for k in buckets if ntraj[k] >= 2], float)

    units = []
    for idxs in buckets.values():
        if len(idxs) < 3:
            continue
        by_t = defaultdict(list)
        for i in idxs:
            by_t[rows[i]["traj_uid"]].append(rows[i]["target"])
        if len(by_t) < 2:
            continue
        val = icc_one(list(by_t.values()))
        if np.isfinite(val):
            units.append(([len(x) for x in by_t.values()],
                          np.array([rows[i]["target"] for i in idxs], float), val))
    if not units:
        return dict(name=name, n_buckets=len(buckets), singleton=singles / n,
                    n_eff=float(usable.mean()) if len(usable) else 0.0, icc=np.nan, k=0)

    real = float(np.mean([u[2] for u in units]))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for p in range(n_perm):
        vals = np.empty(len(units))
        for j, (gs, tg, _) in enumerate(units):
            perm = rng.permutation(tg)
            c, grp = 0, []
            for s in gs:
                grp.append(perm[c:c + s]); c += s
            vals[j] = icc_one(grp)
        null[p] = np.nanmean(vals)
    return dict(name=name, n_buckets=len(buckets), singleton=singles / n,
                n_eff=float(usable.mean()) if len(usable) else 0.0,
                icc=real - float(null.mean()), k=len(units))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--perm", type=int, default=120)
    a = ap.parse_args()

    rows = load(a.dump)
    if not rows:
        sys.exit(f"no usable rows in {a.dump}")
    print(f"{len(rows)} occurrences, {len({r['step'] for r in rows})} step(s)\n")

    sig = visited_signature(rows)
    # Progress bands chosen to split the horizon into early/mid/late thirds; the
    # point is only that returns-to-go differ systematically across them.
    t_edges = [8, 20]

    gates = [
        (lambda r, i: (r["uid"], r["obs"]),
         "baseline: (task, obs)            [GiGPO / G2PO]"),
        (lambda r, i: (r["uid"], r["obs"], band(r["t"], t_edges)),
         "+ progress band                  [fixes time-blindness]"),
        (lambda r, i: (r["uid"], sig[i]),
         "visited-set signature            [replaces obs as node id]"),
        (lambda r, i: (r["uid"], sig[i], band(r["t"], t_edges)),
         "visited-set + progress band      [combined]"),
        (lambda r, i: (r["uid"], r["obs"], r["revisit"]),
         "+ revisit flag"),
    ]
    res = [evaluate(rows, f, nm, n_perm=a.perm) for f, nm in gates]

    print(f"{'gate':<44} {'buckets':>8} {'dead':>7} {'n_traj':>7} {'ICC':>8} {'k':>5}")
    print("-" * 84)
    for r in res:
        flag = "" if r["k"] >= 40 else "   <- k too small to trust"
        print(f"{r['name']:<44} {r['n_buckets']:>8} {r['singleton']:>7.3f} "
              f"{r['n_eff']:>7.2f} {r['icc']:>8.4f} {r['k']:>5}{flag}")
    base = res[0]
    print("\nvs baseline (only rows with k >= 40 are interpretable):")
    for r in res[1:]:
        if r["k"] < 40:
            print(f"  {r['name'][:42]:<44} SKIPPED (k={r['k']})")
            continue
        print(f"  {r['name'][:42]:<44} dead {r['singleton']-base['singleton']:+.3f}  "
              f"n_traj {r['n_eff']-base['n_eff']:+.2f}  ICC {r['icc']-base['icc']:+.4f}")


if __name__ == "__main__":
    main()
