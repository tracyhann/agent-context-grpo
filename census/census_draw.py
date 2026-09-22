#!/usr/bin/env python3
"""Offline 128-task draws from a full-coverage census -- no GPU, no API.

    scripts/census_draw.py experiments/census-ws-*/outputs/episodes.jsonl

Once every held-out task has been played exactly once, a sampled evaluation does
not need to be run: drawing 128 task ids and reading off their recorded outcomes
reproduces exactly what the online draw would have measured. The census fixes the
policy's realised result per task, so these draws isolate DRAW variance and say
nothing about decoding noise -- vary the census's rollout.seed for that term.

    ids  = sorted(task ids)
    draw = random.Random(DRAW_SEED).sample(ids, 128)     # without replacement
"""
import argparse
import json
import os
import random
import statistics as st

SUCC = "success_rate"
SCORE = "webshop_task_score (not success_rate)"
DRAW_SEEDS = [2463, 30929, 62186]


def _numpy():
    """--draws and --enumerate need numpy; --seeds does not. Say so plainly rather
    than dying in a traceback three frames deep."""
    try:
        import numpy as np
        return np
    except ImportError:
        raise SystemExit("--draws/--enumerate need numpy; run this with the project "
                         "venv python, or use --seeds which does not.")


def load(path):
    """Census rows keyed by task id, one row per task.

    ALFWorld dumps written before the id fix label a game by its task directory,
    which is not unique -- a few tasks have two trials. Disambiguating by order of
    appearance is exact rather than a guess: census mode pins worker i to game i
    from the same deterministic game_files list, so the k-th row carrying a given
    label is the same game in every arm's dump, which is what makes draws paired
    across arms.
    """
    rows = [json.loads(l) for l in open(path) if l.strip()]
    seen, by = {}, {}
    for r in rows:
        t = r["task"]
        seen[t] = seen.get(t, 0) + 1
        by[t if seen[t] == 1 else f"{t}#{seen[t]}"] = r
    if len(by) != len(rows):
        raise SystemExit(f"{path}: {len(rows)} episodes but {len(by)} keys")
    return by


def stats(rows):
    su = 100 * st.mean(r[SUCC] for r in rows)
    sc = 100 * st.mean(r.get(SCORE, 0) for r in rows)
    return su, sc


def distribution(x, n, B, seed):
    """Sampling distribution of the mean of n drawn from x WITHOUT replacement.

    argpartition over a random matrix gives B independent samples at once: the n
    smallest random keys in each row are a uniform n-subset, which is exactly a
    without-replacement draw and far cheaper than B calls to choice().
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    N = len(x)
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    step = max(1, 2_000_000 // N)          # cap the random matrix at ~2M cells
    for lo in range(0, B, step):
        hi = min(lo + step, B)
        idx = np.argpartition(rng.random((hi - lo, N)), n, axis=1)[:, :n]
        out[lo:hi] = x[idx].mean(axis=1)
    return 100 * out


def theoretical_sd(x, n):
    """SD of the sample mean under simple random sampling without replacement.

    S^2/n * (1 - n/N), with S^2 the finite-population variance (ddof=1). The
    (1 - n/N) term is the finite-population correction: drawing 128 of 500 is
    substantially more precise than drawing 128 from an infinite pool, and
    omitting it overstates the spread by about 16% here.
    """
    import numpy as np
    x = np.asarray(x, dtype=float)
    N = len(x)
    S2 = x.var(ddof=1)
    return 100 * (S2 / n * (1 - n / N)) ** 0.5


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dumps", nargs="+", help="episodes.jsonl from census runs")
    p.add_argument("--n", type=int, default=128, help="tasks per draw")
    p.add_argument("--seeds", type=int, nargs="+", default=DRAW_SEEDS)
    p.add_argument("--draws", type=int, default=0,
                   help="instead of the named seeds, enumerate this many random "
                        "draws and report the sampling distribution")
    p.add_argument("--draw-seed", type=int, default=0,
                   help="RNG seed for --draws, so the distribution is reproducible")
    p.add_argument("--label", nargs="*", default=None,
                   help="row labels, one per dump, in order")
    p.add_argument("--enumerate", type=int, default=0, metavar="N",
                   help="score draw seeds 0..N-1 individually (same recipe as "
                        "--seeds) and report each arm's best and worst few, plus "
                        "where any --seeds land in that distribution")
    p.add_argument("--top", type=int, default=3, help="how many extremes to show")
    a = p.parse_args()

    if a.enumerate:
        np = _numpy()
        labels = a.label or [os.path.basename(os.path.dirname(os.path.dirname(p)))
                             for p in a.dumps]
        for field, title in [(SUCC, "Success%"), (SCORE, "Score%")]:
            print(f"\n{title}:  draw seeds 0..{a.enumerate-1}, {a.n}/500 without replacement")
            print("| arm | full 500 | " + " | ".join(
                f"best #{i+1}" for i in range(a.top)) + " | best-3 mean ± std | "
                + " | ".join(f"s{s_}" for s_ in a.seeds) + " | named mean ± std |")
            print("|---" * (3 + a.top + len(a.seeds)) + "|")
            for lab, path in zip(labels, a.dumps):
                by = load(path)
                ids = sorted(by)
                val = np.array([100 * st.mean(by[i].get(field, 0.0)
                                for i in random.Random(s_).sample(ids, a.n))
                                for s_ in range(a.enumerate)])
                order = np.argsort(-val)[:a.top]
                named = [val[s_] if s_ < a.enumerate else float("nan") for s_ in a.seeds]
                full = 100 * st.mean(by[i].get(field, 0.0) for i in ids)
                cells = [f"{lab}", f"{full:.2f}"]
                cells += [f"{val[o]:.2f} (s{o})" for o in order]
                b = [val[o] for o in order]
                cells.append(f"**{st.mean(b):.2f} ± {st.stdev(b):.2f}**")
                cells += [f"{v:.2f}" for v in named]
                cells.append(f"{st.mean(named):.2f} ± {st.stdev(named):.2f}")
                print("| " + " | ".join(cells) + " |")
            # where the named seeds sit in the enumerated distribution
            print()
            for lab, path in zip(labels, a.dumps):
                by = load(path); ids = sorted(by)
                val = np.array([100 * st.mean(by[i].get(field, 0.0)
                                for i in random.Random(s_).sample(ids, a.n))
                                for s_ in range(a.enumerate)])
                pcts = [(val < val[s_]).mean() * 100 for s_ in a.seeds if s_ < a.enumerate]
                print(f"  {lab}: named seeds sit at percentiles "
                      + ", ".join(f"{p:.1f}" for p in pcts))
        return

    if a.draws:
        np = _numpy()
        labels = a.label or [os.path.basename(os.path.dirname(os.path.dirname(p)))
                             for p in a.dumps]
        for field, title in [(SUCC, "Success%"), (SCORE, "Score%")]:
            print(f"\n{title}:  {a.draws:,} draws of {a.n}/500 without replacement "
                  f"(draw RNG seed {a.draw_seed})")
            print(f"| arm | full 500 | mean | sd | theoretical sd | min | max | p5 | p95 |")
            print(f"|---|---:|---:|---:|---:|---:|---:|---:|---:|")
            for lab, path in zip(labels, a.dumps):
                by = load(path)
                x = [by[i].get(field, 0.0) for i in sorted(by)]
                d = distribution(x, a.n, a.draws, a.draw_seed)
                print(f"| {lab} | {100*np.mean(x):.2f} | {d.mean():.3f} | "
                      f"{d.std(ddof=1):.3f} | {theoretical_sd(x, a.n):.3f} | "
                      f"{d.min():.2f} | {d.max():.2f} | "
                      f"{np.percentile(d, 5):.2f} | {np.percentile(d, 95):.2f} |")
        return

    for path in a.dumps:
        by = load(path)
        ids = sorted(by)
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))
        print(f"\n### {name}   ({len(ids)} tasks, census)")
        print(f"| draw seed | success | score |")
        print(f"|---|---:|---:|")
        su_l, sc_l = [], []
        for s in a.seeds:
            draw = random.Random(s).sample(ids, a.n)
            su, sc = stats([by[i] for i in draw])
            su_l.append(su); sc_l.append(sc)
            print(f"| {s} | {su:.2f} | {sc:.2f} |")
        print(f"| **mean ± std** | **{st.mean(su_l):.2f} ± {st.stdev(su_l):.2f}** "
              f"| **{st.mean(sc_l):.2f} ± {st.stdev(sc_l):.2f}** |")
        fs, fc = stats(list(by.values()))
        print(f"| **full ({len(ids)})** | **{fs:.2f}** | **{fc:.2f}** |")


if __name__ == "__main__":
    main()
