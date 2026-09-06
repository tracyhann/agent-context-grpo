#!/usr/bin/env python3
"""One-line-per-run status across every experiment, and the comparison table.

    scripts/exp_status.py                 # all runs
    scripts/exp_status.py --md            # markdown, for experiments/README.md
"""
import argparse
import glob
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# published ALFWorld / Qwen2.5-1.5B-Instruct reference points (HGPO Table 1)
REFERENCE = {"GRPO": (72.8, 70.1), "GiGPO (K=2)": (90.16, 84.76),
             "HGPO (K=2)": (92.77, 90.16), "GiGPO (K=4)": (93.29, 91.53),
             "HGPO (K=4)": (94.85, 92.12)}


def rows(exp):
    p = os.path.join(exp, "outputs", "metrics.jsonl")
    out = []
    if os.path.exists(p):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return out


def pick(r, *cands):
    for c in cands:
        if c in r and isinstance(r[c], (int, float)):
            return r[c]
    for c in cands:
        for k in r:
            if k.endswith(c) and isinstance(r[k], (int, float)):
                return r[k]
    return None


def summarise(exp):
    cfg_p = os.path.join(exp, "config.json")
    cfg = json.load(open(cfg_p))["config"] if os.path.exists(cfg_p) else {}
    rs = rows(exp)
    d = {"exp": os.path.basename(exp.rstrip("/")), "arm": cfg.get("arm", "?"),
         "steps": rs[-1]["step"] if rs else 0}
    vals = [(r["step"], pick(r, "val/success_rate", "val-core/success_rate"))
            for r in rs]
    vals = [(s, v) for s, v in vals if v is not None]
    d["n_eval"] = len(vals)
    d["last_val"] = vals[-1][1] if vals else None
    d["best_val"] = max((v for _, v in vals), default=None)
    d["best_step"] = max(vals, key=lambda t: t[1])[0] if vals else None
    d["mean_last3"] = (sum(v for _, v in vals[-3:]) / len(vals[-3:])) if vals else None
    last = rs[-1] if rs else {}
    for k, short in [("ccpo/live_frac", "live"), ("ccpo/lam_u_mean", "lam"),
                     ("ccpo/n_eff_mean", "n_eff"), ("ccpo/effect_rel", "eff_rel"),
                     ("ccpo/r_vs_gigpo", "r_gigpo"), ("ccpo/r_vs_g2po", "r_g2po"),
                     ("ccpo/bucket_singleton_frac", "singleton"),
                     ("ccpo/adv_ep_over_cc", "ep/cc"),
                     ("response_length/mean", "len"),
                     ("episode/valid_action_ratio", "valid")]:
        d[short] = last.get(k)
    bj = os.path.join(exp, "outputs", "checkpoints", "best.json")
    d["ckpt_best"] = json.load(open(bj)).get("step") if os.path.exists(bj) else None
    pid = os.path.join(exp, "outputs", "train.pid")
    alive = False
    if os.path.exists(pid):
        try:
            os.kill(int(open(pid).read().strip()), 0)
            alive = True
        except (OSError, ValueError):
            alive = False
    d["state"] = "running" if alive else ("done" if rs else "not started")
    mp = os.path.join(exp, "outputs", "metrics.jsonl")
    d["age_min"] = ((time.time() - os.path.getmtime(mp)) / 60) if os.path.exists(mp) else None
    return d


def fmt(v, n=3):
    return "-" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true")
    a = ap.parse_args()
    exps = sorted(d for d in glob.glob(os.path.join(ROOT, "experiments", "*"))
                  if os.path.isdir(d) and os.path.exists(os.path.join(d, "config.json")))
    ds = [summarise(e) for e in exps]

    cols = [("exp", 34), ("arm", 6), ("state", 9), ("steps", 6), ("n_eval", 6),
            ("best_val", 9), ("best_step", 10), ("mean_last3", 11), ("live", 6),
            ("lam", 6), ("n_eff", 6), ("singleton", 10), ("eff_rel", 8),
            ("r_gigpo", 8), ("r_g2po", 8), ("ep/cc", 7), ("len", 7), ("valid", 6)]
    if a.md:
        print("| " + " | ".join(c for c, _ in cols) + " |")
        print("|" + "|".join("---" for _ in cols) + "|")
        for d in ds:
            print("| " + " | ".join(fmt(d.get(c)) for c, _ in cols) + " |")
    else:
        print("".join(c.ljust(w) for c, w in cols))
        print("-" * sum(w for _, w in cols))
        for d in ds:
            print("".join(fmt(d.get(c)).ljust(w) for c, w in cols))
    print("\npublished reference (ALFWorld, Qwen2.5-1.5B-Instruct, In / Out success):")
    for k, (i, o) in REFERENCE.items():
        print(f"  {k:<14} {i:5.2f} / {o:5.2f}")


if __name__ == "__main__":
    main()
