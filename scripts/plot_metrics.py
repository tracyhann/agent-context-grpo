#!/usr/bin/env python3
"""Plot a run's metrics from outputs/metrics.jsonl, refreshing while it trains.

    scripts/plot_metrics.py --exp experiments/<id> [--watch] [--every 120]
    scripts/plot_metrics.py --compare experiments/a experiments/b -o plots/compare.png

Two panels of figures are produced:
  progress.png    the standard curves -- success rate, reward, KL, entropy,
                  response length, grad norm, clip fraction, valid-action ratio
  ccpo.png        the estimator's own terms -- lambda, n_eff, bucket occupancy,
                  effect size, correlation against the GiGPO and G2PO references,
                  and the ratio of the two advantage terms
The second panel is the one that says whether the method is doing anything.
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

C = ["#2f6f8f", "#c1663a", "#4f8a5b", "#8a5a9c", "#b0873a", "#7a7a74"]


def load(exp):
    path = os.path.join(exp, "outputs", "metrics.jsonl")
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass          # a torn final line while the run is writing
    return rows


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is not None and isinstance(v, (int, float)):
            xs.append(r["step"])
            ys.append(v)
    return xs, ys


def first_key(rows, *cands):
    """verl's metric names drift between versions; take the first that exists."""
    present = set().union(*[set(r) for r in rows]) if rows else set()
    for c in cands:
        if c in present:
            return c
    for c in cands:                      # fall back to a suffix match
        for k in sorted(present):
            if k.endswith(c):
                return k
    return None


def panel(rows, specs, out, title):
    # Lay out only the panels that have data, so a series that has not been
    # written yet (a held-out score before the first evaluation) does not leave a
    # hole in the grid.
    live = []
    for label, cands, kw in specs:
        keys = cands if isinstance(cands, (list, tuple)) else [cands]
        if any(first_key(rows, c) and series(rows, first_key(rows, c))[0] for c in keys):
            live.append((label, keys, kw))
    if not live:
        return False
    ncol = 3
    nrow = (len(live) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.1 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    drew = 0
    for i, (label, keys, kw) in enumerate(live):
        ax = axes[i // ncol][i % ncol]
        plotted = False
        for j, cand in enumerate(keys):
            k = first_key(rows, cand)
            if not k:
                continue
            xs, ys = series(rows, k)
            if not xs:
                continue
            ax.plot(xs, ys, color=C[j % len(C)], lw=1.6,
                    marker="o" if len(xs) < 40 else None, ms=3,
                    label=k.split("/")[-1] if len(keys) > 1 else None)
            plotted = True
        if not plotted:
            continue
        ax.set_visible(True)
        drew += 1
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)
        if kw.get("hline") is not None:
            ax.axhline(kw["hline"], color="#999", ls="--", lw=0.9)
        if len(keys) > 1 and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, frameon=False)
    if not drew:
        plt.close(fig)
        return False
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return True


STANDARD = [
    ("held-out success rate", ["val/success_rate", "val-core/success_rate",
                               "val/best_success_rate"], {}),
    ("train success rate", ["episode/success_rate", "train/success_rate"], {}),
    ("episode return", ["episode/reward/mean", "critic/score/mean"], {}),
    ("episode length (turns)", ["episode/length/mean", "episode/length/max"], {}),
    ("response truncation rate", ["response_length/clip_ratio"], {"hline": 0.0}),
    ("throughput (tok/s)", ["perf/throughput"], {}),
    ("KL loss", ["actor/kl_loss", "actor/kl"], {}),
    ("policy entropy", ["actor/entropy", "actor/entropy_loss"], {}),
    ("response length (tokens)", ["response_length/mean"], {}),
    ("grad norm", ["actor/grad_norm"], {}),
    ("PPO clip fraction", ["actor/pg_clipfrac"], {}),
    ("valid action ratio", ["episode/valid_action_ratio"], {"hline": 1.0}),
    ("policy loss", ["actor/pg_loss"], {}),
    ("learning rate", ["actor/lr"], {}),
    ("step time (s)", ["timing_s/step"], {}),
    ("time breakdown (s)", ["timing_s/gen", "timing_s/old_log_prob", "timing_s/ref",
                            "timing_s/adv", "timing_s/update_actor", "timing_s/testing"], {}),
    ("KL (reward-side)", ["actor/reward_kl_penalty", "critic/kl"], {}),
    ("advantage magnitude", ["critic/advantages/mean", "critic/advantages/max"], {}),
]

CCPO = [
    ("lambda* applied", ["ccpo/lam_u_mean", "ccpo/lam_u_gt50"], {}),
    ("lambda* by rule (only one applies)", ["ccpo/lam_eb_obs", "ccpo/lam_pooled_obs",
                                           "ccpo/lam_eb_obs_gt0"], {}),
    ("effective neighbourhood n_eff", ["ccpo/n_eff_mean"], {}),
    ("bucket size", ["ccpo/bucket_size_mean", "ccpo/bucket_size_p90"], {}),
    ("singleton bucket fraction", ["ccpo/bucket_singleton_frac"], {}),
    ("live fraction / backoff level 1", ["ccpo/live_frac", "ccpo/lvl1_frac"], {}),
    ("mean affinity weight E[w]", ["ccpo/E_w"], {}),
    ("effect size vs uniform baseline", ["ccpo/effect_mean", "ccpo/effect_p90"], {}),
    ("effect relative to |A|", ["ccpo/effect_rel"], {}),
    ("corr with reference estimators", ["ccpo/r_vs_gigpo", "ccpo/r_vs_g2po"], {"hline": 1.0}),
    ("advantage term magnitudes", ["ccpo/adv_ep_absmean", "ccpo/adv_cc_absmean"], {}),
    ("|A_EP| / |A_CC|", ["ccpo/adv_ep_over_cc"], {"hline": 1.0}),
    ("corr(A_CC, response length)", ["ccpo/acc_len_corr"], {"hline": 0.0}),
    ("n buckets", ["ccpo/n_buckets"], {}),
    ("evals since best", ["train/early_stop_stale_evals"], {}),
]


def render(exp):
    rows = load(exp)
    if not rows:
        return 0
    out = os.path.join(exp, "plots")
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(exp.rstrip("/"))
    k = 0
    k += panel(rows, STANDARD, os.path.join(out, "progress.png"), f"{name} — training")
    k += panel(rows, CCPO, os.path.join(out, "ccpo.png"), f"{name} — estimator terms")
    return k


def compare(exps, out):
    keys = [("held-out success rate", ["val/success_rate", "val-core/success_rate"]),
            ("best so far", ["val/best_success_rate"]),
            ("episode return", ["critic/score/mean"]),
            ("response length", ["response_length/mean"]),
            ("KL loss", ["actor/kl_loss"]),
            ("valid action ratio", ["episode/valid_action_ratio"])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for i, (label, cands) in enumerate(keys):
        ax = axes[i // 3][i % 3]
        any_ = False
        for j, e in enumerate(exps):
            rows = load(e)
            k = first_key(rows, *cands) if rows else None
            if not k:
                continue
            xs, ys = series(rows, k)
            if xs:
                ax.plot(xs, ys, color=C[j % len(C)], lw=1.7,
                        label=os.path.basename(e.rstrip("/")))
                any_ = True
        if any_:
            ax.set_visible(True)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("step", fontsize=8)
            ax.grid(alpha=0.25, lw=0.5)
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("arm comparison", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("-o", "--out", default="experiments/compare.png")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--every", type=int, default=120)
    a = ap.parse_args()
    if a.compare:
        compare(a.compare, a.out)
        return
    if not a.exp:
        ap.error("--exp or --compare required")
    while True:
        try:
            n = render(a.exp)
            print(f"{time.strftime('%H:%M:%S')} rendered {n} panel(s)", flush=True)
        except Exception as e:                                # noqa: BLE001
            print(f"plot error: {e}", flush=True)
        if not a.watch:
            return
        time.sleep(a.every)


if __name__ == "__main__":
    main()
