#!/usr/bin/env python3
"""Plot a run's metrics from outputs/metrics.jsonl, refreshing while it trains.

    scripts/plot_metrics.py --exp experiments/<id> [--watch] [--every 120]
    scripts/plot_metrics.py --compare experiments/a experiments/b -o plots/compare.png

The available metrics determine which figures are produced:
  progress.png    the standard curves -- success rate, reward, KL, entropy,
                  response length, grad norm, clip fraction, valid-action ratio
  ccpo.png        the estimator's own terms -- credibility shrinkage, attention
                  mixing, n_eff, bucket occupancy,
                  effect size, correlation against the GiGPO and G2PO references,
                  and the ratio of the two advantage terms
  outlook.png     historical and future advantage magnitudes, weighted terms,
                  correlation, mixture change, and endpoint coverage
"""
import argparse
import json
import math
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
            _lab = kw.get("labels") or {}
            ax.plot(xs, ys, color=C[j % len(C)], lw=1.6,
                    marker="o" if len(xs) < 40 else None, ms=3,
                    label=(_lab.get(k) or k.split("/")[-1]) if len(keys) > 1 else None)
            plotted = True
        if not plotted:
            continue
        ax.set_visible(True)
        drew += 1
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("step", fontsize=8)
        if kw.get("ylabel"):
            ax.set_ylabel(kw["ylabel"], fontsize=8)
        if kw.get("ylim") is not None:
            ax.set_ylim(*kw["ylim"])
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


# ALFWorld task types, in the order the published tables use.
_TYPES = [("pick_and_place", "Pick"), ("look_at_obj_in_light", "Look"),
          ("pick_clean_then_place_in_recep", "Clean"),
          ("pick_heat_then_place_in_recep", "Heat"),
          ("pick_cool_then_place_in_recep", "Cool"),
          ("pick_two_obj_and_place", "Pick2")]
_BY_TYPE = lambda pre: ([f"{pre}/{k}_success_rate" for k, _ in _TYPES],
                        {"labels": {f"{pre}/{k}_success_rate": v for k, v in _TYPES}})
_WS = "webshop_task_score (not success_rate)"

STANDARD = [
    ("held-out success rate", ["val/success_rate", "val-core/success_rate",
                               "val/best_success_rate"], {}),
    ("train success rate", ["episode/success_rate", "train/success_rate"], {}),
    ("episode return", ["episode/reward/mean", "critic/score/mean"], {}),
    ("turns per episode", ["episode/length/mean", "val/length/mean", "episode/length/max"],
     {"labels": {"episode/length/mean": "train", "val/length/mean": "held-out",
                 "episode/length/max": "train max"}}),
    # held-out and training success by ALFWorld task type -- the columns the results
    # table reports, as curves, so a type that stalls is visible during the run and
    # not only at the end.
    ("held-out success by task type", *_BY_TYPE("val")),
    ("train success by task type", *_BY_TYPE("episode")),
    # WebShop reports score alongside success; both tables quote both.
    ("WebShop task score", [f"episode/{_WS}", f"val/{_WS}"],
     {"labels": {f"episode/{_WS}": "train", f"val/{_WS}": "held-out"}}),
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
    # This is the applied node-to-task credibility blend, distinct from the
    # attention-versus-uniform blend below. It is J/(J+kappa) for supported
    # nodes unless overridden; the metric also includes task-backoff rows.
    (r"credibility: mean $\lambda_k$ (applied)", ["ccpo/lam_k_mean"],
     {"ylabel": "0 = task prior; 1 = node baseline", "ylim": (-0.03, 1.03)}),
    (r"attention mix $\lambda$ (applied)", ["ccpo/lam_u_mean", "ccpo/lam_u_gt50"],
     {"labels": {"ccpo/lam_u_mean": "mean lambda", "ccpo/lam_u_gt50": "fraction > 0.5"}}),
    ("EB lambda estimates (diagnostics)", ["ccpo/lam_eb_obs", "ccpo/lam_pooled_obs",
                                           "ccpo/lam_eb_obs_gt0"], {}),
    ("effective neighbourhood n_eff", ["ccpo/n_eff_mean"], {}),
    ("bucket size", ["ccpo/bucket_size_mean", "ccpo/bucket_size_p90"], {}),
    ("singleton bucket fraction", ["ccpo/bucket_singleton_frac"], {}),
    ("live fraction / backoff level 1", ["ccpo/live_frac", "ccpo/lvl1_frac"], {}),
    ("phi is the hidden state (1) or bag-of-words (0)", ["ccpo/phi_is_hidden"], {"hline": 1.0}),
    ("grouping relevance: corr(phi dist, |target diff|)", ["ccpo/phi_rel_corr",
                                                          "ccpo/phi_rel_gt0"], {"hline": 0.0}),
    ("signal variance tau^2", ["ccpo/tau2"], {"hline": 0.0}),
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


OUTLOOK = [
    ("raw component magnitudes", ["ccpo/history_adv_absmean", "ccpo/outlook_adv_absmean"],
     {"labels": {"ccpo/history_adv_absmean": "mean |history|",
                 "ccpo/outlook_adv_absmean": "mean |future|"}}),
    ("component magnitudes after mixing weights",
     ["plot/history_weighted_absmean", "plot/outlook_weighted_absmean"],
     {"labels": {"plot/history_weighted_absmean": "mean |(1-beta) history|",
                 "plot/outlook_weighted_absmean": "mean |beta future|"}}),
    ("future / history magnitude", ["plot/outlook_history_absratio", "plot/outlook_history_weighted_absratio"],
     {"labels": {"plot/outlook_history_absratio": "raw ratio",
                 "plot/outlook_history_weighted_absratio": "after mixing weights"}, "hline": 1.0}),
    ("history / future correlation", ["ccpo/outlook_history_corr"],
     {"ylim": (-1.03, 1.03), "hline": 0.0}),
    ("change from history: mean |mixed - history|", ["ccpo/outlook_delta_absmean"], {}),
    ("outlook endpoint coverage", ["ccpo/outlook_used_frac", "ccpo/outlook_terminal_frac", "ccpo/outlook_fallback_frac"],
     {"labels": {"ccpo/outlook_used_frac": "used (includes terminal)",
                 "ccpo/outlook_terminal_frac": "terminal endpoint",
                 "ccpo/outlook_fallback_frac": "fallback to history"}, "ylim": (-0.03, 1.03)}),
]



FIXED_ANCHOR = [
    ("readout means", ["ccpo/fixed_history_baseline_mean", "ccpo/fixed_future_baseline_mean"], {}),
    ("readout standard deviations", ["ccpo/fixed_history_baseline_std", "ccpo/fixed_future_baseline_std"], {}),
    ("history and joint-future kernel means", ["ccpo/fixed_history_kernel_mean", "ccpo/fixed_future_kernel_mean"], {}),
    ("task-prior means", ["ccpo/fixed_history_task_prior_mean", "ccpo/fixed_future_task_prior_mean"], {}),
    ("applied credibility coefficients", ["ccpo/fixed_history_lambda_k_mean", "ccpo/fixed_future_lambda_k_mean"], {"ylim":(-.03,1.03)}),
    ("distinct peer trajectories", ["ccpo/fixed_history_J_mean", "ccpo/fixed_future_J_mean"], {}),
    ("effective peer trajectories", ["ccpo/fixed_history_n_eff_mean", "ccpo/fixed_future_n_eff_mean"], {}),
    ("signed credit means", ["ccpo/fixed_history_adv_mean", "ccpo/fixed_future_gain_mean", "ccpo/fixed_future_residual_mean"], {"hline":0.}),
    ("credit standard deviations", ["ccpo/fixed_history_adv_std", "ccpo/fixed_future_gain_std", "ccpo/fixed_future_residual_std"], {}),
    ("credit magnitudes before task normalization", ["ccpo/fixed_history_adv_absmean", "ccpo/fixed_future_gain_absmean", "ccpo/fixed_future_residual_absmean", "ccpo/fixed_combined_pre_absmean"], {}),
    ("applied actor component magnitudes", ["ccpo/fixed_history_applied_absmean", "ccpo/fixed_gain_applied_absmean", "ccpo/fixed_combined_applied_absmean"], {}),
    ("gain / history magnitude", ["ccpo/fixed_future_history_absratio"], {}),
    ("gain correlations", ["ccpo/fixed_gain_history_corr", "ccpo/fixed_gain_edge_corr", "ccpo/fixed_baselines_corr"], {"ylim":(-1.03,1.03),"hline":0.}),
    ("correlations excluding terminal windows", ["ccpo/fixed_gain_history_nonterminal_corr", "ccpo/fixed_gain_edge_nonterminal_corr"], {"ylim":(-1.03,1.03),"hline":0.}),
    ("support and terminal windows", ["ccpo/fixed_eligible_frac", "ccpo/fixed_no_peer_frac", "ccpo/fixed_terminal_frac"], {"ylim":(-.03,1.03)}),
    ("future prompt length", ["ccpo/fixed_future_prompt_tokens_mean", "ccpo/fixed_future_prompt_tokens_max"], {}),
    ("future prompt truncation", ["ccpo/fixed_future_prompt_truncated_frac"], {"ylim":(-.03,1.03)}),
    ("applied component coefficients", ["ccpo/fixed_history_weight", "ccpo/fixed_gain_weight", "ccpo/fixed_episode_weight", "ccpo/fixed_edge_weight"], {}),
    ("credit identity errors", ["ccpo/fixed_identity_error", "ccpo/fixed_applied_identity_error"], {"hline":0.}),
    ("reference feature encoding time", ["timing_s/ref", "timing_s/ref_future"], {}),
]

for _title, _keys, _opts in FIXED_ANCHOR:
    _opts.setdefault("labels", {k:k.removeprefix("ccpo/fixed_").replace("_", " ") for k in _keys})
    if any(k.endswith(("_kernel_mean", "_task_prior_mean", "_lambda_k_mean", "_J_mean", "_n_eff_mean")) for k in _keys):
        _opts["labels"] = {k: ("history (includes task fallback)" if "fixed_history_" in k
                               else "future (exact anchors)") for k in _keys}

FUTURE_PROGRESS = [
    ("current / future contextual potentials", ["ccpo/progress_current_value_mean", "ccpo/progress_future_value_mean"], {}),
    ("potential standard deviations", ["ccpo/progress_current_value_std", "ccpo/progress_future_value_std"], {}),
    ("current / future kernel readouts", ["ccpo/progress_current_kernel_mean", "ccpo/progress_future_kernel_mean"], {}),
    ("current / future task priors", ["ccpo/progress_current_task_prior_mean", "ccpo/progress_future_task_prior_mean"], {}),
    ("applied credibility (future excludes terminals)", ["ccpo/progress_current_lambda_k_mean", "ccpo/progress_future_lambda_k_mean"], {"ylim":(-.03,1.03)}),
    ("peer trajectories (future excludes terminals)", ["ccpo/progress_current_J_mean", "ccpo/progress_future_J_mean"], {}),
    ("effective peers (future excludes terminals)", ["ccpo/progress_current_n_eff_mean", "ccpo/progress_future_n_eff_mean"], {}),
    ("raw progress / task normalization scale", ["ccpo/progress_raw_progress_absmean", "ccpo/progress_progress_norm_std_mean"], {}),
    ("history / standardized future / combined", ["ccpo/progress_history_adv_absmean", "ccpo/progress_progress_normalized_absmean", "ccpo/progress_combined_pre_absmean"], {}),
    ("applied H / F / episode / total", ["ccpo/progress_history_applied_absmean", "ccpo/progress_future_applied_absmean", "ccpo/progress_episode_applied_absmean", "ccpo/progress_combined_applied_absmean", "ccpo/progress_actor_applied_absmean"], {}),
    ("correlation with original M5 edge", ["ccpo/progress_future_edge_corr", "ccpo/progress_history_edge_corr", "ccpo/progress_combined_edge_corr"], {"ylim":(-1.03,1.03)}),
    ("future / edge correlation, nonterminal", ["ccpo/progress_future_edge_nonterminal_corr"], {"ylim":(-1.03,1.03)}),
    ("support / terminal coverage", ["ccpo/progress_eligible_frac", "ccpo/progress_terminal_frac", "ccpo/progress_current_exact_frac", "ccpo/progress_current_backoff_frac"], {"ylim":(-.03,1.03)}),
    ("future / history magnitude ratio", ["ccpo/progress_future_history_absratio"], {}),
    ("applied weights", ["ccpo/progress_weight", "ccpo/progress_history_weight", "ccpo/progress_episode_weight", "ccpo/progress_original_edge_weight"], {}),
    ("step / full actor identity errors", ["ccpo/progress_applied_identity_error", "ccpo/progress_actor_identity_error"], {}),
]
for _title, _keys, _opts in FUTURE_PROGRESS:
    _opts.setdefault("labels", {k:k.removeprefix("ccpo/progress_").replace("_", " ") for k in _keys})


def outlook_rows(rows):
    """Derive display-only scales from logged raw magnitudes and each row's beta.

    These components share a pre-normalization scale. Do not overlay them with
    adv_cc_absmean: that metric is computed after the trainer's normalization
    and restoration of padded rows. No derived values are written to metrics.
    """
    result = []
    for row in rows:
        if "ccpo/history_adv_absmean" not in row or "ccpo/outlook_adv_absmean" not in row:
            continue
        item = dict(row)
        history, future, beta = (row.get(k) for k in
                                ("ccpo/history_adv_absmean", "ccpo/outlook_adv_absmean", "ccpo/outlook_beta"))
        if all(isinstance(v, (int, float)) and math.isfinite(v) for v in (history, future, beta)) and 0 <= beta <= 1:
            weighted_history, weighted_future = (1 - beta) * history, beta * future
            item["plot/history_weighted_absmean"] = weighted_history
            item["plot/outlook_weighted_absmean"] = weighted_future
            if history > 1e-12:
                item["plot/outlook_history_absratio"] = future / history
            if weighted_history > 1e-12:
                item["plot/outlook_history_weighted_absratio"] = weighted_future / weighted_history
        result.append(item)
    return result


def ccpo_specs(exp):
    """Label a configured fixed attention mix without changing metric values."""
    try:
        with open(os.path.join(exp, "config.json")) as fh:
            config = json.load(fh)
    except (OSError, ValueError):
        config = {}
    fixed = config.get("env", {}).get("ACG_CCPO_LAM_FIX",
                                   config.get("config", {}).get("ccpo_lam_fix"))
    specs = list(CCPO)
    if fixed is not None and str(fixed) != "":
        label, keys, options = specs[1]
        specs[1] = (rf"attention mix $\lambda$ (fixed at {fixed})", keys, options)
    return specs


def render(exp):
    rows = load(exp)
    if not rows:
        return 0
    out = os.path.join(exp, "plots")
    os.makedirs(out, exist_ok=True)
    name = os.path.basename(exp.rstrip("/"))
    k = 0
    k += panel(rows, STANDARD, os.path.join(out, "progress.png"), f"{name} — training")
    k += panel(rows, ccpo_specs(exp), os.path.join(out, "ccpo.png"), f"{name} — estimator terms")
    k += panel(outlook_rows(rows), OUTLOOK, os.path.join(out, "outlook.png"),
               f"{name} — history / future credit (before task normalization)")
    if any("ccpo/fixed_enabled" in r for r in rows):
        k += panel(rows, FIXED_ANCHOR, os.path.join(out, "fixed_anchor.png"),
                   f"{name} — fixed-anchor history and joint-future credit")
    if any("ccpo/progress_enabled" in r for r in rows):
        k += panel(rows, FUTURE_PROGRESS, os.path.join(out, "future_progress.png"),
                   f"{name} — contextual future-state value progress")
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
