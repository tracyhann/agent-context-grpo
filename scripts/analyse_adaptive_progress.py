#!/usr/bin/env python3
"""Offline conditional trajectory bootstrap of saved M10/M11 future credit.

No model, environment, optimizer, torch, or GPU is used. Features, bandwidths,
labels and normalization statistics are held fixed. Only peer trajectory votes
are resampled. This assesses conditional sensitivity, not policy improvement.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np

KEYS = ('uid traj_uid turn_index episode_lengths anchor_obs episode_rewards '
        'value_target current_phi current_value current_kernel current_J '
        'current_lambda_k current_level future_value endpoint_index raw_progress '
        'progress_normalized progress_norm_mean progress_norm_std history_adv '
        'combined_pre combined_applied history_applied future_applied '
        'terminal eligible live response_lengths').split()


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def snr_gate(progress, variance, epsilon=1e-12):
    signal = np.square(progress)
    return np.maximum(signal - np.maximum(variance, 0), 0) / (signal + epsilon)


def kernel_parts(features, labels, trajectory, k, tau_scale=.15):
    """Match core_ccpo's Gram distances, bandwidth and per-trajectory floors."""
    square = (features * features).sum(1)
    distance = np.sqrt(np.maximum(square[:, None] + square[None, :]
                                 - 2 * (features @ features.T), 0))
    off = distance[np.triu_indices(len(features), 1)]
    median = np.median(off) if len(off) else 0.
    tau = tau_scale * (float(median) if median > 1e-9 else 1.)
    weight = np.exp(-distance / tau)
    weight[trajectory[:, None] == trajectory[None, :]] = 0.
    den = np.zeros((len(features), k)); num = den.copy(); present = den.copy()
    for j in range(k):
        mask = trajectory == j
        den[:, j] = weight[:, mask].sum(1)
        num[:, j] = weight[:, mask] @ labels[mask]
        present[:, j] = mask.any() & (trajectory != j)
    # The training implementation floors each trajectory's denominator before
    # recombining. Preserve even the tiny-kernel numerical behavior exactly.
    num = den * (num / np.maximum(den, 1e-12))
    return num, den, present


class TaskReadout:
    def __init__(self, features, labels, observation, trajectory, kappa=2., tau=.15):
        self.trajectory_names, self.trajectory = np.unique(trajectory, return_inverse=True)
        self.k = len(self.trajectory_names); self.kappa = kappa
        self.n = len(labels)
        self.task_num = np.bincount(self.trajectory, weights=labels, minlength=self.k)
        self.task_den = np.bincount(self.trajectory, minlength=self.k)
        self.fnum, self.fden, _ = kernel_parts(features, labels, self.trajectory, self.k, tau)
        self.num = np.zeros((self.n, self.k)); self.den = self.num.copy(); self.present = self.num.copy()
        groups = defaultdict(list)
        for i, obs in enumerate(observation):
            groups[str(obs)].append(i)
        for group in groups.values():
            ix = np.asarray(group)
            self.num[ix], self.den[ix], self.present[ix] = kernel_parts(
                features[ix], labels[ix], self.trajectory[ix], self.k, tau)

    def read(self, query_rows, counts):
        """counts has shape (other trajectories, replicates); queries share own id.

        Bootstrap copies are distinct draws from the empirical peer distribution.
        Their counts affect the kernel, task prior and observation support J.
        Original padding duplicates were already removed in the saved snapshot.
        """
        own = self.trajectory[query_rows[0]]
        assert np.all(self.trajectory[query_rows] == own)
        other = np.flatnonzero(np.arange(self.k) != own)
        if not len(other):
            raise ValueError('No peer trajectories')
        counts = np.asarray(counts, dtype=float)
        assert counts.shape[0] == len(other)
        ix = np.ix_(query_rows, other)
        support = self.present[ix] @ counts
        kernel = (self.num[ix] @ counts) / np.maximum(self.den[ix] @ counts, 1e-12)
        fallback = (self.fnum[ix] @ counts) / np.maximum(self.fden[ix] @ counts, 1e-12)
        prior = (self.task_num[other] @ counts) / (self.task_den[other] @ counts)
        lam = support / (support + self.kappa)
        value = np.where(support > 0, lam * kernel + (1 - lam) * prior, fallback)
        return value, np.where(support > 0, kernel, fallback), support, np.where(support > 0, lam, 1.)


def paired_progress(values, endpoint, terminal_reward):
    future = np.broadcast_to(np.asarray(terminal_reward, dtype=float)[:, None], values.shape).copy()
    nt = endpoint >= 0
    future[nt] = values[endpoint[nt]]
    return future - values


def mean(x, mask=None):
    x = np.asarray(x); x = x if mask is None else x[mask]
    return float(x.mean()) if len(x) else None


def corr(x, y, mask):
    x, y = x[mask], y[mask]
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 1e-12 and y.std() > 1e-12 else None


def audit_snapshot(source, tag, bootstrap=256, seed=190919):
    path = Path(source['path']); before = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (source['size'], source['mtime_ns'])
    with np.load(path, allow_pickle=False) as z:
        a = {k: z[k] for k in KEYS}
    n = len(a['uid']); point = np.zeros(n); kernels = point.copy(); supports = point.copy(); lambdas = point.copy()
    variance = point.copy(); variance_test = point.copy(); sign_agreement = point.copy()
    task_indices = []; successes_per_task = {}; task_gate = np.zeros(n)
    fallback_probability = np.zeros(n)
    for task in np.unique(a['uid']):
        ids = np.flatnonzero(a['uid'] == task); task_indices.append(ids)
        # core_ccpo casts the labels to float32 before its float64 aggregation.
        labels = a['value_target'][ids].astype(np.float32).astype(float)
        readout = TaskReadout(a['current_phi'][ids], labels, a['anchor_obs'][ids], a['traj_uid'][ids])
        successes_per_task[str(task)] = int(np.sum((a['turn_index'][ids] == 0) & (a['episode_rewards'][ids] == 10)))
        for own in range(readout.k):
            local = np.flatnonzero(readout.trajectory == own); absolute = ids[local]
            original = readout.read(local, np.ones((readout.k - 1, 1)))
            point[absolute] = original[0][:, 0]; kernels[absolute] = original[1][:, 0]
            j = original[2][:, 0]
            supports[absolute] = np.where(j > 0, j, readout.k - 1)
            lambdas[absolute] = original[3][:, 0]
            key = f'{seed}:{tag}:{source["step"]}:{task}:{readout.trajectory_names[own]}'
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little'))
            # Separate halves estimate gate variance and independently assess it.
            counts = rng.multinomial(readout.k - 1, np.full(readout.k - 1, 1 / (readout.k - 1)), size=2 * bootstrap).T
            values, _, boot_support, _ = readout.read(local, counts)
            fallback_probability[absolute] = np.mean(boot_support == 0, axis=1)
            lookup = {int(v): q for q, v in enumerate(absolute)}
            endpoints = np.array([lookup[int(e)] if e >= 0 else -1 for e in a['endpoint_index'][absolute]])
            boot_progress = paired_progress(values, endpoints, a['episode_rewards'][absolute])
            variance[absolute] = boot_progress[:, :bootstrap].var(axis=1, ddof=1)
            testing = boot_progress[:, bootstrap:]
            variance_test[absolute] = testing.var(axis=1, ddof=1)
            sign = np.where(abs(a['raw_progress'][absolute]) > 1e-8, np.sign(a['raw_progress'][absolute]), 0)
            boot_sign = np.where(abs(testing) > 1e-8, np.sign(testing), 0)
            sign_agreement[absolute] = np.mean(boot_sign == sign[:, None], axis=1)
    parity = {}
    for key, actual in [('current_value', point), ('current_kernel', kernels), ('current_J', supports), ('current_lambda_k', lambdas)]:
        parity[key] = float(np.max(abs(actual - a[key])))
        np.testing.assert_allclose(actual, a[key], atol=2e-7, rtol=1e-8, err_msg=f'{tag} {source["step"]} {key}')
    np.testing.assert_allclose(a['history_adv'] + a['progress_normalized'], a['combined_pre'], atol=1e-6)
    np.testing.assert_allclose(a['history_applied'] + a['future_applied'], a['combined_applied'], atol=3e-5)
    nt = ~a['terminal']; success = a['episode_rewards'] == 10
    lone = success & np.array([successes_per_task[str(t)] == 1 for t in a['uid']])
    P = a['raw_progress']; F = a['progress_normalized']; H = a['history_adv']; w = snr_gate(P, variance)
    centered = w * F; matched = np.zeros(n)
    for ids in task_indices:
        centered[ids] -= centered[ids].mean()
        matched[ids] = abs(w[ids] * F[ids]).sum() / max(abs(F[ids]).sum(), 1e-12)
        use = ids[nt[ids]]
        if len(use):
            signal = np.mean(P[use] ** 2); noise = np.mean(variance[use])
            task_gate[ids] = max(signal - noise, 0) / (signal + 1e-12)
    variants = {'fixed_0': (np.zeros(n), np.zeros(n)), 'fixed_025': (.25 * F, np.full(n, .25)),
                'fixed_1': (F, np.ones(n)), 'row_snr': (w * F, w),
                'row_snr_centered': (centered, None), 'matched_task_scale': (matched * F, matched),
                'task_nonterminal_snr': (task_gate * F, task_gate)}
    # This noise proxy freezes the original task centering/scales. It is not
    # policy-gradient variance, nor uncertainty over features or reward labels.
    noise_F = variance_test / np.square(a['progress_norm_std'] + 1e-6)
    summaries = {}
    for label, (future, weight) in variants.items():
        combined = H + future
        d = dict(future_absmean=mean(abs(future)), future_mass_retention=float(abs(future).sum() / max(abs(F).sum(), 1e-12)),
                 terminal_mass_share=float(abs(future[a['terminal']]).sum() / max(abs(future).sum(), 1e-12)),
                 nonterminal_mass_retention=float(abs(future[nt]).sum() / max(abs(F[nt]).sum(), 1e-12)),
                 future_mean=mean(future), task_mean_abs=float(np.mean([abs(future[ix].mean()) for ix in task_indices])),
                 history_sign_flip_fraction=mean((H * combined) < 0),
                 original_combined_sign_flip_fraction=mean(((H + F) * combined) < 0),
                 success_nonterminal_future_negative_fraction=mean(future < -1e-8, success & nt),
                 failure_nonterminal_future_negative_fraction=mean(future < -1e-8, ~success & nt),
                 success_nonterminal_combined_negative_fraction=mean(combined < -1e-8, success & nt),
                 lone_nonterminal_rows=int(np.sum(lone & nt)),
                 lone_nonterminal_future_negative_rows=int(np.sum(lone & nt & (future < -1e-8))),
                 lone_nonterminal_combined_negative_rows=int(np.sum(lone & nt & (combined < -1e-8))),
                 future_history_corr=corr(future, H, a['live']))
        if weight is not None:
            d.update(weight_mean=mean(weight), weight_nonterminal_mean=mean(weight, nt),
                     weight_terminal_mean=mean(weight, ~nt), weight_zero_fraction=mean(weight <= 1e-12),
                     weight_above_08_fraction=mean(weight >= .8),
                     frozen_norm_noise_fraction=float(np.sum(weight ** 2 * noise_F) / max(np.sum(noise_F), 1e-12)))
        summaries[label] = d
    active = nt & (abs(P) > 1e-8)
    reliability = {}
    for name, mask in [('low_weight', active & (w < .2)), ('high_weight', active & (w > .8)), ('all_active_nt', active)]:
        reliability[name] = dict(rows=int(mask.sum()), test_sign_agreement=mean(sign_agreement, mask),
                                test_progress_variance=mean(variance_test, mask))
    result = dict(tag=tag, step=source['step'], rows=n, trajectories=int(np.sum(a['turn_index'] == 0)),
                  tasks=len(task_indices), nonterminal_rows=int(nt.sum()), terminal_rows=int((~nt).sum()),
                  all_failed_tasks=sum(v == 0 for v in successes_per_task.values()),
                  one_success_tasks=sum(v == 1 for v in successes_per_task.values()),
                  raw_zero_rows=int(np.sum(abs(P) <= 1e-8)),
                  raw_zero_normalized_nonzero_rows=int(np.sum((abs(P) <= 1e-8) & (abs(F) > 1e-8))),
                  parity=parity, reliability=reliability, variants=summaries,
                  bootstrap_fallback_probability_mean=mean(fallback_probability))
    arrays = dict(raw_progress=P, future=F, history=H, variance_gate=variance,
                  variance_test=variance_test, gate=w, task_gate=task_gate,
                  centered_future=centered, matched_weight=matched,
                  terminal=a['terminal'], success=success, lone_success=lone,
                  uid=a['uid'], traj_uid=a['traj_uid'], sign_agreement=sign_agreement)
    return result, arrays


def aggregate(results):
    summary = {}
    for tag in sorted({r['tag'] for r in results}):
        available = [r for r in results if r['tag'] == tag]
        last = max(r['step'] for r in available)
        windows = [(1, last), (1, 30), (31, 75), (100, last), (100, min(last, 120))]
        periods = {}
        for lo, hi in windows:
            rows = [r for r in available if lo <= r['step'] <= hi]
            if not rows:
                continue
            stats = dict(batches=len(rows), rows=sum(r['rows'] for r in rows),
                         trajectories=sum(r['trajectories'] for r in rows),
                         all_failed_tasks=sum(r['all_failed_tasks'] for r in rows),
                         one_success_tasks=sum(r['one_success_tasks'] for r in rows),
                         max_parity_error=max(max(r['parity'].values()) for r in rows), variants={})
            for variant in rows[0]['variants']:
                merged = {}
                for key in rows[0]['variants'][variant]:
                    values = [r['variants'][variant][key] for r in rows if r['variants'][variant].get(key) is not None]
                    merged[key] = sum(values) if key.endswith('_rows') else float(np.mean(values)) if values else None
                stats['variants'][variant] = merged
            stats['reliability'] = {}
            for group in rows[0]['reliability']:
                count = sum(r['reliability'][group]['rows'] for r in rows)
                stats['reliability'][group] = {'rows':count}
                for key in ['test_sign_agreement', 'test_progress_variance']:
                    stats['reliability'][group][key] = sum(r['reliability'][group]['rows'] * (r['reliability'][group][key] or 0) for r in rows) / count if count else None
            periods[f'{lo}-{hi}'] = stats
        summary[tag] = periods
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--only', default='')
    p.add_argument('--steps', default='')
    p.add_argument('--bootstrap', type=int)
    p.add_argument('--no-resume', action='store_true')
    args = p.parse_args(); plan = json.loads(args.plan.read_text())
    B = args.bootstrap or plan['bootstrap_replicates_per_split']; seed = plan['seed']
    selected = {int(x) for x in args.steps.split(',') if x}; results = []
    args.output.mkdir(parents=True, exist_ok=True)
    os.nice(10)
    for run in plan['runs']:
        if args.only and run['tag'] != args.only:
            continue
        cfg = json.loads(Path(run['config_path']).read_text())['config']
        assert float(cfg['ccpo_tau']) == .15 and float(cfg['ccpo_prior_kappa']) == 2
        assert cfg['ccpo_gate'] == 'hard' and cfg['ccpo_wmode'] == 'soft'
        assert float(cfg['ccpo_lam_fix']) == 1 and cfg['ccpo_lk_fix'] == ''
        assert cfg['ccpo_progress_horizon'] == 1 and cfg['ccpo_progress_weight'] == 1
        for source in run['snapshots']:
            step = source['step']
            if selected and step not in selected:
                continue
            dest = args.output / 'per_step' / f'{run["tag"]}-{step:04d}.json'
            if dest.exists() and not args.no_resume:
                result = json.loads(dest.read_text())
                assert result['bootstrap_per_split'] == B and result['seed'] == seed
            else:
                start = time.monotonic()
                result, arrays = audit_snapshot(source, run['tag'], B, seed)
                result.update(bootstrap_per_split=B, seed=seed, source=source,
                              elapsed_seconds=time.monotonic() - start)
                write_json(dest, result)
                detail = args.output / 'row_diagnostics' / f'{run["tag"]}-{step:04d}.npz'
                detail.parent.mkdir(exist_ok=True)
                np.savez_compressed(detail, **arrays)
            results.append(result)
            status = dict(status='running', run=run['tag'], step=step,
                          completed_snapshots=len(results), seconds=result.get('elapsed_seconds'),
                          parity_max=max(result['parity'].values()),
                          row_gate_mean=result['variants']['row_snr']['weight_mean'],
                          terminal_share_before=result['variants']['fixed_1']['terminal_mass_share'],
                          terminal_share_after=result['variants']['row_snr']['terminal_mass_share'])
            write_json(args.output / 'status.json', status)
            if step % 5 == 0 or step == 1 or selected:
                print(json.dumps(clean(status)), flush=True)
    write_json(args.output / 'summary.json', dict(
        interpretation='Conditional credit-estimator sensitivity, not success rates, policy-gradient alignment, or causal training benefit.',
        averaging='Equal-weight means of per-batch metrics; *_rows are totals. Reliability groups are pooled by row count.',
        bootstrap_per_split=B, seed=seed, results=aggregate(results)))
    write_json(args.output / 'status.json', dict(status='complete', completed_snapshots=len(results)))


if __name__ == '__main__':
    main()
