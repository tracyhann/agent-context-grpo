"""Context-conditioned endpoint value increases, with M3/M5 credit scaling.

H_t = Y_t - B_t; P_t = V_context(t+h) - V_context(t).
A_pre = H_t + weight * task_standardize(P_t).
The trainer then applies its existing benchmark-specific combined normalization.
No new model forward is needed: the frozen reference already encodes every turn.
"""
from collections import defaultdict
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from ccpo import core_ccpo as core
from ccpo.outlook import canonical_trajectory_rows, _take


def endpoint_map(groups, n, horizon):
    """A negative endpoint index denotes the trajectory's terminal sentinel."""
    if horizon not in (1, 2):
        raise ValueError('Future progress supports horizon 1 or 2')
    endpoint = np.full(n, -1, dtype=int)
    window = np.zeros(n, dtype=int)
    for ids in groups:
        for t, i in enumerate(ids):
            end = min(t + horizon, len(ids))
            window[i] = end - t
            if end < len(ids):
                endpoint[i] = ids[end]
    return endpoint, window


def progress_from_values(values, groups, episode_rewards, horizon=1, success_reward=10.):
    """M5 terminal convention: success potential 10, other endings 0.

    There is no added reward or outer gamma here. Discounting is already in the
    value labels. A two-step window telescopes over the same value function.
    """
    values = np.asarray(values, dtype=float)
    rewards = np.asarray(episode_rewards, dtype=float)
    endpoint, window = endpoint_map(groups, len(values), horizon)
    terminal = endpoint < 0
    future = np.where(np.abs(rewards - success_reward) < 1e-9, success_reward, 0.)
    future[~terminal] = values[endpoint[~terminal]]
    eligible = np.isfinite(values) & np.isfinite(future)
    raw = np.zeros(len(values))
    raw[eligible] = future[eligible] - values[eligible]
    return raw, future, endpoint, window, eligible


def standardize_progress(raw, tasks, eligible):
    """Sample standard deviation, epsilon 1e-6, as in M3/M5's original edge."""
    out = np.zeros(len(raw)); means = np.zeros(len(raw)); scales = np.zeros(len(raw))
    tasks = np.asarray(tasks).astype(str)
    for task in np.unique(tasks):
        ids = np.flatnonzero((tasks == task) & eligible)
        if len(ids) > 1:
            means[ids] = np.mean(raw[ids]); scales[ids] = np.std(raw[ids], ddof=1)
            out[ids] = (raw[ids] - means[ids]) / (scales[ids] + 1e-6)
    return out, means, scales


def _numpy(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def _stats(diag, name, x, mask):
    values = np.asarray(x)[mask]
    values = values[np.isfinite(values)]
    for suffix, fn in [('mean', np.mean), ('std', np.std),
                       ('absmean', lambda y: np.mean(abs(y))), ('min', np.min), ('max', np.max)]:
        diag['progress_' + name + '_' + suffix] = float(fn(values)) if len(values) else float('nan')


def _corr(x, y, mask):
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 1e-12 and y.std() > 1e-12 else float('nan')


# BF16 has seven explicit fraction bits. This is a numerical tolerance for
# recomputed finite features, not a bound on end-to-end forward-pass error.
_PHI_DUP_RTOL = 2. ** -7
_PHI_DUP_ATOL = 1e-3


def _check_round_trip(kwargs, canonical, restore):
    """Check padding copies without changing the first-occurrence readout.

    Metadata must match exactly. Frozen features can be recomputed in different
    micro-batches, so finite copies may differ slightly. Every input feature must
    still be finite, including copies that canonicalization would discard. There
    is no missing-feature sentinel; accepting NaN/Inf only in later copies would
    make validation depend on batch ordering and hide extraction/numeric failures.
    """
    for key in ('step_rewards', 'response_mask', 'anchor_obs', 'index', 'traj_index',
                'episode_rewards', 'is_action_valid'):
        if kwargs.get(key) is not None and not np.array_equal(_numpy(kwargs[key]), _numpy(canonical[key])[restore]):
            raise ValueError('Inconsistent padded future-progress field: ' + key)
    if kwargs.get('phi_feats') is None:
        return {}
    original, folded = _numpy(kwargs['phi_feats']), _numpy(canonical['phi_feats'])[restore]
    if original.shape != folded.shape:
        raise ValueError('Inconsistent padded future-progress shape: phi_feats '
                         f'{original.shape} vs {folded.shape}')
    feature_axes = tuple(range(1, original.ndim))
    nonfinite = ~np.isfinite(original)
    if nonfinite.any():
        bad_rows = np.flatnonzero(nonfinite.any(axis=feature_axes)) if feature_axes else np.flatnonzero(nonfinite)
        raise ValueError('Non-finite frozen features in future-progress phi_feats '
                         f'({len(bad_rows)} row(s); first {bad_rows[:5].tolist()}); '
                         'all rows, including padding copies, must be finite')
    tolerance = _PHI_DUP_ATOL + _PHI_DUP_RTOL * abs(folded)
    deviation = abs(original - folded)
    beyond = deviation > tolerance
    if beyond.any():
        rows = np.flatnonzero(beyond.any(axis=feature_axes)) if feature_axes else np.flatnonzero(beyond)
        worst = np.unravel_index(np.argmax(np.where(beyond, deviation - tolerance, -np.inf)), deviation.shape)
        raise ValueError('Inconsistent padded future-progress field: phi_feats -- '
                         f'{len(rows)} row(s) differ beyond bf16 tolerance '
                         f'(first {rows[:5].tolist()}; worst element {tuple(int(i) for i in worst)} '
                         f'|diff| {deviation[worst]:.3g} > allowed {tolerance[worst]:.3g})')
    duplicate = np.ones(len(restore), dtype=bool)
    duplicate[np.unique(restore, return_index=True)[1]] = False
    return dict(progress_phi_duplicate_rows=float(duplicate.sum()),
                progress_phi_duplicate_max_diff=float(deviation[duplicate].max()) if duplicate.any() else 0.,
                # Nonfinite rows fail above; successful batches always report zero.
                progress_phi_duplicate_nonfinite=0.)


def ccpo_future_progress_advantage(*, turn_index, episode_lengths, horizon=1,
                                   progress_weight=1., readout='context', **kwargs):
    """A future-state value increase added to the unchanged historical residual.

    readout='m5' is a CPU parity/reference option: self-inclusive uniform node
    potentials. The registered training method always uses readout='context'.
    """
    if horizon not in (1, 2) or not np.isfinite(progress_weight) or progress_weight < 0:
        raise ValueError('Invalid future-progress horizon or weight')
    if readout not in ('context', 'm5'):
        raise ValueError('Unknown future-progress readout')
    edge = core._EDGE_W if kwargs.get('edge_w') is None else float(kwargs['edge_w'])
    if edge != 0 or (kwargs.get('target') or core._TARGET) != 'return':
        raise ValueError('Future progress replaces the edge and requires the M3/M5 return target')
    if core._PHI_MODE != 'hidden+ctx' or kwargs.get('phi_feats') is None:
        raise ValueError('Future progress requires frozen history-plus-context features')
    if core._STD_MODE == 'local' or core._JW_C != 0 or core._LAM_FIX != '1' and core._LAM_FIX != '1.0':
        raise ValueError('Future progress requires the M3/M5 attention readout without local scaling/J weighting')
    if core._GATE != 'hard' or float(kwargs.get('sim') or core._SIM) != 0 or core._SIM_BACKOFF != 0:
        raise ValueError('Future progress requires exact observation groups with task backoff')
    take, restore, groups = canonical_trajectory_rows(
        kwargs['index'], kwargs['traj_index'], turn_index, episode_lengths)
    canonical = dict(kwargs)
    for key in ('step_rewards', 'response_mask', 'anchor_obs', 'index', 'traj_index',
                'phi_feats', 'episode_rewards', 'is_action_valid', 'aff_labels', 'ctx_override'):
        if key in canonical:
            canonical[key] = _take(canonical[key], take)
    phi_diag = _check_round_trip(kwargs, canonical, restore)
    n = len(take); tasks = np.asarray(canonical['index']).astype(str)
    episodes = np.asarray(canonical['episode_rewards'], dtype=float)
    if episodes.shape != (n,) or not np.isfinite(episodes).all():
        raise ValueError('One finite episode return is required per row')
    gamma = float(kwargs.get('gamma', .95)); success_reward = float(kwargs.get('success_reward', 10.))
    if not 0 <= gamma <= 1:
        raise ValueError('Invalid discount')
    labels = np.zeros(n)
    for ids in groups:
        if not np.all(episodes[ids] == episodes[ids[0]]):
            raise ValueError('Episode reward must be constant within a trajectory')
        labels[ids] = gamma ** (len(ids) - np.arange(len(ids))) * episodes[ids]
    # History is the same penalized return residual and feature construction.
    history_tensor, diag = core.ccpo_step_advantage(**dict(canonical, return_diag=True))
    history = _numpy(history_tensor).astype(float)
    # Reuse the exact processed features; only the value labels change. No
    # discounted invalid-action penalty is allowed into these node potentials.
    value_args = dict(canonical, step_rewards=torch.as_tensor(labels, dtype=torch.float64,
                      device=history_tensor.device), prepared_phi=diag['prepared_phi_values'],
                      value_targets=None, target='return', edge_w=0., return_diag=True, dump_enabled=False)
    _, value_diag = core.ccpo_step_advantage(**value_args)
    current = np.asarray(value_diag['baseline_values']).copy()
    m5_values, nodes, successors = core.g2po_node_values(canonical['anchor_obs'], tasks,
        canonical['traj_index'], episodes, gamma, success_reward)
    m5_current = np.asarray([m5_values[node] for node in nodes])
    if readout == 'm5':
        current = m5_current.copy()
    raw, future, endpoint, window, eligible = progress_from_values(
        current, groups, episodes, horizon, success_reward)
    progress, norm_mean, norm_std = standardize_progress(raw, tasks, eligible)
    # Diagnostic reference is the ORIGINAL one-step edge, including for horizon 2.
    m5_raw = core._successor_values(m5_values, successors, n) - m5_current
    m5_edge, _, _ = standardize_progress(m5_raw, tasks, np.ones(n, dtype=bool))
    mixed = history + progress_weight * progress
    if not np.isfinite(mixed).all():
        raise ValueError('Nonfinite future-progress credit')
    hist_live = np.asarray(diag['live_mask']).copy()
    live = hist_live | eligible
    terminal = endpoint < 0
    arrays = dict(uid=tasks, traj_uid=np.asarray(canonical['traj_index']).astype(str),
        turn_index=np.asarray(turn_index, dtype=int)[take],
        episode_lengths=np.asarray(episode_lengths, dtype=int)[take],
        anchor_obs=np.asarray(canonical['anchor_obs']).astype(str), episode_rewards=episodes,
        target=_numpy(canonical['step_rewards']).astype(float), value_target=labels,
        history_baseline=np.asarray(diag['baseline_values']).copy(), history_adv=history,
        current_value=current, future_value=future, raw_progress=raw,
        progress_normalized=progress, weighted_progress=progress_weight * progress,
        progress_norm_mean=norm_mean, progress_norm_std=norm_std, combined_pre=mixed,
        m5_edge_raw=m5_raw, m5_edge_normalized=m5_edge, eligible=eligible,
        history_live=hist_live, live=live, terminal=terminal, endpoint_index=endpoint, window_length=window,
        is_action_valid=np.asarray(canonical['is_action_valid'], dtype=float),
        response_lengths=_numpy(canonical['response_mask'].sum(-1)).astype(int))
    if arrays['target'].ndim > 1:
        arrays['target'] = (arrays['target'] * _numpy(canonical['response_mask'])).sum(-1)
    diag.update(progress_enabled=1., progress_horizon=float(horizon), progress_weight=float(progress_weight),
                progress_history_weight=1., progress_episode_weight=0., progress_original_edge_weight=0.,
                progress_eligible_frac=float(eligible.mean()), progress_terminal_frac=float(terminal.mean()),
                progress_unique_rows=float(n), progress_padding_frac=float(1 - n / len(restore)),
                progress_m5_reference_mode=float(readout == 'm5'), **phi_diag)
    for key in ('target', 'value_target', 'history_baseline', 'history_adv', 'current_value',
                'future_value', 'raw_progress', 'progress_normalized', 'weighted_progress', 'combined_pre',
                'progress_norm_mean', 'progress_norm_std', 'm5_edge_normalized'):
        _stats(diag, key, arrays[key], live)
    # Value diagnostics describe the unpenalized contextual estimator. Terminal
    # destinations have a fixed sentinel and no peer/kernel/shrinkage estimate.
    for source, src in [('history', diag), ('current', value_diag)]:
        for label, key in [('kernel', 'kernel_values'), ('uniform', 'uniform_values'),
                           ('task_prior', 'task_prior_values'), ('J', 'support_values'),
                           ('n_eff', 'effective_support_values'), ('lambda_k', 'credibility_values'),
                           ('level', 'level_values')]:
            x = np.asarray(src[key]).copy(); arrays[source + '_' + label] = x
            _stats(diag, source + '_' + label, x, hist_live if source == 'history' else np.isfinite(current))
            if source == 'current':
                y = np.full(n, np.nan); y[~terminal] = x[endpoint[~terminal]]
                arrays['future_' + label] = y
                _stats(diag, 'future_' + label, y, eligible & ~terminal)
    diag['progress_current_exact_frac'] = float(np.mean(value_diag['level_values'] == 0))
    diag['progress_current_backoff_frac'] = float(np.mean(value_diag['level_values'] > 0))
    for label, x in [('future_edge', progress), ('history_edge', history), ('combined_edge', mixed)]:
        diag['progress_' + label + '_corr'] = _corr(x, m5_edge, eligible)
        diag['progress_' + label + '_nonterminal_corr'] = _corr(x, m5_edge, eligible & ~terminal)
    diag['progress_future_history_corr'] = _corr(progress, history, eligible)
    diag['progress_future_history_absratio'] = float(np.mean(abs(progress[live])) / max(np.mean(abs(history[live])), 1e-12)) if live.any() else 0.
    ctx = core.derive_context(canonical['anchor_obs'], tasks, canonical['traj_index'])
    context = np.asarray([[c[k] for k in ('t', 'n_unique', 'revisit', 'progress')] for c in ctx])
    phi = np.asarray(diag['prepared_phi_values']); hidden = _numpy(canonical['phi_feats'])
    arrays.update(history_hidden=hidden, current_phi=phi, current_context=context)
    for key, x in [('future_hidden', hidden), ('future_phi', phi), ('future_context', context)]:
        y = np.full_like(x, np.nan, dtype=float); y[~terminal] = x[endpoint[~terminal]]; arrays[key] = y
    diag['progress_payload'] = dict(arrays=arrays, take=take, restore=restore)
    diag['progress_components_history'] = history[restore]
    diag['progress_components_future'] = (progress_weight * progress)[restore]
    diag['progress_components_mixed'] = mixed[restore]
    diag['history_adv_absmean'] = float(np.mean(abs(history[hist_live]))) if hist_live.any() else 0.
    diag['r_vs_g2po'] = _corr(mixed, core.g2po_step_advantage(tasks, m5_values, nodes,
                                        successors, canonical.get('is_action_valid')), live)
    uniform_residual = np.zeros(n); buckets = defaultdict(list)
    for i, key in enumerate(zip(tasks, canonical['anchor_obs'])):
        buckets[key].append(i)
    for ids in buckets.values():
        if len(ids) > 1:
            uniform_residual[ids] = arrays['target'][ids] - arrays['target'][ids].mean()
    diag['r_vs_gigpo'] = _corr(mixed, uniform_residual, live)
    diag['acc_len_corr'] = _corr(mixed, arrays['response_lengths'], live)
    diag['effect_rel'] = diag['effect_mean'] / max(np.mean(abs(mixed[live])), 1e-12) if live.any() else float('nan')
    diag['live_mask'] = live[restore]
    for key in ('baseline_values', 'bootstrap_values'):
        diag[key] = diag[key][restore]
    result = torch.as_tensor(mixed[restore], dtype=history_tensor.dtype, device=history_tensor.device)
    return (result, diag) if kwargs.get('return_diag', False) else result


def finalize_progress_logging(diag, step_adv, uid, normalize, step_tag, episode_weight, step_weight=1.):
    """Record components after the trainer's existing combined normalization."""
    if episode_weight != 0 or step_weight != 1:
        raise ValueError('Prepared future-progress arms require episode weight 0 and step weight 1')
    history = diag['progress_components_history'].copy()
    future = diag['progress_components_future'].copy()
    mixed = diag['progress_components_mixed']; live = np.asarray(diag['live_mask'])
    uid = np.asarray(uid).astype(str)
    if normalize:
        for task in np.unique(uid):
            ids = np.flatnonzero((uid == task) & live)
            if len(ids) > 1:
                scale = float(torch.as_tensor(mixed[ids], dtype=step_adv.dtype).std()) + 1e-6
                history[ids] = (history[ids] - history[ids].mean()) / scale
                future[ids] = (future[ids] - future[ids].mean()) / scale
    applied = _numpy(step_adv)
    error = float(np.max(abs(history + future - applied)))
    if error > 3e-5:
        raise ValueError('Applied history and future-progress credits do not sum to actor advantage')
    diag['progress_applied_identity_error'] = error
    payload = diag['progress_payload']; arrays = payload['arrays']; take = payload['take']
    for name, x in [('history_applied', history), ('future_applied', future), ('combined_applied', applied)]:
        arrays[name] = x[take]; _stats(diag, name, x, live)
    directory = os.environ.get('ACG_EXP_DIR')
    if not directory:
        return
    out = Path(directory) / 'outputs/future_progress'; out.mkdir(parents=True, exist_ok=True)
    stem = f'step-{int(step_tag):04d}'
    scalars = {k:float(v) for k,v in diag.items() if k.startswith('progress_') and isinstance(v, (float,int,np.number))}
    target = out / (stem + '.metrics.json'); tmp = target.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(scalars, indent=2) + '\n'); tmp.replace(target)
    columns = [k for k,v in arrays.items() if np.asarray(v).ndim == 1 and k != 'anchor_obs']
    target = out / (stem + '.csv'); tmp = target.with_suffix('.csv.tmp')
    with tmp.open('w') as fh:
        writer = csv.writer(fh); writer.writerow(columns)
        writer.writerows(zip(*(arrays[k] for k in columns)))
    tmp.replace(target)
    every = int(os.environ.get('ACG_CCPO_PROGRESS_SNAPSHOT_EVERY', '1'))
    if every < 1:
        raise ValueError('Future-progress snapshots must remain enabled')
    if int(step_tag) == 1 or int(step_tag) % every == 0:
        target = out / (stem + '.npz'); tmp = target.with_suffix('.npz.tmp')
        with tmp.open('wb') as fh:
            np.savez_compressed(fh, **arrays)
        tmp.replace(target)
