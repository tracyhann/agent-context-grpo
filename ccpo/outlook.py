"""Context-conditioned n-step outlook, mixed with the Monte Carlo advantage.

The future endpoint uses the ordinary history encoder at that endpoint: with a
history length of two, it contains the two transitions following the query turn.
No future information is inserted into the baseline at the query turn.
"""
import numpy as np
import torch

from ccpo import core_ccpo as core


def canonical_trajectory_rows(index, traj_index, turn_index, episode_lengths):
    """Recover complete trajectories and collapse copies introduced by padding.

    Return canonical row indices, a map back to input rows, and trajectory slices.
    Missing turns fail loudly; an adjacent row in a balanced batch is not a successor.
    """
    n = len(index)
    fields = [np.asarray(x).reshape(-1) for x in (index, traj_index, turn_index, episode_lengths)]
    if any(len(x) != n for x in fields):
        raise ValueError("Outlook trajectory metadata must match the batch length")
    tasks, trajs, turns, lengths = fields
    if not np.isfinite(turns.astype(float)).all() or not np.isfinite(lengths.astype(float)).all():
        raise ValueError("Outlook turn indices and lengths must be finite")
    if np.any(turns.astype(float) != turns.astype(int)) or np.any(lengths.astype(float) != lengths.astype(int)):
        raise ValueError("Outlook turn indices and lengths must be integers")
    first, declared, owner = {}, {}, {}
    keys = []
    for i in range(n):
        task, traj, turn, length = str(tasks[i]), str(trajs[i]), int(turns[i]), int(lengths[i])
        if turn < 0 or length <= turn:
            raise ValueError("Outlook turn lies outside its declared episode length")
        if traj in owner and owner[traj] != task:
            raise ValueError("Trajectory IDs must be unique across tasks")
        owner[traj] = task
        group = (task, traj)
        if group in declared and declared[group] != length:
            raise ValueError("Inconsistent episode lengths within a trajectory")
        declared[group] = length
        key = (task, traj, turn)
        first.setdefault(key, i)
        keys.append(key)
    ordered = sorted(first)
    position = {key: i for i, key in enumerate(ordered)}
    groups = []
    offset = 0
    for group in sorted(declared):
        found = [k for k in ordered if k[:2] == group]
        if [k[2] for k in found] != list(range(declared[group])):
            raise ValueError(f"Incomplete trajectory for outlook: {group}")
        groups.append(np.arange(offset, offset + len(found)))
        offset += len(found)
    return (np.array([first[k] for k in ordered], dtype=int),
            np.array([position[k] for k in keys], dtype=int), groups)


def _take(value, rows):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.index_select(0, torch.as_tensor(rows, device=value.device))
    return np.asarray(value)[rows]


def mix_outlook_returns(returns, rewards, history_values, future_values, live,
                        groups, horizon, beta, gamma):
    """Mix MC and n-step targets; preserve the existing local penalty exactly once.

    The current estimator's target is discounted ENV reward minus the CURRENT
    invalid-action penalty, not discounted penalties from the whole suffix.
    Future values therefore predict unpenalised env returns. The current local
    adjustment is added once to either target. A terminal endpoint has value 0.
    """
    if int(horizon) != horizon or horizon < 1 or not 0 <= beta <= 1 or not 0 <= gamma <= 1:
        raise ValueError("Invalid outlook horizon, beta or gamma")
    arrays = [np.asarray(x, dtype=float) for x in (returns, rewards, history_values, future_values)]
    returns, rewards, history_values, future_values = arrays
    live = np.asarray(live, dtype=bool)
    raw_returns = np.zeros_like(returns)
    for ids in groups:
        running = 0.0
        for i in reversed(ids):
            running = rewards[i] + gamma * running
            raw_returns[i] = running
    local_adjustment = returns - raw_returns
    historical = np.where(live, returns - history_values, 0.0)
    outlook = historical.copy()
    used = np.zeros(len(returns), dtype=bool)
    terminal = np.zeros(len(returns), dtype=bool)
    for ids in groups:
        for t, i in enumerate(ids):
            if not live[i]:
                continue
            end = min(t + horizon, len(ids))
            k = end - t
            prefix = sum(gamma ** j * rewards[ids[t + j]] for j in range(k))
            if end == len(ids):
                boot = 0.0
                terminal[i] = True
            else:
                successor = ids[end]
                if not live[successor] or not np.isfinite(future_values[successor]):
                    continue  # no supported endpoint: preserve the MC advantage
                boot = gamma ** k * future_values[successor]
            outlook[i] = prefix + boot + local_adjustment[i] - history_values[i]
            used[i] = True
    mixed = (1.0 - beta) * historical + beta * outlook
    return mixed, historical, outlook, raw_returns, used, terminal


def ccpo_outlook_advantage(*, turn_index, episode_lengths, immediate_rewards,
                           horizon=2, beta=0.25, **kwargs):
    """Same CCPO value readout, with explicit chronological order and no pad votes."""
    gamma = float(kwargs.get('gamma', 0.95))
    if int(horizon) != horizon or horizon < 1 or not 0 <= beta <= 1 or not 0 <= gamma <= 1:
        raise ValueError("Invalid outlook horizon, beta or gamma")
    edge_w = kwargs.get('edge_w')
    edge_w = core._EDGE_W if edge_w is None else float(edge_w)
    target = kwargs.get('target') or core._TARGET
    if edge_w != 0 or target != 'return' or core._STD_MODE == 'local' or core._JW_C != 0:
        raise ValueError("Outlook requires edge_w=0, target=return, no local scaling or J reweighting")
    if core._PHI_MODE != 'hidden+ctx' or kwargs.get('phi_feats') is None:
        raise ValueError("Outlook requires captured hidden+ctx reference features")
    take, restore, groups = canonical_trajectory_rows(
        kwargs['index'], kwargs['traj_index'], turn_index, episode_lengths)
    canonical = dict(kwargs)
    row_fields = ('step_rewards', 'response_mask', 'anchor_obs', 'index', 'traj_index',
                  'phi_feats', 'episode_rewards', 'is_action_valid', 'aff_labels', 'ctx_override')
    for key in row_fields:
        if key in canonical:
            canonical[key] = _take(canonical[key], take)
    rewards = np.asarray(immediate_rewards, dtype=float).reshape(-1)
    if len(rewards) != len(restore) or not np.isfinite(rewards).all():
        raise ValueError("Outlook needs one finite immediate reward per input row")
    # Padding is allowed; contradictory copies are not.
    for key in ('anchor_obs', 'index', 'traj_index', 'episode_rewards', 'is_action_valid'):
        if kwargs.get(key) is not None:
            original = np.asarray(kwargs[key])
            if not np.array_equal(original, np.asarray(canonical[key])[restore]):
                raise ValueError(f"Inconsistent padded outlook rows: {key}")
    if not np.array_equal(rewards, rewards[take][restore]):
        raise ValueError("Inconsistent rewards on padded outlook rows")
    rewards = rewards[take]
    raw_returns = np.zeros(len(take), dtype=float)
    for ids in groups:
        running = 0.0
        for i in reversed(ids):
            running = rewards[i] + gamma * running
            raw_returns[i] = running
    canonical.update(return_diag=True, value_targets=raw_returns)
    hist_tensor, diag = core.ccpo_step_advantage(**canonical)
    scores = canonical['step_rewards']
    if scores.dim() > 1:
        scores = (scores * canonical['response_mask']).sum(-1)
    returns = scores.detach().float().cpu().numpy().astype(float)
    mixed, history, outlook, _, used, terminal = mix_outlook_returns(
        returns, rewards, diag['baseline_values'], diag['bootstrap_values'],
        diag['live_mask'], groups, horizon, beta, gamma)
    live = diag['live_mask']
    if not np.isfinite(mixed).all():
        raise ValueError("Non-finite outlook advantages")
    # The history result must match the existing estimator before outer scaling.
    if not np.allclose(hist_tensor.detach().float().cpu().numpy(), history, atol=1e-5):
        raise ValueError("Outlook history value does not reproduce the CCPO residual")
    def mean_abs(x):
        return float(np.abs(x[live]).mean()) if live.any() else 0.0
    def corr(x, y):
        x, y = np.asarray(x)[live], np.asarray(y)[live]
        return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 1e-12 and y.std() > 1e-12 else float('nan')
    diag.update(outlook_horizon=float(horizon), outlook_beta=float(beta),
                outlook_adv_absmean=mean_abs(outlook), history_adv_absmean=mean_abs(history),
                outlook_delta_absmean=mean_abs(mixed - history),
                outlook_history_corr=corr(history, outlook),
                outlook_used_frac=float(used.mean()), outlook_terminal_frac=float(terminal.mean()),
                outlook_fallback_frac=float((live & ~used).mean()),
                outlook_unique_rows=float(len(take)),
                outlook_padding_frac=float(1 - len(take) / len(restore)))
    scale = mean_abs(mixed)
    diag['effect_rel'] = diag['effect_mean'] / scale if scale > 1e-12 else float('nan')
    diag['acc_len_corr'] = corr(mixed, canonical['response_mask'].sum(-1).detach().cpu().numpy())
    # Correlations refer to the mixed estimator, on the same canonical samples.
    reference = np.zeros(len(take))
    buckets = {}
    for i, (task, obs) in enumerate(zip(canonical['index'], canonical['anchor_obs'])):
        buckets.setdefault((str(task), str(obs)), []).append(i)
    for ids in buckets.values():
        if len(ids) > 1:
            reference[ids] = returns[ids] - returns[ids].mean()
    diag['r_vs_gigpo'] = corr(mixed, reference)
    if canonical.get('episode_rewards') is not None:
        values, nodes, successors = core.g2po_node_values(
            canonical['anchor_obs'], canonical['index'], canonical['traj_index'],
            canonical['episode_rewards'], gamma, kwargs.get('success_reward', 10.0))
        diag['r_vs_g2po'] = corr(mixed, core.g2po_step_advantage(
            canonical['index'], values, nodes, successors, canonical.get('is_action_valid')))
    for key in ('live_mask', 'baseline_values', 'bootstrap_values'):
        diag[key] = diag[key][restore]
    result = torch.as_tensor(mixed[restore], dtype=hist_tensor.dtype, device=hist_tensor.device)
    return (result, diag) if kwargs.get('return_diag', False) else result
