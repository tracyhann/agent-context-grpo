#!/usr/bin/env python3
"""Offline comparison of the two credit channels and of simpler future signals.

Reads the per-turn logs every future-progress run already writes,
`<run>/outputs/future_progress/step-NNNN.csv`, and starts no environment, model
forward, Ray runtime or optimizer. Two questions, per training step and averaged
over step ranges:

A  Historical credit H versus future credit F. Correlation, how often the signs
   disagree, how much of each channel's within-task variance is explained by which
   trajectory a turn belongs to and by whether that trajectory succeeded, and how
   often a turn of a failed trajectory still receives positive credit.

B  The run's future credit versus simpler signals recomputed from the same logged
   contextual values on the same rollouts: the one-step value difference, the
   one-step and K-step discount-aligned (TD) residuals, the discounted final
   reward, and the episode outcome.

Both are statistics of credit signals on the rollouts of the logged policy. They
say how different two signals are, not which one trains a better policy.

    .venv/bin/python scripts/analyse_credit_channels.py experiments/<run>

Before any statistic is computed, each step's logged future credit is recomputed
from the logged values with the training code; a mismatch stops the analysis.
"""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ccpo.future_progress import PROGRESS_MODES, progress_from_values, standardize_progress

TEXT = ('uid', 'traj_uid')
NUMERIC = ('turn_index', 'episode_lengths', 'episode_rewards', 'history_adv', 'current_value',
           'raw_progress', 'progress_normalized', 'window_length', 'history_applied', 'future_applied')
BOOLEAN = ('eligible', 'history_live', 'terminal')
DEFAULT_PHASES = '1-30,31-120,121-150'
EPS = 1e-12
# (label, H column, F column). `applied` is what enters the policy gradient, after
# the trainer's task-wise normalization; `raw` keeps each channel's own zero point:
# H = 0 is "as good as the peers", F = 0 is "the contextual value did not change".
PAIRS = (('applied', 'history_applied', 'future_applied'), ('raw', 'history_adv', 'raw_progress'))
POPULATIONS = ('all', 'nonterminal')


# ---------------------------------------------------------------------------
# logs
# ---------------------------------------------------------------------------
def read_step(path):
    """One step's per-turn log as arrays. Rows are unique turns of complete trajectories."""
    with Path(path).open(newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = list(reader)
    missing = [c for c in TEXT + NUMERIC + BOOLEAN if c not in header]
    if missing:
        raise ValueError(f'{path}: missing columns {missing}')
    if not rows:
        raise ValueError(f'{path}: no rows')
    at = {name: i for i, name in enumerate(header)}
    log = {k: np.array([r[at[k]] for r in rows]) for k in TEXT}
    log.update({k: np.array([r[at[k]] for r in rows], dtype=float) for k in NUMERIC})
    for k in BOOLEAN:
        tokens = np.array([r[at[k]] for r in rows])
        if not np.isin(tokens, ['True', 'False']).all():
            raise ValueError(f'{path}: column {k} is not boolean')
        log[k] = tokens == 'True'
    return log


def trajectory_groups(log):
    """Row indices of each trajectory in turn order. Incomplete trajectories fail."""
    rows = defaultdict(list)
    for i, key in enumerate(zip(log['uid'], log['traj_uid'])):
        rows[key].append(i)
    groups = []
    for key in sorted(rows):
        ids = np.array(rows[key])
        ids = ids[np.argsort(log['turn_index'][ids], kind='stable')]
        if (not np.array_equal(log['turn_index'][ids], np.arange(len(ids)))
                or np.any(log['episode_lengths'][ids] != len(ids))):
            raise ValueError(f'Incomplete trajectory in log: {key}')
        groups.append(ids)
    return groups


def verify_reconstruction(log, groups, gamma, horizon, mode, success_reward, source='log'):
    """The logged future credit must follow from the logged values under (gamma, horizon, mode)."""
    raw, _, _, window, eligible = progress_from_values(
        log['current_value'], groups, log['episode_rewards'], horizon, success_reward, gamma=gamma, mode=mode)
    problem = None
    if not np.array_equal(eligible, log['eligible']):
        problem = 'eligible rows differ'
    elif not np.array_equal(window, log['window_length'].astype(int)):
        problem = 'window lengths differ'
    elif not np.allclose(raw, log['raw_progress'], rtol=1e-5, atol=1e-5):
        problem = f'raw future credit differs by up to {np.max(abs(raw - log["raw_progress"])):.3g}'
    if problem:
        raise ValueError(f'{source}: {problem} when recomputed with gamma={gamma}, horizon={horizon}, '
                         f'mode={mode}. Pass the run\'s own --gamma/--horizon/--mode.')


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or x.std() <= EPS or y.std() <= EPS:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def ranks(x):
    """Ranks from 1, ties sharing their average rank."""
    _, inverse, counts = np.unique(np.asarray(x, dtype=float), return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    return ((last - counts + 1 + last) / 2.)[inverse]


def spearman(x, y):
    return pearson(ranks(x), ranks(y)) if len(x) else float('nan')


def sign_disagreement(x, y):
    """Share of rows with opposite signs, among rows where neither is zero."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    both = (abs(x) > EPS) & (abs(y) > EPS)
    return float(np.mean(np.sign(x[both]) != np.sign(y[both]))) if both.any() else float('nan')


def variance_share(x, tasks, labels, mask):
    """Mean over tasks of SS_between / SS_total, for groups given by `labels` inside each task.

    Returns (share, chance, tasks used). `chance` is the share expected when the
    labels carry no information, (groups - 1) / (rows - 1), averaged over the same
    tasks. A task needs at least two groups, more rows than groups, and spread.
    """
    shares, chances = [], []
    for task in np.unique(tasks[mask]):
        ids = np.flatnonzero(mask & (tasks == task))
        values = x[ids]
        _, inverse = np.unique(labels[ids], return_inverse=True)
        k = int(inverse.max()) + 1
        if k < 2 or len(ids) <= k or np.ptp(values) <= EPS:
            continue
        counts = np.bincount(inverse)
        means = np.bincount(inverse, weights=values) / counts
        shares.append(float((counts * (means - values.mean()) ** 2).sum() / ((values - values.mean()) ** 2).sum()))
        chances.append((k - 1) / (len(ids) - 1))
    if not shares:
        return float('nan'), float('nan'), 0
    return float(np.mean(shares)), float(np.mean(chances)), len(shares)


def outcome_signs(x, tasks, success, mask):
    """Among tasks with both outcomes: share of failed-trajectory rows with positive
    credit, and share of successful-trajectory rows with negative credit."""
    mixed = np.zeros(len(x), dtype=bool)
    for task in np.unique(tasks[mask]):
        ids = np.flatnonzero(mask & (tasks == task))
        if success[ids].any() and not success[ids].all():
            mixed[ids] = True
    failed, won = mixed & ~success, mixed & success
    return (float(np.mean(x[failed] > EPS)) if failed.any() else float('nan'),
            float(np.mean(x[won] < -EPS)) if won.any() else float('nan'))


def relation_statistics(h, f, tasks, success, mask):
    """How the two channels relate turn by turn. Depends on each channel's zero point."""
    out = dict(pearson=pearson(h[mask], f[mask]), spearman=spearman(h[mask], f[mask]),
               sign_disagreement=sign_disagreement(h[mask], f[mask]), rows=float(mask.sum()))
    for name, x in (('h', h), ('f', f)):
        out['positive_in_failed_' + name], out['negative_in_successful_' + name] = outcome_signs(
            x, tasks, success, mask)
    return out


def share_statistics(h, f, tasks, trajectories, success, mask):
    """What explains each channel's within-task variance: the trajectory a turn belongs
    to, and that trajectory's outcome. Unchanged by task-wise centering and scaling, so
    the raw and the applied credit give the same shares."""
    out = {}
    for label, groups in (('trajectory', trajectories), ('outcome', success)):
        for name, x in (('h', h), ('f', f)):
            share, chance, _ = variance_share(x, tasks, groups, mask)
            out[f'{label}_{name}'] = share
            out[f'{label}_chance_{name}'] = chance
    return out


def alternative_signals(log, groups, gamma, horizon, success_reward):
    """Raw future signals from the logged contextual values: name -> (raw, eligible)."""
    def signal(h, mode):
        raw, _, _, _, eligible = progress_from_values(
            log['current_value'], groups, log['episode_rewards'], h, success_reward, gamma=gamma, mode=mode)
        return raw, eligible
    k_step = signal(horizon, 'discount_aligned')
    return {
        'one_step_difference': signal(1, 'difference'),            # V(t+1) - V(t)
        'one_step_td_residual': signal(1, 'discount_aligned'),     # gamma V(t+1) - V(t)
        'k_step_td_residual': k_step,                              # gamma^w V(t+w) - V(t), w <= K
        'discounted_final_reward': signal(horizon, 'terminal_proximity'),   # gamma^(T-t) R
        'episode_outcome': (log['episode_rewards'].copy(), k_step[1]),      # the trajectory's return
    }


def with_future_signal(log):
    """Rows of tasks whose future credit is not identically flat (e.g. every rollout failed)."""
    keep = np.zeros(len(log['uid']), dtype=bool)
    for task in np.unique(log['uid']):
        ids = np.flatnonzero((log['uid'] == task) & log['eligible'])
        if len(ids) > 1 and np.ptp(log['raw_progress'][ids]) > EPS:
            keep[log['uid'] == task] = True
    return keep


def analyse_step(log, gamma, horizon, mode, success_reward, source='log'):
    """Flat {statistic: value} for one training step."""
    groups = trajectory_groups(log)
    verify_reconstruction(log, groups, gamma, horizon, mode, success_reward, source)
    tasks, trajectories = log['uid'], log['traj_uid']
    success = abs(log['episode_rewards'] - success_reward) < 1e-9
    informative = with_future_signal(log)
    both = log['eligible'] & log['history_live']
    record = {'rows_both_channels': float(both.sum()),
              'rows_without_future_signal': float((both & ~informative).sum()),
              'terminal_window_frac': float(np.mean(log['terminal'][both])) if both.any() else float('nan')}
    populations = {'all': both & informative, 'nonterminal': both & informative & ~log['terminal']}
    for population, mask in populations.items():
        for pair, h_key, f_key in PAIRS:
            stats = relation_statistics(log[h_key], log[f_key], tasks, success, mask)
            record.update({f'{pair}/{population}/{k}': v for k, v in stats.items()})
        stats = share_statistics(log['history_adv'], log['raw_progress'], tasks, trajectories, success, mask)
        record.update({f'shares/{population}/{k}': v for k, v in stats.items()})
    reference = log['progress_normalized']
    for name, (raw, eligible) in alternative_signals(log, groups, gamma, horizon, success_reward).items():
        alternative, _, _ = standardize_progress(raw, tasks, eligible)
        for population in POPULATIONS:
            mask = log['eligible'] & eligible & informative
            if population == 'nonterminal':
                mask = mask & ~log['terminal']
            x, y = reference[mask], alternative[mask]
            disagreement = sign_disagreement(x, y)
            record.update({f'alternative/{name}/{population}/pearson': pearson(x, y),
                           f'alternative/{name}/{population}/spearman': spearman(x, y),
                           f'alternative/{name}/{population}/sign_agreement': 1. - disagreement,
                           f'alternative/{name}/{population}/rows': float(mask.sum())})
    return record


# ---------------------------------------------------------------------------
# aggregation and report
# ---------------------------------------------------------------------------
def parse_ranges(text):
    ranges = []
    for part in text.split(','):
        match = re.fullmatch(r'\s*(\d+)\s*-\s*(\d+)\s*', part)
        if not match or int(match.group(1)) > int(match.group(2)):
            raise ValueError(f'Invalid step range: {part!r} (expected A-B)')
        ranges.append((int(match.group(1)), int(match.group(2))))
    return ranges


def summarize(per_step, ranges):
    """{range label: {statistic: {mean, std, steps}}}, averaging finite per-step values."""
    summary = {}
    for first, last in ranges:
        records = [r for step, r in per_step.items() if first <= step <= last]
        if not records:
            continue
        stats = {}
        for key in records[0]:
            values = np.array([r[key] for r in records], dtype=float)
            values = values[np.isfinite(values)]
            stats[key] = dict(mean=float(values.mean()) if len(values) else None,
                              std=float(values.std()) if len(values) else None, steps=int(len(values)))
        summary[f'{first}-{last}'] = stats
    return summary


def _cell(stat):
    return '—' if stat['mean'] is None else f"{stat['mean']:.3f} ± {stat['std']:.3f}"


def _table(columns, rows):
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    return '\n'.join(lines + ['| ' + ' | '.join(r) + ' |' for r in rows])


def render(meta, summary):
    out = [f"# Credit-channel analysis: {meta['run']}", '',
           f"{meta['steps_analysed']} step logs ({meta['first_step']}-{meta['last_step']}), "
           f"gamma {meta['gamma']:g}, future horizon {meta['horizon']}, mode `{meta['mode']}`. "
           'Entries are mean ± standard deviation across training steps.', '',
           'These are statistics of credit signals on the logged rollouts. They say how '
           'different two signals are, not which one trains a better policy.', '',
           '## A. Historical credit H versus future credit F', '',
           '`applied` is each channel\'s contribution to the actor advantage; `raw` is '
           'H = Y − C[Y] and F = ΔV before any normalization, where zero means "as good as '
           'the peers" and "the contextual value did not change". Positive-in-failed is the '
           'share of turns of failed trajectories with positive credit, in tasks that have '
           'both outcomes; negative-in-successful is its mirror image.', '',
           'A share is the fraction of a channel\'s within-task variance explained by which '
           'trajectory a turn belongs to, or by whether that trajectory succeeded; `chance` '
           'is that fraction for labels that carry no information.']
    steps = {label: f"{label} (n={stats['rows_both_channels']['steps']})" for label, stats in summary.items()}
    for population, title in (('all', 'All turns with both channels'),
                              ('nonterminal', 'Non-terminal windows only (F does not use the turn\'s own outcome)')):
        relation, shares = [], []
        for label, stats in summary.items():
            for pair, _, _ in PAIRS:
                p = f'{pair}/{population}/'
                relation.append([steps[label], pair] + [_cell(stats[p + k]) for k in (
                    'pearson', 'spearman', 'sign_disagreement', 'positive_in_failed_h', 'positive_in_failed_f',
                    'negative_in_successful_h', 'negative_in_successful_f')])
            p = f'shares/{population}/'
            shares.append([steps[label]] + [_cell(stats[p + k]) for k in (
                'trajectory_h', 'trajectory_f', 'trajectory_chance_h', 'outcome_h', 'outcome_f', 'outcome_chance_h')])
        out += ['', f'### {title}', '',
                _table(['steps', 'credit', 'Pearson', 'Spearman', 'sign disagreement',
                        'positive in failed, H', 'positive in failed, F',
                        'negative in successful, H', 'negative in successful, F'], relation), '',
                _table(['steps', 'trajectory share, H', 'trajectory share, F', 'chance',
                        'outcome share, H', 'outcome share, F', 'chance'], shares)]
    out += ['', '## B. The run\'s future credit versus simpler signals', '',
            'Each signal is recomputed from the logged contextual values and standardized '
            'within task exactly as the run\'s future credit is.']
    names = sorted({k.split('/')[1] for stats in summary.values() for k in stats if k.startswith('alternative/')})
    for population, title in (('all', 'All eligible turns'), ('nonterminal', 'Non-terminal windows only')):
        rows = [[steps[label], f'`{name}`'] + [_cell(stats[f'alternative/{name}/{population}/{k}'])
                                              for k in ('pearson', 'spearman', 'sign_agreement')]
                for label, stats in summary.items() for name in names]
        out += ['', f'### {title}', '', _table(['steps', 'signal', 'Pearson', 'Spearman', 'sign agreement'], rows)]
    return '\n'.join(out) + '\n'


def resolve_run(run, gamma, horizon, mode):
    """(log directory, gamma, horizon, mode); explicit arguments override config.json."""
    run = Path(run)
    logs = run / 'outputs/future_progress'
    config = {}
    if (run / 'config.json').is_file():
        config = json.loads((run / 'config.json').read_text()).get('config', {})
    if not logs.is_dir():
        logs = run
    gamma = config.get('gamma') if gamma is None else gamma
    horizon = config.get('ccpo_progress_horizon') if horizon is None else horizon
    mode = (config.get('ccpo_progress_mode') or 'difference') if mode is None else mode
    if gamma is None or horizon is None:
        raise ValueError(f'No config.json with gamma and ccpo_progress_horizon under {run}; '
                         'pass --gamma and --horizon')
    return logs, float(gamma), int(horizon), str(mode)


def analyse_run(run, *, gamma=None, horizon=None, mode=None, steps=None, phases=DEFAULT_PHASES,
                success_reward=10.):
    logs, gamma, horizon, mode = resolve_run(run, gamma, horizon, mode)
    if mode not in PROGRESS_MODES:
        raise ValueError(f'Unknown future-progress mode: {mode}')
    if horizon < 1:
        raise ValueError('This run has no future credit (horizon 0); there is nothing to compare')
    selected = parse_ranges(steps) if steps else None
    files = {}
    for path in sorted(logs.glob('step-*.csv')):
        match = re.fullmatch(r'step-(\d+)\.csv', path.name)
        if match and (selected is None or any(a <= int(match.group(1)) <= b for a, b in selected)):
            files[int(match.group(1))] = path
    if not files:
        raise FileNotFoundError(f'No step-NNNN.csv logs selected under {logs}')
    per_step = {step: analyse_step(read_step(path), gamma, horizon, mode, success_reward, str(path))
                for step, path in files.items()}
    ranges = parse_ranges(phases) + [(min(files), max(files))]
    meta = dict(run=Path(run).name, logs=str(logs), gamma=gamma, horizon=horizon, mode=mode,
                success_reward=success_reward, steps_analysed=len(files),
                first_step=min(files), last_step=max(files))
    return meta, per_step, summarize(per_step, list(dict.fromkeys(ranges)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path, help='experiment folder, or a directory of step-NNNN.csv logs')
    p.add_argument('--steps', help='restrict to step ranges, e.g. 1-150 or 1-30,121-150')
    p.add_argument('--phases', default=DEFAULT_PHASES, help='step ranges to average over (default %(default)s)')
    p.add_argument('--gamma', type=float, help='default: the run\'s config.json')
    p.add_argument('--horizon', type=int, help='future horizon K_F; default: the run\'s config.json')
    p.add_argument('--mode', choices=PROGRESS_MODES, help='future-credit mode; default: the run\'s config.json')
    p.add_argument('--success-reward', type=float, default=10.)
    p.add_argument('--output', type=Path, help='default: <run>/outputs/credit_channels')
    a = p.parse_args(argv)
    meta, per_step, summary = analyse_run(a.run, gamma=a.gamma, horizon=a.horizon, mode=a.mode, steps=a.steps,
                                          phases=a.phases, success_reward=a.success_reward)
    output = a.output or ((a.run / 'outputs' if (a.run / 'outputs').is_dir() else a.run) / 'credit_channels')
    output.mkdir(parents=True, exist_ok=True)
    report = render(meta, summary)
    (output / 'credit_channels.md').write_text(report)
    (output / 'credit_channels.json').write_text(json.dumps(
        dict(meta=meta, summary=summary, per_step={str(k): v for k, v in per_step.items()}),
        indent=2, sort_keys=True) + '\n')
    keys = list(next(iter(per_step.values())))
    with (output / 'credit_channels_per_step.csv').open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['step'] + keys)
        writer.writerows([step] + [record[k] for k in keys] for step, record in per_step.items())
    print(report)
    print(f'Wrote {output}/credit_channels.{{md,json}} and credit_channels_per_step.csv')


if __name__ == '__main__':
    main()
