#!/usr/bin/env python3
"""CPU-only, paired mechanism audit of the saved M10/M11 step-150 batches.

No checkpoint, model, environment, training process, or GPU is started.
Counterfactual estimates use identical archived trajectories and reward labels.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analyse_adaptive_progress import TaskReadout, paired_progress, clean

RUNS = {
    'M10': 'm10-ccpo-attncred-ctxadv-future-progress-alfworld-1.5b-2gpu-20260918',
    'M11': 'm11-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(clean(data), indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(clean(rows))


def mean(x, mask=None):
    x = np.asarray(x)
    x = x if mask is None else x[mask]
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def corr(x, y, mask):
    x, y = np.asarray(x)[mask], np.asarray(y)[mask]
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 1e-10 and y.std() > 1e-10 else None


def ranking_metrics(a, x, y, mask):
    """Within-task ranks; ties retain average ranks and expand top-20% sets."""
    def rank(z):
        _, inverse, counts = np.unique(z, return_inverse=True, return_counts=True)
        return (np.cumsum(counts) - (counts-1)/2)[inverse]
    ranks, overlaps = [], []
    for task in np.unique(a['uid']):
        use = mask & (a['uid']==task)
        xx, yy = x[use], y[use]
        if len(xx)<3 or xx.std()<=1e-10 or yy.std()<=1e-10:
            continue
        ranks.append(corr(rank(xx),rank(yy),np.ones(len(xx),dtype=bool)))
        if abs(xx).std()>1e-10 and abs(yy).std()>1e-10:
            top_x = abs(xx)>=np.quantile(abs(xx),.8)
            top_y = abs(yy)>=np.quantile(abs(yy),.8)
            overlaps.append(float(np.sum(top_x&top_y)/np.sum(top_x|top_y)))
    return dict(task_spearman_mean=mean(ranks), rank_tasks=len(ranks),
                top_abs20_jaccard_mean=mean(overlaps), top_abs20_tasks=len(overlaps))


def sign(x):
    return np.where(np.abs(x) <= 1e-8, 0, np.sign(x))


def task_macro(a, x, mask):
    """Equal trajectory weight inside each task, then equal task weight."""
    out = {}
    for task in np.unique(a['uid']):
        values = []
        for traj in np.unique(a['traj_uid'][a['uid'] == task]):
            take = mask & (a['uid'] == task) & (a['traj_uid'] == traj)
            value = mean(x, take)
            if value is not None:
                values.append(value)
        if values:
            out[str(task)] = float(np.mean(values))
    return out


def ci(values, seed=150, repeats=5000):
    x = np.asarray(list(values), dtype=float)
    if not len(x):
        return [None, None]
    rng = np.random.default_rng(seed)
    boot = x[rng.integers(len(x), size=(repeats, len(x)))].mean(1)
    return np.quantile(boot, [.025, .975]).tolist()


def blend(diag, kappa=2., fixed=None, uniform=False):
    base = np.asarray(diag['uniform_values'] if uniform else diag['kernel_values']).copy()
    exact = np.asarray(diag['level_values']) == 0
    prior = np.asarray(diag['task_prior_values'])
    j = np.asarray(diag['support_values'])
    lam = j / (j + kappa) if fixed is None else np.full(len(j), fixed)
    use = exact & np.isfinite(prior)
    base[use] = lam[use] * base[use] + (1-lam[use]) * prior[use]
    return base


def boot_read(readout, rows, counts, kappa=2., fixed=None):
    own = readout.trajectory[rows[0]]
    assert np.all(readout.trajectory[rows] == own)
    other = np.flatnonzero(np.arange(readout.k) != own)
    ix = np.ix_(rows, other)
    support = readout.present[ix] @ counts
    kernel = (readout.num[ix] @ counts) / np.maximum(readout.den[ix] @ counts, 1e-12)
    fallback = (readout.fnum[ix] @ counts) / np.maximum(readout.fden[ix] @ counts, 1e-12)
    prior = (readout.task_num[other] @ counts) / (readout.task_den[other] @ counts)
    lam = support / (support + kappa) if fixed is None else fixed
    return np.where(support > 0, lam * kernel + (1-lam) * prior, fallback)


def analyze(tag, out, draws):
    exp = ROOT / 'experiments' / RUNS[tag]
    snapshot, config = exp/'outputs/future_progress/step-0150.npz', exp/'config.json'
    cfg = json.loads(config.read_text())['config']
    assert cfg['ccpo_progress_horizon'] == 1 and cfg['group_size'] == 8
    for key, value in cfg.items():
        if key.startswith('ccpo_'):
            os.environ['ACG_' + key.upper()] = str(value)
    os.environ.update(ACG_CCPO_DUMP='', ACG_CCPO_GDUMP='')
    os.environ.pop('ACG_EXP_DIR', None)
    from ccpo import core_ccpo
    core = importlib.reload(core_ccpo)
    from ccpo.future_progress import progress_from_values, standardize_progress
    with np.load(snapshot, allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    n = len(a['uid'])
    live, nt = a['live'].astype(bool), ~a['terminal'].astype(bool)
    first = a['turn_index'] == 0
    success = a['episode_rewards'] == 10
    groups = defaultdict(list)
    for i, key in enumerate(zip(a['uid'], a['traj_uid'])):
        groups[key].append(i)
    groups = [np.array(ids) for ids in groups.values()]
    assert all(np.array_equal(a['turn_index'][ids], np.arange(len(ids))) for ids in groups)
    assert len(set(zip(a['uid'], a['traj_uid'], a['turn_index']))) == n
    assert np.all(np.isfinite(a['history_hidden'])) and np.all(np.isfinite(a['current_phi']))
    np.testing.assert_allclose(a['history_applied']+a['future_applied'], a['combined_applied'], atol=3e-5)
    labels = {'Y': a['target'], 'Z': a['value_target']}
    hidden_dim = a['history_hidden'].shape[1]
    hidden = a['current_phi'][:, :hidden_dim].copy()
    hidden /= np.maximum(np.linalg.norm(hidden, axis=1, keepdims=True), 1e-9)
    assert len(np.unique(a['traj_uid'])) == len(groups), 'Trajectory IDs must be globally unique'
    # Recompute whitening independently once, checking the saved block extraction.
    np.testing.assert_allclose(hidden, core.whiten_feats(a['history_hidden']), atol=2e-7)
    kwargs = dict(response_mask=torch.ones((n, 1)), anchor_obs=a['anchor_obs'],
                  index=a['uid'], traj_index=a['traj_uid'], phi=core.FrozenPhi(),
                  episode_rewards=a['episode_rewards'], is_action_valid=a['is_action_valid'],
                  gamma=cfg['gamma'], edge_w=0., target='return', return_diag=True, dump_enabled=False)
    diags, estimates, parity = {}, {}, {}
    for label, target in labels.items():
        for feature_name, features in [('context', a['current_phi']), ('hidden_only', hidden)]:
            _, d = core.ccpo_step_advantage(step_rewards=torch.tensor(target, dtype=torch.float64),
                                            prepared_phi=features, **kwargs)
            diags[label, feature_name] = d
        d = diags[label, 'context']
        estimates[label] = {
            'context_k2': d['baseline_values'],
            'hidden_only_k2': diags[label, 'hidden_only']['baseline_values'],
            'uniform_k2': blend(d, uniform=True),
            'context_k4': blend(d, kappa=4),
            'context_full': blend(d, fixed=1.),
            'context_fixed05': blend(d, fixed=.5),
        }
        expected = a['history_baseline'] if label == 'Y' else a['current_value']
        parity[label+'_baseline_max_abs_error'] = float(np.max(abs(d['baseline_values']-expected)))
        np.testing.assert_allclose(d['baseline_values'], expected, atol=2e-7, rtol=1e-8)
        np.testing.assert_allclose(blend(d), expected, atol=2e-7, rtol=1e-8)
        # Independently validate closed-form counterfactuals against the actual
        # estimator, rather than testing the formula against itself.
        saved = core._PRIOR_KAPPA, core._LK_FIX, core._LAM_FIX
        for name, kappa, pin, attention in [('uniform_k2',2.,'', '0'),
                ('context_k4',4.,'', '1'), ('context_full',2.,'1', '1'),
                ('context_fixed05',2.,'.5', '1')]:
            core._PRIOR_KAPPA, core._LK_FIX, core._LAM_FIX = kappa, pin, attention
            _, check = core.ccpo_step_advantage(step_rewards=torch.tensor(target, dtype=torch.float64),
                                                prepared_phi=a['current_phi'], **kwargs)
            np.testing.assert_allclose(check['baseline_values'], estimates[label][name], atol=2e-7, rtol=1e-8)
        core._PRIOR_KAPPA, core._LK_FIX, core._LAM_FIX = saved
        parity[label+'_counterfactual_core_checks_passed'] = True
    np.testing.assert_allclose(labels['Y'].astype(np.float32).astype(float)-estimates['Y']['context_k2'],
                               a['history_adv'], atol=2e-6)

    def potential(value, horizon=1):
        raw, future, endpoint, window, eligible = progress_from_values(value, groups, a['episode_rewards'], horizon)
        f, _, _ = standardize_progress(raw, a['uid'], eligible)
        return raw, f

    raw, f = potential(estimates['Z']['context_k2'])
    np.testing.assert_allclose(raw, a['raw_progress'], atol=2e-7)
    np.testing.assert_allclose(f, a['progress_normalized'], atol=2e-7)
    vals, nodes, successors = core.g2po_node_values(a['anchor_obs'], a['uid'], a['traj_uid'], a['episode_rewards'], cfg['gamma'])
    edge_value = np.asarray([vals[node] for node in nodes])
    edge_raw, edge_f = potential(edge_value)
    np.testing.assert_allclose(edge_raw, a['m5_edge_raw'], atol=1e-10)
    np.testing.assert_allclose(edge_f, a['m5_edge_normalized'], atol=1e-10)
    parity['edge_max_abs_error'] = float(np.max(abs(edge_f-a['m5_edge_normalized'])))

    # Paired prediction errors. Return targets are noisy realized outcomes,
    # not independently measured true state values or policy-gradient variance.
    exact = a['current_level'] == 0
    revisit = a['current_context'][:, 2] > 0
    strata = {'all': live, 'exact': exact & live, 'exact_revisit': exact & revisit & live,
              'exact_first_visit': exact & ~revisit & live, 'task_fallback': ~exact & live}
    errors = []
    for label, target in labels.items():
        for subset, mask in strata.items():
            base = task_macro(a, (estimates[label]['context_k2']-target)**2, mask)
            for name, estimate in estimates[label].items():
                task_mse = task_macro(a, (estimate-target)**2, mask)
                keys = sorted(set(base) & set(task_mse))
                diffs = [task_mse[k]-base[k] for k in keys]
                low, high = ci(diffs)
                errors.append(dict(run=tag, target=label, subset=subset, variant=name, rows=int(mask.sum()),
                                   tasks=len(keys), mse=mean(list(task_mse.values())),
                                   delta_mse_vs_context=mean(diffs), delta_ci_low=low, delta_ci_high=high))

    # Same peer resampling at current and future endpoints preserves covariance.
    schemes = {'context_k2': (2., None), 'context_k4': (4., None),
               'context_full': (2., 1.), 'context_fixed05': (2., .5)}
    raw_by_scheme = {name: potential(estimates['Z'][name])[0] for name in schemes}
    boot = {name: {k: np.zeros(n) for k in ['value_var', 'progress_var', 'sign_agreement', 'independent_var']}
            for name in schemes}
    for task in np.unique(a['uid']):
        ids = np.flatnonzero(a['uid'] == task)
        reader = TaskReadout(a['current_phi'][ids], a['value_target'][ids].astype(np.float32).astype(float),
                             a['anchor_obs'][ids], a['traj_uid'][ids], tau=cfg['ccpo_tau'])
        for own in range(reader.k):
            loc = np.flatnonzero(reader.trajectory == own)
            absolute = ids[loc]
            original = reader.read(loc, np.ones((reader.k-1, 1)))[0][:, 0]
            np.testing.assert_allclose(original, a['current_value'][absolute], atol=2e-7, rtol=1e-8)
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(f'{tag}:{task}:{own}:150'.encode()).digest()[:8], 'little'))
            counts = rng.multinomial(reader.k-1, np.full(reader.k-1, 1/(reader.k-1)), size=draws).T
            lookup = {int(v): j for j, v in enumerate(absolute)}
            endpoint = np.array([lookup[int(e)] if e >= 0 else -1 for e in a['endpoint_index'][absolute]])
            for name, (kappa, fixed) in schemes.items():
                values = boot_read(reader, loc, counts, kappa, fixed)
                p = paired_progress(values, endpoint, a['episode_rewards'][absolute])
                reference_raw = raw_by_scheme[name][absolute]
                b = boot[name]
                b['value_var'][absolute] = values.var(1, ddof=1)
                b['progress_var'][absolute] = p.var(1, ddof=1)
                b['sign_agreement'][absolute] = np.mean(sign(p) == sign(reference_raw[:, None]), axis=1)
                vfuture = np.zeros(len(loc))
                keep = endpoint >= 0
                vfuture[keep] = values[endpoint[keep]].var(1, ddof=1)
                b['independent_var'][absolute] = b['value_var'][absolute] + vfuture
    parity['bootstrap_point_reconstruction_passed'] = True

    support = []
    for j in range(1, 8):
        mask = exact & (a['current_J'] == j)
        if mask.any():
            support.append(dict(run=tag, level='exact', J=j, rows=int(mask.sum()),
                                lambda_k=mean(a['current_lambda_k'], mask), n_eff=mean(a['current_n_eff'], mask),
                                value_mse=mean(list(task_macro(a, (a['current_value']-a['value_target'])**2, mask).values())),
                                value_boot_sd=mean(np.sqrt(boot['context_k2']['value_var']), mask)))
    fallback = ~exact
    support.append(dict(run=tag, level='task_fallback', J=None, rows=int(fallback.sum()),
                        lambda_k=mean(a['current_lambda_k'], fallback), n_eff=mean(a['current_n_eff'], fallback),
                        value_mse=mean(list(task_macro(a, (a['current_value']-a['value_target'])**2, fallback).values())),
                        value_boot_sd=mean(np.sqrt(boot['context_k2']['value_var']), fallback)))
    stability = []
    for name, b in boot.items():
        p = potential(estimates['Z'][name])[0]
        nz = nt & (abs(p) > 1e-8)
        stability.append(dict(run=tag, variant=name, nonterminal_nonzero_rows=int(nz.sum()),
                              value_boot_sd=mean(np.sqrt(b['value_var'])),
                              progress_boot_sd_nonterminal=mean(np.sqrt(b['progress_var']), nt),
                              sign_agreement_nonterminal_nonzero=mean(b['sign_agreement'], nz),
                              weak_sign_rows_fraction=mean(b['sign_agreement'] < .8, nz),
                              paired_progress_variance=mean(b['progress_var'], nt),
                              independent_endpoint_variance=mean(b['independent_var'], nt)))

    terminal_share = lambda x: float(abs(x[~nt]).sum()/max(abs(x).sum(), 1e-12))
    H, F, A = a['history_applied'], a['future_applied'], a['combined_applied']
    component = []
    successful_per_task = {t:int(np.sum(first & success & (a['uid']==t))) for t in np.unique(a['uid'])}
    lone = success & np.array([successful_per_task[t] == 1 for t in a['uid']])
    component_masks = {'all': live, 'nonterminal': nt & live, 'terminal': ~nt & live,
                       'success_nonterminal': success & nt & live, 'failure_nonterminal': ~success & nt & live,
                       'sole_success_nonterminal': lone & nt & live, 'invalid_action': a['is_action_valid'] == 0}
    for subset, mask in component_masks.items():
        nz = mask & (abs(H)>1e-8) & (abs(F)>1e-8)
        component.append(dict(run=tag, subset=subset, rows=int(mask.sum()),
                              history_abs=mean(abs(H), mask), future_abs=mean(abs(F), mask),
                              history_future_corr=corr(H,F,mask), nonzero_opposite_sign_fraction=mean(H*F < 0,nz),
                              nonzero_rows=int(nz.sum()), combined_flips_history_fraction=mean(H*A < -1e-10,mask),
                              negative_future_fraction=mean(F < -1e-8,mask), negative_combined_fraction=mean(A < -1e-8,mask),
                              zero_raw_negative_normalized_count=int(np.sum(mask & (abs(raw)<1e-8) & (f < -1e-8)))))

    fusion = []
    for weight in (0., .25, .5, 1.):
        mixed = H + weight * F
        fusion.append(dict(run=tag, weight=weight,
                           changed_sign_vs_history=mean(H*mixed < -1e-10),
                           negative_success_nonterminal=mean(mixed < -1e-8, success & nt),
                           positive_failure_nonterminal=mean(mixed > 1e-8, ~success & nt),
                           future_to_history_abs=float(np.mean(abs(weight*F))/max(np.mean(abs(H)),1e-12))))

    ladder_values = {'m5_uniform_self_inclusive': edge_value,
                     'uniform_loo_no_shrink': blend(diags['Z','context'], fixed=1., uniform=True),
                     'context_loo_no_shrink': estimates['Z']['context_full'],
                     'context_loo_k2': estimates['Z']['context_k2'],
                     'hidden_loo_k2': estimates['Z']['hidden_only_k2']}
    edge = []
    exact_pair = nt & exact & (a['future_level']==0)
    for name, value in ladder_values.items():
        p, credit = potential(value)
        for subset, mask in [('all', live), ('nonterminal', nt & live), ('exact_both_nonterminal', exact_pair)]:
            edge.append(dict(run=tag, variant=name, subset=subset, rows=int(mask.sum()),
                             raw_corr_to_m5=corr(p,edge_raw,mask), normalized_corr_to_m5=corr(credit,edge_f,mask),
                             sign_agreement_including_zero=mean(sign(credit)==sign(edge_f),mask),
                             **ranking_metrics(a,credit,edge_f,mask),
                             normalized_rmse_to_m5=float(np.sqrt(np.mean((credit[mask]-edge_f[mask])**2)))))

    temporal = []
    remaining = a['episode_lengths']-a['turn_index']
    for label, mask in [('terminal',remaining==1), ('1 step before terminal',remaining==2),
                        ('2-3 steps before', (remaining>=3)&(remaining<=4)),
                        ('4-7 steps before',(remaining>=5)&(remaining<=8)), ('8+ steps before',remaining>=9)]:
        temporal.append(dict(run=tag, position=label, rows=int(mask.sum()),
                             raw_abs=mean(abs(raw),mask), normalized_abs=mean(abs(f),mask),
                             history_applied_abs=mean(abs(H),mask), future_applied_abs=mean(abs(F),mask),
                             future_absolute_mass=float(abs(F[mask]).sum()/max(abs(F).sum(),1e-12))))
    behavior, case_rows = [], []
    for ids in groups:
        behavior.append(dict(run=tag, task=str(a['uid'][ids[0]]), trajectory=str(a['traj_uid'][ids[0]]),
                             success=bool(success[ids[0]]), length=len(ids), revisit_fraction=float(revisit[ids].mean()),
                             invalid_fraction=float((a['is_action_valid'][ids]==0).mean()),
                             conflict_fraction=float((H[ids]*F[ids]<-1e-10).mean()),
                             future_abs=float(abs(F[ids]).mean())))
    for outcome in (True, False):
        candidates = [b for b in behavior if b['success']==outcome]
        if not candidates:
            continue
        chosen = max(candidates, key=lambda b:(b['conflict_fraction'],b['length']))
        ids = np.flatnonzero(a['traj_uid']==chosen['trajectory'])
        for i in ids:
            case_rows.append(dict(run=tag, selection='highest opposite-sign turn fraction within outcome; length tiebreak',
                                  success=outcome, task=str(a['uid'][i]), trajectory=str(a['traj_uid'][i]),
                                  turn=int(a['turn_index'][i]), terminal=bool(a['terminal'][i]),
                                  observation=str(a['anchor_obs'][i])[:350], action_valid=float(a['is_action_valid'][i]),
                                  history=float(H[i]), future=float(F[i]), combined=float(A[i]), raw_progress=float(raw[i]),
                                  current_J=float(a['current_J'][i]), future_J=float(a['future_J'][i]),
                                  current_lambda=float(a['current_lambda_k'][i]), future_lambda=float(a['future_lambda_k'][i])))
    source = dict(run=tag, snapshot=str(snapshot.relative_to(ROOT)), snapshot_sha256=digest(snapshot),
                  config_sha256=digest(config), horizon=cfg['ccpo_progress_horizon'],
                  rows=n, tasks=len(np.unique(a['uid'])), trajectories=len(groups),
                  training_batch_success=float(success[first].mean()),
                  terminal_rows_fraction=float((~nt).mean()), terminal_raw_mass=terminal_share(raw),
                  terminal_normalized_mass=terminal_share(f), terminal_applied_future_mass=terminal_share(F),
                  task_fallback_rows_fraction=float((~exact).mean()),
                  no_success_tasks=sum(x==0 for x in successful_per_task.values()),
                  one_success_tasks=sum(x==1 for x in successful_per_task.values()),
                  parity=parity)
    print(tag, json.dumps(source), flush=True)
    result = dict(source=source, errors=errors, support=support, stability=stability,
                  components=component, fusion=fusion, edge=edge, temporal=temporal, behavior=behavior, cases=case_rows)
    write_json(out/f'{tag}-results.json', result)
    return result


def figures(results, out):
    plt.rcParams.update({'font.size':9, 'axes.spines.top':False, 'axes.spines.right':False})
    colors = ['#2878a0','#ce6b32','#50876b','#8c64a5','#ad8b36','#777777']
    fig, axes = plt.subplots(2,3,figsize=(16,9))
    for row, r in enumerate(results):
        name=r['source']['run']
        ts=[x for x in r['temporal'] if x['rows']]
        ax=axes[row,0]
        ax.bar(np.arange(len(ts)), [x['future_absolute_mass']*100 for x in ts],color=colors[0])
        ax.set_xticks(np.arange(len(ts)),[x['position'] for x in ts],rotation=25,ha='right')
        ax.set_ylabel('% of absolute applied future credit');ax.set_title(name+': where future credit lands')
        es=[x for x in r['errors'] if x['target']=='Z' and x['subset']=='exact']
        ax=axes[row,1]
        ax.bar(np.arange(len(es)),[x['mse'] for x in es],color=colors)
        ax.set_xticks(np.arange(len(es)),[x['variant'] for x in es],rotation=35,ha='right')
        ax.set_ylabel('Task/trajectory-balanced MSE');ax.set_title(name+': leave-trajectory-out potential prediction')
        ss=[x for x in r['support'] if x['level']=='exact']
        ax=axes[row,2]
        ax.plot([x['J'] for x in ss],[x['value_boot_sd'] for x in ss],'-o',color=colors[0])
        ax.set_xlabel('Distinct peer trajectories J');ax.set_ylabel('Conditional bootstrap SD of V')
        for x in ss: ax.annotate(str(x['rows']), (x['J'],x['value_boot_sd']),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7)
        ax.set_title(name+': support vs sensitivity (labels: row counts)')
    fig.suptitle('Step 150 snapshots: descriptive mechanism audit, not validation performance')
    fig.tight_layout();fig.savefig(out/'mechanisms.png',dpi=150);fig.savefig(out/'mechanisms.pdf');plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(13,9))
    for row,r in enumerate(results):
        ax=axes[row,0];items=[x for x in r['components'] if x['subset'] in ('nonterminal','terminal','success_nonterminal','failure_nonterminal')]
        xs=np.arange(len(items))
        ax.bar(xs-.18,[100*(x['nonzero_opposite_sign_fraction'] or 0) for x in items],.36,label='H/F opposite signs among nonzero pairs',color=colors[0])
        ax.bar(xs+.18,[100*(x['combined_flips_history_fraction'] or 0) for x in items],.36,label='Combined flips history sign, all rows',color=colors[1])
        ax.set_xticks(xs,[x['subset'] for x in items],rotation=25,ha='right');ax.set_ylabel('%');ax.legend(fontsize=7)
        ax.set_ylim(0, 1.4*max(1., max(100*(x['nonzero_opposite_sign_fraction'] or 0) for x in items)))
        ax.set_title(r['source']['run']+': scalar credit interaction (not gradient alignment)')
        ax=axes[row,1];es=[x for x in r['edge'] if x['subset']=='nonterminal'];xs=np.arange(len(es))
        ax.bar(xs,[x['normalized_corr_to_m5'] or 0 for x in es],color=colors[:len(es)])
        ax.set_xticks(xs,[x['variant'] for x in es],rotation=30,ha='right');ax.set_ylim(-1,1.05)
        ax.set_ylabel('Pearson correlation, nonterminal rows');ax.set_title(r['source']['run']+': edge readout changes, H1 fixed')
    fig.tight_layout();fig.savefig(out/'credit_and_edge.png',dpi=150);fig.savefig(out/'credit_and_edge.pdf');plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(13,8))
    for row,r in enumerate(results):
        for col,outcome in enumerate((True,False)):
            cs=[x for x in r['cases'] if x['success']==outcome]
            ax=axes[row,col]
            for i,k in enumerate(('history','future','combined')):
                ax.plot([x['turn'] for x in cs],[x[k] for x in cs],'-o',ms=3,label=k,color=colors[i])
            ax.axhline(0,color='#888',lw=.7);ax.set_xlabel('Turn');ax.set_ylabel('Recorded applied credit');ax.legend()
            ax.set_title(r['source']['run']+(' success' if outcome else ' failure')+': selected high-conflict example')
    fig.tight_layout();fig.savefig(out/'trajectory_cases.png',dpi=150);fig.savefig(out/'trajectory_cases.pdf');plt.close(fig)


def self_check():
    # Query has two visits, both of which must be excluded from its peer pool.
    features = np.tile([1., 0.], (4,1))
    obs = np.array(['same']*4); traj = np.array(['A','A','B','C'])
    labels = np.array([99.,-99.,2.,8.])
    counts = np.array([[1.,2.,0.],[1.,0.,2.]])
    reader = TaskReadout(features, labels, obs, traj)
    values = boot_read(reader, np.array([0,1]), counts)
    np.testing.assert_allclose(values, [[5.,2.,8.],[5.,2.,8.]])
    labels[:2] = [-222.,999.]
    other = TaskReadout(features, labels, obs, traj)
    np.testing.assert_allclose(boot_read(other,np.array([0,1]),counts), values)
    progress = paired_progress(values,np.array([1,-1]),np.array([10.,10.]))
    np.testing.assert_allclose(progress,[[0.,0.,0.],[5.,8.,2.]])
    assert progress[0].var() == 0 and values[0].var()+values[1].var() > 0
    assert task_macro({'uid':np.array(['q','q','q']), 'traj_uid':np.array(['a','a','b'])},
                      np.array([0.,0.,6.]),np.ones(3,dtype=bool)) == {'q':3.}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=ROOT/'experiments/step150-posthoc-analysis-20260919')
    ap.add_argument('--bootstrap',type=int,default=512)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    self_check()
    results=[analyze(tag,args.output,args.bootstrap) for tag in RUNS]
    for key in ('errors','support','stability','components','fusion','edge','temporal','behavior','cases'):
        write_csv(args.output/(key+'.csv'),[row for r in results for row in r[key]])
    figures(results,args.output)
    write_json(args.output/'summary.json', dict(bootstrap_draws=args.bootstrap, results=results,
        gpu_used=False, training_launched=False, snapshot_step=150, analytical_self_checks_passed=True,
        source_hashes={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'ccpo/core_ccpo.py',
                       ROOT/'ccpo/future_progress.py',ROOT/'scripts/analyse_adaptive_progress.py']}))
    print('Wrote',args.output,flush=True)


if __name__=='__main__':
    main()
