#!/usr/bin/env python3
"""CPU-only fixed-observation future-context comparison; never runs a policy.

Input NPZ fields (one row per rollout occurrence; Unicode strings, no pickle):
 uid, traj_uid, turn_index, episode_lengths, anchor_obs, returns,
 immediate_rewards, episode_rewards, is_action_valid,
 history_hidden, future_hidden, future_context.

Hidden arrays are frozen reference-model last-prompt-token features, before PCA.
future_hidden must encode the anchor's own next <=2 transitions, not a different
reference group. future_context columns are t, n_unique, revisit, progress at the
end of that window. Terminal windows must be explicitly encoded and marked in
provenance; this tool never invents missing future features from scalar logs.

Same-anchor peers exclude the entire query trajectory. Unsupported exact
anchors get zero added future signal; task-prior shrinkage is retained on
supported anchors. The old OUTLOOK comparator retains its ordinary task backoff.
"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  CUDA_VISIBLE_DEVICES='')
os.environ.update(ACG_CCPO_PHI='hidden+ctx', ACG_CCPO_LAM_FIX='1',
                  ACG_CCPO_PRIOR_KAPPA='2', ACG_CCPO_EDGE_W='0',
                  ACG_CCPO_TARGET='return', ACG_CCPO_STD='task',
                  ACG_CCPO_JWEIGHT_C='0', ACG_CCPO_BACKOFF_TASK='0',
                  ACG_CCPO_LK_FIX='', ACG_CCPO_GATE='hard', ACG_CCPO_TAU='.15',
                  ACG_CCPO_WHITEN='3', ACG_CCPO_CTX_W='1',
                  ACG_CCPO_WMODE='soft', ACG_CCPO_SIM='0', ACG_CCPO_SIM_BACKOFF='0')
os.environ.pop('ACG_CCPO_DUMP', None)
os.environ.pop('ACG_CCPO_GDUMP', None)
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from collections import defaultdict
import hashlib
import json
from unittest.mock import patch
import numpy as np
import torch
from scipy.stats import spearmanr
from ccpo import core_ccpo as core
from ccpo.outlook import canonical_trajectory_rows, ccpo_outlook_advantage

REQUIRED = ('uid', 'traj_uid', 'turn_index', 'episode_lengths', 'anchor_obs',
            'returns', 'immediate_rewards', 'episode_rewards', 'is_action_valid',
            'history_hidden', 'future_hidden', 'future_context')


def canonicalize(batch):
    missing = sorted(set(REQUIRED)-set(batch))
    if missing:
        raise ValueError('Missing full-context inputs: '+', '.join(missing))
    n = len(batch['uid'])
    for key in REQUIRED:
        if len(batch[key]) != n:
            raise ValueError(f'{key}: inconsistent row count')
    take, restore, groups = canonical_trajectory_rows(
        batch['uid'], batch['traj_uid'], batch['turn_index'], batch['episode_lengths'])
    out = {key: np.asarray(batch[key])[take] for key in REQUIRED}
    for key in REQUIRED:
        if not np.array_equal(np.asarray(batch[key]), out[key][restore]):
            raise ValueError(f'Contradictory padded copies: {key}')
    for key in ('history_hidden', 'future_hidden'):
        if out[key].ndim != 2 or not np.isfinite(out[key]).all():
            raise ValueError(f'{key}: need finite, explicitly captured features')
    if out['future_context'].shape != (len(take), 4) or not np.isfinite(out['future_context']).all():
        raise ValueError('future_context must have four finite columns')
    return out, groups


def standardize_tasks(values, tasks, mask=None):
    result = np.full(len(values), np.nan)
    if mask is None:
        mask = np.isfinite(values)
    for task in np.unique(tasks):
        ids = np.flatnonzero((tasks == task) & mask)
        if len(ids) > 1:
            x = values[ids]
            result[ids] = (x-x.mean())/(x.std(ddof=1)+1e-6)
        elif len(ids):
            result[ids] = values[ids]
    return result


def evaluate(batch, gamma=.95):
    a, groups = canonicalize(batch)
    n = len(a['uid'])
    ctx_plus = [dict(zip(('t','n_unique','revisit','progress'), row))
                for row in a['future_context']]
    raw_returns = np.zeros(n)
    for ids in groups:
        running = 0.
        for i in ids[::-1]:
            running = a['immediate_rewards'][i]+gamma*running
            raw_returns[i] = running
    expected = raw_returns-.1*(1-a['is_action_valid'].astype(float))
    if not np.allclose(expected, a['returns'], atol=1e-5):
        raise ValueError('Returns must be env reward-to-go minus current invalid penalty exactly once')
    common = dict(step_rewards=torch.tensor(a['returns'], dtype=torch.float32),
        response_mask=torch.ones(n,1), anchor_obs=a['anchor_obs'], index=a['uid'],
        traj_index=a['traj_uid'], episode_rewards=a['episode_rewards'],
        is_action_valid=a['is_action_valid'], phi=core.FrozenPhi(), gamma=gamma,
        return_diag=True, target='return', edge_w=0, sim=0, sim_backoff=0)
    _, minus = core.ccpo_step_advantage(**common, phi_feats=a['history_hidden'])
    _, plus = core.ccpo_step_advantage(**common, phi_feats=a['future_hidden'], ctx_override=ctx_plus)
    # This explicitly rejects grouping changes between the two readouts.
    if not np.array_equal(minus['live_mask'], plus['live_mask']):
        raise AssertionError('History and future peer support differs')
    eligible = minus['live_mask'] & plus['live_mask']
    bh, bf = minus['baseline_values'], plus['baseline_values']
    fixed = np.full(n, np.nan)
    fixed[eligible] = bf[eligible]-bh[eligible]
    hist = np.where(eligible, a['returns']-bh, np.nan)
    residual = np.where(eligible, a['returns']-bf, np.nan)
    np.testing.assert_allclose(hist[eligible], (fixed+residual)[eligible], atol=1e-10)
    vals, nodes, successors = core.g2po_node_values(
        a['anchor_obs'], a['uid'], a['traj_uid'], a['episode_rewards'], gamma, 10.)
    raw_edge = core._successor_values(vals, successors, n)-np.array([vals[x] for x in nodes])
    edge = standardize_tasks(raw_edge, a['uid'])
    # Original OUTLOOK retains endpoint regrouping and task fallback.
    with patch.object(core, '_BACKOFF_TASK', True):
        old, old_diag = ccpo_outlook_advantage(**common,
            phi_feats=a['history_hidden'], turn_index=a['turn_index'],
            episode_lengths=a['episode_lengths'], immediate_rewards=a['immediate_rewards'])
    old_hist = a['returns']-old_diag['baseline_values']
    old_future = (old.numpy()-.75*old_hist)/.25
    np.testing.assert_allclose(hist[eligible], old_hist[eligible], atol=1e-6)
    peers = defaultdict(set)
    for task,obs,traj in zip(a['uid'],a['anchor_obs'],a['traj_uid']):
        peers[(task,obs)].add(traj)
    support = np.array([len(peers[(task,obs)]-{traj})
        for task,obs,traj in zip(a['uid'],a['anchor_obs'],a['traj_uid'])])
    if not np.array_equal(eligible, support > 0):
        raise AssertionError('Readout support no longer matches exact-anchor peers')
    return dict(task=a['uid'], eligible=eligible,
        terminal=a['turn_index']+2 >= a['episode_lengths'],
        baseline_history=bh, baseline_future=bf,
        fixed_future=fixed, history=hist, future_residual=residual,
        edge=edge, raw_edge=raw_edge, old_outlook=old_future,
        old_outlook_change=.25*(old_future-old_hist),
        fixed_future_normalized=standardize_tasks(fixed,a['uid'],eligible),
        added_signal=np.where(eligible,fixed,0.),
        support=support, shrinkage=np.where(support>0,support/(support+2.),np.nan))


def compare(x,y,mask):
    ids=mask & np.isfinite(x) & np.isfinite(y)
    x,y=x[ids],y[ids]
    nonzero=(abs(x)>1e-5)&(abs(y)>1e-5)
    nonconstant=len(x)>2 and x.std()>1e-10 and y.std()>1e-10
    return dict(n=int(len(x)),pearson=float(np.corrcoef(x,y)[0,1]) if nonconstant else None,
        spearman=float(spearmanr(x,y).statistic) if nonconstant else None,
        nonzero_pairs=int(nonzero.sum()),
        sign_agreement=float(np.mean(np.sign(x[nonzero])==np.sign(y[nonzero]))) if nonzero.any() else None,
        x_absmean=float(np.mean(abs(x))) if len(x) else None,
        y_absmean=float(np.mean(abs(y))) if len(y) else None)


def summarize(a):
    eligible=a['eligible']; nt=eligible & ~a['terminal']
    results=dict(n_all=len(eligible), n_eligible=int(eligible.sum()),coverage=float(eligible.mean()),
        n_tasks=int(len(np.unique(a['task']))),
        n_tasks_with_edge_variance=int(sum(np.ptp(a['edge'][a['task']==t])>1e-10 for t in np.unique(a['task']))),
        eligible_shrinkage_mean=float(a['shrinkage'][eligible].mean()) if eligible.any() else None)
    for name in ('fixed_future','fixed_future_normalized','history','future_residual','old_outlook','old_outlook_change'):
        results[name+'_vs_edge']=compare(a[name],a['edge'],eligible)
        results[name+'_vs_edge_nonterminal']=compare(a[name],a['edge'],nt)
    results['fixed_future_vs_history']=compare(a['fixed_future'],a['history'],eligible)
    return results


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('batches', nargs='+', type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    report={}
    for p in args.batches:
        with np.load(p,allow_pickle=False) as source:
            data=evaluate(dict(source))
        record=summarize(data)
        record['source']=str(p.resolve())
        record['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
        provenance=p.with_suffix('.json')
        record['provenance']=json.loads(provenance.read_text()) if provenance.exists() else None
        report[p.stem]=record
        np.savez_compressed(args.output/(p.stem+'.paired.npz'),**data)
    (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__':
    main()
