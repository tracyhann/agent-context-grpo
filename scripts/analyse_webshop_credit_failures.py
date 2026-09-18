#!/usr/bin/env python3
"""Summarize saved M11 training credit arrays without running a model."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
M11=ROOT/'experiments/m11-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918'
keys=['uid','traj_uid','turn_index','episode_rewards','terminal','raw_progress','progress_normalized','m5_edge_normalized','m5_edge_raw','current_value','future_value','current_kernel','current_J','current_n_eff','current_level','history_adv','combined_applied','is_action_valid']
rows=[]
for step in range(1,151):
    with np.load(M11/f'outputs/future_progress/step-{step:04d}.npz',allow_pickle=False) as z:
        a={k:z[k] for k in keys}
    success=a['episode_rewards']==10; nt=~a['terminal']; first=a['turn_index']==0
    taskwins={t:int(np.sum(first & (a['uid']==t) & success)) for t in np.unique(a['uid'])}
    lone=success & np.array([taskwins[t]==1 for t in a['uid']])
    def frac(mask,condition):return float(np.mean(condition[mask])) if np.any(mask) else None
    def corr(x,y,mask):
        x,y=x[mask],y[mask]
        return float(np.corrcoef(x,y)[0,1]) if len(x)>2 and x.std()>1e-10 and y.std()>1e-10 else None
    rows.append({'step':step,'rows':len(nt),'episodes':int(first.sum()),'successes':int((first&success).sum()),
        'tasks':len(taskwins),'all_failed_tasks':sum(v==0 for v in taskwins.values()),'one_success_tasks':sum(v==1 for v in taskwins.values()),
        'success_nonterminal_rows':int((success&nt).sum()),
        'successful_nonterminal_raw_negative_fraction':frac(success&nt,a['raw_progress']<0),
        'successful_nonterminal_normalized_negative_fraction':frac(success&nt,a['progress_normalized']<0),
        'successful_nonterminal_m5_edge_negative_fraction':frac(success&nt,a['m5_edge_normalized']<0),
        'successful_nonterminal_combined_negative_fraction':frac(success&nt,a['combined_applied']<0),
        'successful_nonterminal_zero_peer_kernel_fraction':frac(success&nt,np.abs(a['current_kernel'])<1e-9),
        'lone_success_episodes':int((lone&first).sum()),'lone_success_nonterminal_rows':int((lone&nt).sum()),
        'lone_success_nonterminal_zero_value_fraction':frac(lone&nt,np.abs(a['current_value'])<1e-9),
        'lone_success_nonterminal_zero_raw_progress_fraction':frac(lone&nt,np.abs(a['raw_progress'])<1e-9),
        'lone_success_nonterminal_negative_progress_fraction':frac(lone&nt,a['progress_normalized']<0),
        'terminal_abs_progress_mass_fraction':float(abs(a['progress_normalized'][~nt]).sum()/max(abs(a['progress_normalized']).sum(),1e-20)),
        'edge_corr_all':corr(a['progress_normalized'],a['m5_edge_normalized'],np.ones(len(nt),bool)),
        'edge_corr_nonterminal':corr(a['progress_normalized'],a['m5_edge_normalized'],nt),
        'nonterminal_task_fallback_fraction':float(np.mean(a['current_level'][nt]>0)),
        'nonterminal_mean_effective_peers':float(np.mean(a['current_n_eff'][nt])),
        'future_history_abs_ratio':float(np.mean(abs(a['progress_normalized']))/max(np.mean(abs(a['history_adv'])),1e-20))})
periods={}
for lo,hi in [(1,30),(31,75),(100,150),(1,150)]:
    subset=[r for r in rows if lo<=r['step']<=hi]
    periods[f'{lo}-{hi}']={k:float(np.mean([r[k] for r in subset if r[k] is not None])) for k in rows[0] if k!='step' and any(r[k] is not None for r in subset)}
    for k in ['tasks','all_failed_tasks','one_success_tasks','episodes','successes','lone_success_episodes','lone_success_nonterminal_rows']:
        periods[f'{lo}-{hi}'][k+'_total']=sum(r[k] for r in subset)
out=M11/'failure-analysis-20260918'
out.mkdir(exist_ok=True)
(out/'credit_diagnostics.json').write_text(json.dumps({'per_step':rows,'period_means':periods},indent=2))
print(json.dumps(periods,indent=2))
