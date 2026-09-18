#!/usr/bin/env python3
"""CPU-only replay of contextual future progress on a trusted saved fixed-anchor batch.

No environment, model forward, Ray runtime or optimizer is started. Input is a
locally generated outputs/fixed_anchor/step-NNNN.npz plus its run config.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',type=Path)
    parser.add_argument('--source-config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.source_config.read_text())['config']
    os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for key in ['phi','ctx_w','whiten','tau','wmode','lam_fix','rho','shrink','prior_kappa','lk_fix','jweight_c','std','std_floor','step_norm','gate','sim','sim_backoff','backoff_task','backoff_rho']:
        os.environ['ACG_CCPO_'+key.upper()]=str(cfg['ccpo_'+key])
    os.environ.update(ACG_CCPO_EDGE_W='0',ACG_CCPO_TARGET='return',ACG_CCPO_EP_W='0',ACG_CCPO_DUMP='',ACG_CCPO_GDUMP='')
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    import numpy as np
    import torch
    from ccpo import core_ccpo as core
    from ccpo.future_progress import ccpo_future_progress_advantage,finalize_progress_logging
    # These trusted local snapshots predate the new pickle-free metadata schema.
    with np.load(args.snapshot,allow_pickle=True) as archive:
        old={k:archive[k] for k in archive.files}
    lengths=np.asarray(old['response_lengths'],dtype=int)
    mask=torch.as_tensor(np.arange(max(1,int(lengths.max())))[None,:]<lengths[:,None],dtype=torch.float32)
    kw=dict(step_rewards=torch.tensor(old['target'],dtype=torch.float32),response_mask=mask,
            anchor_obs=old['anchor_obs'],index=old['uid'],traj_index=old['traj_uid'],
            turn_index=old['turn_index'],episode_lengths=old['episode_lengths'],
            episode_rewards=old['episode_rewards'],is_action_valid=old['is_action_valid'],
            phi_feats=torch.tensor(old['history_hidden'],dtype=torch.float32),phi=core.FrozenPhi(),
            gamma=float(cfg['gamma']),return_diag=True)
    args.output.mkdir(parents=True,exist_ok=True)
    def corr(a,b,mask):
        a,b=np.asarray(a)[mask],np.asarray(b)[mask]
        return float(np.corrcoef(a,b)[0,1]) if len(a)>2 and a.std()>1e-12 and b.std()>1e-12 else None
    records=[]
    for horizon in [1,2]:
        directory=args.output/f'h{horizon}';directory.mkdir(exist_ok=True)
        os.environ['ACG_EXP_DIR']=str(directory)
        advantage,diag=ccpo_future_progress_advantage(**kw,horizon=horizon)
        a=diag['progress_payload']['arrays'];live=diag['live_mask'];uid=np.asarray(old['uid']).astype(str)
        np.testing.assert_allclose(a['m5_edge_normalized'],old['edge_diagnostic'],atol=1e-8)
        np.testing.assert_allclose(a['history_adv'],old['history_adv'],atol=1e-5)
        normalize=cfg['adv_mode']=='mean_std_norm'
        if normalize:
            for task in np.unique(uid):
                ids=np.flatnonzero((uid==task)&live)
                if len(ids)>1:
                    ix=torch.as_tensor(ids);v=advantage[ix];advantage[ix]=(v-v.mean())/(v.std()+1e-6)
        finalize_progress_logging(diag,advantage,uid,normalize,1,0.,1.)
        common=np.asarray(old['eligible'],bool)&a['eligible']
        nonterminal=common&~np.asarray(old['terminal'],bool)&~a['terminal']
        metrics={'step':1,'offline_replay':1,**{'ccpo/'+k:float(v) for k,v in diag.items() if k.startswith('progress_') and isinstance(v,(float,int,np.number))}}
        (directory/'outputs/metrics.jsonl').write_text(json.dumps(metrics)+'\n')
        record={'horizon':horizon,'rows':len(a['target']),'common_supported_rows':int(common.sum()),'common_nonterminal_rows':int(nonterminal.sum()),
                'old_fixed_gain_edge_corr':corr(old['future_gain'],a['m5_edge_normalized'],common),
                'new_progress_edge_corr':corr(a['progress_normalized'],a['m5_edge_normalized'],common),
                'old_fixed_gain_edge_nonterminal_corr':corr(old['future_gain'],a['m5_edge_normalized'],nonterminal),
                'new_progress_edge_nonterminal_corr':corr(a['progress_normalized'],a['m5_edge_normalized'],nonterminal),
                'new_all_supported_edge_corr':corr(a['progress_normalized'],a['m5_edge_normalized'],a['eligible']),
                'future_history_absratio':diag['progress_future_history_absratio'],
                'applied_identity_error':diag['progress_applied_identity_error'],
                'history_preserved':True,'edge_reference_preserved':True}
        records.append(record);print(json.dumps(record),flush=True)
    result={'source_snapshot':str(args.snapshot.resolve()),'source_config':str(args.source_config.resolve()),
            'source_sha256':hashlib.sha256(args.snapshot.read_bytes()).hexdigest(),
            'benchmark':'alfworld' if 'alfworld' in str(cfg['env_name']).lower() else 'webshop',
            'source_step':int(args.snapshot.stem.split('-')[-1]),'training_launched':False,'gpu_used':False,
            'interpretation':'Credit similarity on the same archived fixed-anchor-policy rollouts; not training or validation performance of this method.',
            'comparisons':records}
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
