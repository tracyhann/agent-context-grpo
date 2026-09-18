"""Fixed-O_t gain from a history-plus-future value readout.

A = (Y - B_history) + gain_weight * (B_joint - B_history).
The extra readout uses the same exact anchor peers and anchor-time returns.
Unsupported exact anchors receive zero gain, preserving historical task backoff.
"""
from collections import defaultdict
import csv
import gzip
import json
import os
from pathlib import Path
import numpy as np
import torch
from ccpo import core_ccpo as core
from ccpo.outlook import canonical_trajectory_rows, _take

CTX_KEYS = ('t', 'n_unique', 'revisit', 'progress')


def _norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def context_block(context):
    phi = core.FrozenPhi()
    return _norm(np.array([np.r_[phi._thermo(c['t'],0,30),
        phi._thermo(c['n_unique'],0,25),phi._thermo(c['progress'],0,1),
        float(c['revisit'] > 0)] for c in context]))


def future_records(non_tensor, horizon=2):
    """Canonical complete trajectories, including the observed terminal endpoint."""
    if int(horizon) != horizon or horizon < 1:
        raise ValueError('Fixed-anchor horizon must be a positive integer')
    take, restore, groups = canonical_trajectory_rows(non_tensor['uid'],non_tensor['traj_uid'],
        non_tensor['ccpo_turn_index'],non_tensor['episode_lengths'])
    keys=('uid','traj_uid','anchor_obs','ccpo_prompt_text','ccpo_action','ccpo_next_obs')
    a={k:np.asarray(non_tensor[k])[take] for k in keys}
    for key in keys:
        if not np.array_equal(np.asarray(non_tensor[key]),a[key][restore]):
            raise ValueError('Inconsistent padded future metadata: '+key)
    ctx=core.derive_context(a['anchor_obs'],a['uid'],a['traj_uid'])
    result=[None]*len(take)
    for ids in groups:
        # A nonterminal next observation must be the subsequent turn's anchor.
        for p,i in enumerate(ids[:-1]):
            if str(a['ccpo_next_obs'][i]) != str(a['anchor_obs'][ids[p+1]]):
                raise ValueError('Broken future observation chain')
        seen=set(map(str,a['anchor_obs'][ids])); last=str(a['ccpo_next_obs'][ids[-1]])
        terminal_context=dict(t=len(ids),n_unique=len(seen|{last}),revisit=int(last in seen),
                              progress=len(seen|{last})/(len(ids)+1))
        for t,i in enumerate(ids):
            end=min(t+horizon,len(ids)); endpoint=ctx[ids[end]] if end<len(ids) else terminal_context
            transitions=[dict(action=str(a['ccpo_action'][j]),next_observation=str(a['ccpo_next_obs'][j])) for j in ids[t:end]]
            result[i]=dict(uid=str(a['uid'][i]),traj_uid=str(a['traj_uid'][i]),turn=t,
                anchor=str(a['anchor_obs'][i]),history_text=str(a['ccpo_prompt_text'][i]),
                transitions=transitions,terminal=end==len(ids),window_length=end-t,
                history_context=ctx[i],future_context=endpoint)
    return take,restore,groups,result


def tokenize_future(record, tokenizer, max_tokens):
    """Retain history and the full local future; budget both blocks if necessary."""
    header='\n\nObserved continuation for credit estimation:\n'
    future='\n'.join(f"Step {i+1} action: {s['action']}\nNext observation: {s['next_observation']}"
                     for i,s in enumerate(record['transitions']))
    if record['terminal']:
        future+='\nRecorded trajectory ends here.'
    history=record['history_text']
    def encode(h,f):
        return tokenizer.apply_chat_template([dict(role='user',content=h+header+f)],
            add_generation_prompt=True,tokenize=True,enable_thinking=False)
    ids=encode(history,future); original=len(ids); truncated=False
    if len(ids)>max_tokens:
        # Preserve the beginning (task) and end (current observation) of history,
        # and both ends of the future block. Record exact encoded tokens below.
        hs=tokenizer.encode(history,add_special_tokens=False); fs=tokenizer.encode(future,add_special_tokens=False)
        budget=max_tokens-256
        while len(ids)>max_tokens and budget>128:
            hb=budget//2;fb=budget-hb
            def trim(x,n):return x if len(x)<=n else x[:n//2]+x[-(n-n//2):]
            ids=encode(tokenizer.decode(trim(hs,hb),skip_special_tokens=False),
                       tokenizer.decode(trim(fs,fb),skip_special_tokens=False))
            budget-=128;truncated=True
        if len(ids)>max_tokens:raise ValueError('Cannot fit joint future context without unbounded truncation')
    return list(ids),original,truncated


def capture_future_features(batch, tokenizer, reference_workers, horizon=2, max_tokens=8192):
    """Extra frozen-reference pass only; its dummy log probabilities are discarded."""
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
    take,restore,groups,records=future_records(batch.non_tensor_batch,horizon)
    encoded=[]
    for record in records:
        ids,original,truncated=tokenize_future(record,tokenizer,max_tokens)
        record.update(input_ids=ids,original_tokens=original,truncated=truncated)
        encoded.append(ids)
    n=len(encoded); width=max(map(len,encoded))
    prompts=torch.full((n,width),tokenizer.pad_token_id,dtype=torch.long)
    attention=torch.zeros((n,width+1),dtype=torch.long)
    for i,ids in enumerate(encoded):
        prompts[i,-len(ids):]=torch.tensor(ids);attention[i,width-len(ids):]=1
    responses=torch.full((n,1),tokenizer.eos_token_id,dtype=torch.long)
    position=(attention.cumsum(-1)-1).clamp_min(0)*attention
    proto=DataProto.from_dict(tensors=dict(prompts=prompts,responses=responses,
        input_ids=torch.cat([prompts,responses],-1),attention_mask=attention,position_ids=position))
    padded,padding=pad_dataproto_to_divisor(proto,reference_workers.world_size)
    captured=unpad_dataproto(reference_workers.compute_ref_log_prob(padded),padding)
    if 'ccpo_phi_feats' not in captured.batch:
        raise RuntimeError('Missing fixed-anchor future reference features')
    features=captured.batch['ccpo_phi_feats'].detach().cpu()
    if len(features)!=n or not torch.isfinite(features).all():
        raise RuntimeError('Nonfinite or misaligned future reference features')
    batch.batch['ccpo_future_phi_feats']=features[torch.as_tensor(restore)]
    batch.non_tensor_batch['ccpo_future_context']=np.array([[r['future_context'][k] for k in CTX_KEYS] for r in records])[restore]
    batch.meta_info['ccpo_future_metadata']=records


def _corr(x,y,mask):
    x,y=np.asarray(x)[mask],np.asarray(y)[mask]
    return float(np.corrcoef(x,y)[0,1]) if len(x)>2 and x.std()>1e-12 and y.std()>1e-12 else float('nan')


def _stats(diag,name,values,mask):
    x=np.asarray(values)[mask]; x=x[np.isfinite(x)]
    for suffix,fn in [('mean',np.mean),('std',np.std),('absmean',lambda a:np.mean(abs(a))),
                      ('min',np.min),('max',np.max)]:
        diag['fixed_'+name+'_'+suffix]=float(fn(x)) if len(x) else float('nan')


def ccpo_fixed_anchor_advantage(*,turn_index,episode_lengths,immediate_rewards,
        future_phi_feats,future_context,prompt_metadata=None,horizon=2,gain_weight=1.,**kwargs):
    edge=core._EDGE_W if kwargs.get('edge_w') is None else float(kwargs['edge_w'])
    if edge!=0 or (kwargs.get('target') or core._TARGET)!='return' or core._STD_MODE=='local' or core._JW_C!=0:
        raise ValueError('Fixed-anchor gain requires return targets, no edge, no local scaling/J weighting')
    if core._PHI_MODE!='hidden+ctx' or kwargs.get('phi_feats') is None:
        raise ValueError('Fixed-anchor gain requires frozen reference features')
    if horizon!=2 or not np.isfinite(gain_weight) or gain_weight<0:
        raise ValueError('Fixed-anchor arm requires horizon 2 and a finite nonnegative gain weight')
    take,restore,groups=canonical_trajectory_rows(kwargs['index'],kwargs['traj_index'],turn_index,episode_lengths)
    fields=('step_rewards','response_mask','anchor_obs','index','traj_index','phi_feats',
            'episode_rewards','is_action_valid','aff_labels','ctx_override')
    canonical=dict(kwargs)
    for k in fields:
        if k in canonical:canonical[k]=_take(canonical[k],take)
    # Features, observations and targets must survive duplicate padding unchanged.
    for k in ('step_rewards','anchor_obs','index','traj_index','phi_feats','episode_rewards','is_action_valid'):
        v=kwargs.get(k)
        if v is not None:
            orig=v.detach().cpu().numpy() if torch.is_tensor(v) else np.asarray(v)
            cv=canonical[k].detach().cpu().numpy() if torch.is_tensor(canonical[k]) else np.asarray(canonical[k])
            if not np.array_equal(orig,cv[restore]):raise ValueError('Inconsistent padded fixed-anchor field: '+k)
    joint=_take(future_phi_feats,take)
    joint=joint.detach().float().cpu().numpy() if torch.is_tensor(joint) else np.asarray(joint,float)
    ctx_future=[dict(zip(CTX_KEYS,v)) for v in np.asarray(future_context)[take]]
    ctx_history=core.derive_context(canonical['anchor_obs'],canonical['index'],canonical['traj_index'])
    combined_ctx=_norm(np.concatenate([context_block(ctx_history),context_block(ctx_future)],axis=1))
    joint_phi=_norm(np.concatenate([core.whiten_feats(joint),core._CTX_W*combined_ctx],axis=1))
    hist_tensor,diag=core.ccpo_step_advantage(**dict(canonical,return_diag=True))
    _,plus=core.ccpo_step_advantage(**dict(canonical,prepared_phi=joint_phi,exact_only=True,
                                         dump_enabled=False,return_diag=True))
    live=np.asarray(diag['live_mask']); eligible=np.asarray(plus['live_mask'])
    if np.any(eligible & (diag['level_values']!=0)):raise ValueError('Future estimator changed anchor peers')
    for key in ('support_values','credibility_values','task_prior_values'):
        if not np.allclose(diag[key][eligible],plus[key][eligible],atol=1e-8):
            raise ValueError('Fixed-anchor readouts disagree on '+key)
    scores=canonical['step_rewards']
    if scores.dim()>1:scores=(scores*canonical['response_mask']).sum(-1)
    target=scores.detach().float().cpu().numpy().astype(float)
    raw=np.zeros(len(take)); rewards=np.asarray(immediate_rewards,float)[take]
    gamma=float(kwargs.get('gamma',.95))
    for ids in groups:
        running=0.
        for i in ids[::-1]:running=rewards[i]+gamma*running;raw[i]=running
    history=hist_tensor.detach().float().cpu().numpy().astype(float)
    bh=diag['baseline_values']; measured_future=plus['baseline_values']
    bf=np.where(eligible,measured_future,bh)
    gain=np.where(eligible,bf-bh,0.)
    residual=np.where(live,target-bf,0.)
    mixed=history+gain_weight*gain
    identity=float(np.max(abs((history-gain-residual)[live]))) if live.any() else 0.
    if identity>1e-5 or not np.isfinite(mixed).all():raise ValueError('Fixed-anchor credit identity failed')
    terminal=np.asarray(turn_index)[take]+horizon>=np.asarray(episode_lengths)[take]
    task=np.asarray(canonical['index']).astype(str)
    vals,nodes,successors=core.g2po_node_values(canonical['anchor_obs'],task,canonical['traj_index'],canonical['episode_rewards'],gamma,10.)
    raw_edge=core._successor_values(vals,successors,len(take))-np.array([vals[k] for k in nodes])
    edge_adv=np.zeros(len(take))
    for t in np.unique(task):
        ids=np.flatnonzero(task==t);g=raw_edge[ids]
        if len(ids)>1:edge_adv[ids]=(g-g.mean())/(g.std(ddof=1)+1e-6)
    diag.update(fixed_enabled=1.,fixed_horizon=float(horizon),fixed_gain_weight=float(gain_weight),
        fixed_history_weight=1.,fixed_episode_weight=0.,fixed_edge_weight=0.,
        fixed_eligible_frac=float(eligible.mean()),fixed_no_peer_frac=float((~eligible).mean()),
        fixed_terminal_frac=float(terminal.mean()),fixed_unique_rows=float(len(take)),
        fixed_padding_frac=float(1-len(take)/len(restore)),fixed_identity_error=identity)
    arrays=dict(target=target,raw_return=raw,local_adjustment=target-raw,history_baseline=bh,
        measured_future_baseline=measured_future,
        future_baseline=bf,history_adv=history,future_gain=gain,future_residual=residual,
        weighted_gain=gain_weight*gain,combined_pre=mixed,edge_diagnostic=edge_adv)
    for name,x in arrays.items():_stats(diag,name,x,live)
    for name,src,mask in [('history',diag,live),('future',plus,eligible)]:
        for label,key in [('kernel','kernel_values'),('uniform','uniform_values'),('task_prior','task_prior_values'),
                          ('J','support_values'),('n_eff','effective_support_values'),('lambda_k','credibility_values')]:
            arr=np.asarray(src[key]);arrays[name+'_'+label]=arr.copy();_stats(diag,name+'_'+label,arr,mask)
    for label,x,y in [('gain_history',gain,history),('gain_edge',gain,edge_adv),
                       ('history_edge',history,edge_adv),('baselines',bh,bf),('combined_edge',mixed,edge_adv)]:
        diag['fixed_'+label+'_corr']=_corr(x,y,eligible)
        diag['fixed_'+label+'_nonterminal_corr']=_corr(x,y,eligible & ~terminal)
    diag['fixed_future_history_absratio']=float(np.mean(abs(gain[live]))/max(np.mean(abs(history[live])),1e-12))
    if prompt_metadata is not None:
        if len(prompt_metadata)!=len(take):raise ValueError('Future prompt metadata count mismatch')
        for i,r in enumerate(prompt_metadata):
            if (r['uid'],r['traj_uid'],r['turn'])!=(task[i],str(canonical['traj_index'][i]),int(np.asarray(turn_index)[take][i])):
                raise ValueError('Future prompt metadata order mismatch')
        diag['fixed_future_prompt_truncated_frac']=float(np.mean([r['truncated'] for r in prompt_metadata]))
        _stats(diag,'future_prompt_tokens',np.array([len(r['input_ids']) for r in prompt_metadata]),live)
    # Every scalar component and both raw/processed feature sets remain available.
    history_hidden=canonical['phi_feats']
    history_hidden=history_hidden.detach().float().cpu().numpy() if torch.is_tensor(history_hidden) else np.asarray(history_hidden)
    arrays.update(uid=task,traj_uid=np.asarray(canonical['traj_index']).astype(str),
        turn_index=np.asarray(turn_index)[take],episode_lengths=np.asarray(episode_lengths)[take],
        anchor_obs=np.asarray(canonical['anchor_obs']).astype(str),eligible=eligible,live=live,
        terminal=terminal,immediate_rewards=rewards,episode_rewards=np.asarray(canonical['episode_rewards']),
        is_action_valid=np.asarray(canonical['is_action_valid']),
        response_lengths=canonical['response_mask'].sum(-1).detach().cpu().numpy(),history_hidden=history_hidden,
        future_hidden=joint,history_phi=diag['prepared_phi_values'],future_phi=joint_phi,
        history_context=np.array([[r[k] for k in CTX_KEYS] for r in ctx_history]),
        future_context=np.array([[r[k] for k in CTX_KEYS] for r in ctx_future]))
    diag['fixed_log_payload']=dict(arrays=arrays,take=take,restore=restore,prompts=prompt_metadata)
    diag['components_history']=history[restore]
    diag['components_gain']=(gain_weight*gain)[restore]
    diag['components_mixed']=mixed[restore]
    for key in ('live_mask','baseline_values','bootstrap_values'):
        diag[key]=diag[key][restore]
    diag['history_adv_absmean']=float(np.mean(abs(history[live]))) if live.any() else 0.
    reference=np.zeros(len(take));buckets=defaultdict(list)
    for i,(q,o) in enumerate(zip(task,canonical['anchor_obs'])):buckets[(q,str(o))].append(i)
    for ids in buckets.values():
        if len(ids)>1:reference[ids]=target[ids]-target[ids].mean()
    diag['r_vs_gigpo']=_corr(mixed,reference,live)
    diag['r_vs_g2po']=_corr(mixed,core.g2po_step_advantage(task,vals,nodes,successors,canonical['is_action_valid']),live)
    diag['acc_len_corr']=_corr(mixed,canonical['response_mask'].sum(-1).detach().cpu().numpy(),live)
    diag['effect_rel']=diag['effect_mean']/max(np.mean(abs(mixed[live])),1e-12)
    result=torch.as_tensor(mixed[restore],dtype=hist_tensor.dtype,device=hist_tensor.device)
    return (result,diag) if kwargs.get('return_diag',False) else result


def finalize_fixed_logging(diag, step_adv, uid, normalize, step_tag, episode_weight):
    """Record exact actor components after the same combined normalization."""
    if episode_weight!=0:raise ValueError('Fixed-anchor run must keep episode advantage diagnostic-only')
    h=np.asarray(diag['components_history']).copy(); f=np.asarray(diag['components_gain']).copy()
    mixed=np.asarray(diag['components_mixed']); live=np.asarray(diag['live_mask']);uid=np.asarray(uid).astype(str)
    if normalize:
        for task in np.unique(uid):
            ids=np.flatnonzero((uid==task)&live)
            if len(ids)>1:
                # Match the trainer's float32 sample standard deviation.
                scale=float(torch.as_tensor(mixed[ids],dtype=step_adv.dtype).std())+1e-6
                h[ids]=(h[ids]-h[ids].mean())/scale
                f[ids]=(f[ids]-f[ids].mean())/scale
    applied=step_adv.detach().float().cpu().numpy()
    error=float(np.max(abs(h+f-applied)))
    if error>3e-5:raise ValueError('Applied fixed-anchor components do not sum to actor advantage')
    diag['fixed_applied_identity_error']=error
    for name,x in [('history_applied',h),('gain_applied',f),('combined_applied',applied)]:
        _stats(diag,name,x,live)
    payload=diag['fixed_log_payload'];take=payload['take'];arrays=payload['arrays']
    arrays.update(history_applied=h[take],gain_applied=f[take],combined_applied=applied[take])
    directory=os.environ.get('ACG_EXP_DIR')
    if not directory:return
    out=Path(directory)/'outputs/fixed_anchor';out.mkdir(parents=True,exist_ok=True)
    stem=f'step-{int(step_tag):04d}'
    scalars={k:float(v) for k,v in diag.items() if k.startswith('fixed_') and isinstance(v,(float,int,np.number))}
    (out/(stem+'.metrics.json')).write_text(json.dumps(scalars,indent=2)+'\n')
    columns=[k for k,v in arrays.items() if np.asarray(v).ndim==1 and k not in ['anchor_obs']]
    csvpath=out/(stem+'.csv');temp=csvpath.with_suffix('.csv.tmp')
    with temp.open('w') as fh:
        writer=csv.writer(fh);writer.writerow(columns)
        for i in range(len(take)):writer.writerow([arrays[k][i] for k in columns])
    temp.replace(csvpath)
    every=int(os.environ.get('ACG_CCPO_FIXED_SNAPSHOT_EVERY','1'))
    if every<1:raise ValueError('Feature snapshots must be enabled')
    if int(step_tag)==1 or int(step_tag)%every==0:
        path=out/(stem+'.npz');temp=path.with_suffix('.npz.tmp')
        with temp.open('wb') as fh:np.savez_compressed(fh,**arrays)
        temp.replace(path)
        if payload['prompts'] is not None:
            path=out/(stem+'.prompts.jsonl.gz');temp=path.with_suffix('.gz.tmp')
            with gzip.open(temp,'wt',encoding='utf-8') as fh:
                for row in payload['prompts']:fh.write(json.dumps(row,ensure_ascii=False)+'\n')
            temp.replace(path)
