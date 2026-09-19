#!/usr/bin/env python3
"""Verify the hidden-only configuration on an existing M11 batch, on CPU."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CONTROL = ROOT / 'experiments/m11-ccpo-attncred-ctxadv-future-progress-webshop-1.5b-2gpu-20260918'
cfg = json.loads((HERE / 'config.json').read_text())
os.environ.update({k: str(v) for k, v in cfg['env'].items() if k.startswith('ACG_CCPO_')})
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1', ACG_CCPO_DUMP='', ACG_CCPO_GDUMP='')
os.environ.pop('ACG_EXP_DIR', None)
sys.path[:0] = [str(ROOT), str(ROOT / 'verl-agent')]
import numpy as np
import torch
from ccpo import core_ccpo as core
from ccpo.future_progress import ccpo_future_progress_advantage, finalize_progress_logging


def main():
    torch.set_num_threads(1)
    source = CONTROL / 'outputs/future_progress/step-0100.npz'
    keys = ('target response_lengths anchor_obs uid traj_uid turn_index episode_lengths '
            'episode_rewards is_action_valid history_hidden history_adv current_value raw_progress current_phi').split()
    with np.load(source, allow_pickle=False) as z:
        old = {k: z[k] for k in keys}
    lengths = old['response_lengths'].astype(int)
    kw = dict(step_rewards=torch.as_tensor(old['target'], dtype=torch.float32),
              response_mask=torch.as_tensor(np.arange(int(lengths.max()))[None, :] < lengths[:, None], dtype=torch.float32),
              anchor_obs=old['anchor_obs'], index=old['uid'], traj_index=old['traj_uid'],
              turn_index=old['turn_index'], episode_lengths=old['episode_lengths'],
              episode_rewards=old['episode_rewards'], is_action_valid=old['is_action_valid'],
              phi_feats=torch.as_tensor(old['history_hidden'], dtype=torch.float32),
              phi=core.FrozenPhi(), gamma=.95, return_diag=True, dump_enabled=False)
    assert core._PHI_MODE == 'hidden+ctx' and core._CTX_W == 0
    core._CTX_W = 1.
    _, baseline = ccpo_future_progress_advantage(**kw)
    baseline_arrays = baseline['progress_payload']['arrays']
    parity = {}
    for key in ['history_adv', 'current_value', 'raw_progress', 'current_phi']:
        parity[key] = float(np.max(abs(baseline_arrays[key] - old[key])))
        np.testing.assert_allclose(baseline_arrays[key], old[key], atol=2e-6, rtol=1e-7)

    core._CTX_W = 0.
    advantage, noctx = ccpo_future_progress_advantage(**kw)
    a = noctx['progress_payload']['arrays']
    expected = core.whiten_feats(old['history_hidden'])
    assert a['current_phi'].shape[1] == 1536
    assert baseline_arrays['current_phi'].shape[1] == 1573
    np.testing.assert_allclose(a['current_phi'], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(a['future_phi'][~a['terminal']], expected[a['endpoint_index'][~a['terminal']]], atol=1e-12)
    changed_context = [{'t': 30, 'n_unique': 25, 'revisit': 1, 'progress': 1.} for _ in old['uid']]
    changed, diagnostics = ccpo_future_progress_advantage(**dict(kw, ctx_override=changed_context))
    torch.testing.assert_close(advantage, changed, rtol=0, atol=0)
    for key in ['history_adv', 'current_value', 'future_value', 'raw_progress', 'current_phi']:
        np.testing.assert_array_equal(a[key], diagnostics['progress_payload']['arrays'][key])
    assert not advantage.requires_grad
    finalize_progress_logging(noctx, advantage, old['uid'], normalize=False, step_tag=100, episode_weight=0.)

    config_old = json.loads((CONTROL / 'config.json').read_text())['config']
    config_new = cfg['config']
    delta = {k: {'control': config_old.get(k), 'variant': config_new.get(k)}
             for k in set(config_old) | set(config_new) if config_old.get(k) != config_new.get(k)}
    assert set(delta) == {'ccpo_ctx_w', 'gpus', 'exp_id'}
    result = dict(status='passed', source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  rows=len(old['uid']), control_replay_errors=parity, hidden_dimensions=1536,
                  control_similarity_dimensions=1573, variant_similarity_dimensions=1536,
                  both_endpoints_use_hidden_only=True, context_perturbation_changes_credit=False,
                  detached_advantage=True, applied_identity_error=noctx['progress_applied_identity_error'],
                  config_delta=delta, gpu_used=False, environment_rollouts_started=False,
                  code_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                               for p in ['ccpo/core_ccpo.py', 'ccpo/future_progress.py']})
    (HERE / 'VALIDATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
