#!/usr/bin/env python3
"""ACG_CCPO_TARGET=score: dense step target for WebShop's all-zero groups.

WebShop's reward reaching the trainer is binary -- verl-agent overwrites the
env's partial score with 10.0 on an exact match and 0 otherwise
(env_package/webshop/envs.py:47-53). Measured on ccpo-attncred-ws-20260914,
69.2% of task groups over steps 1-25 and 30.2% over steps 51+ score zero on
EVERY one of their 8 rollouts. A group-relative estimator computes exactly zero
advantage on every turn of such a group: b_loo, b_obs and B_TASK are all means
of the same zeros, so there is no signal for any baseline to sharpen.

The env's dense score still varies inside those groups (a rollout matching 0.9
of the goal's attributes scores 0.9 while earning 0 reward), and it survives in
info['task_score']. Pointing the STEP target at it restores a gradient there.
The episode term keeps the published binary reward, so the outcome signal the
objective optimises is unchanged.

Test 1: the dense return-to-go is gamma-discounted per trajectory and scaled to
        the binary channel's units.
Test 2: the property the arm exists for -- on a group whose binary returns are
        all zero, target=return gives every turn exactly zero advantage while
        target=score does not; and attncred's own machinery (lam_k, the J=0
        fallback, live coverage) is still the thing computing it.
Test 3: source guard on the two plumbing points, so renaming either cannot leave
        the behavioural tests silently exercising dead code.
"""
import os
import sys
import types

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
# attncred's own configuration, read at import time, so the behavioural test
# exercises the shipped arm rather than a bare estimator.
os.environ["ACG_CCPO_PRIOR_KAPPA"] = "2.0"
os.environ["ACG_CCPO_LAM_FIX"] = "1.0"
os.environ["ACG_CCPO_BACKOFF_TASK"] = "1"

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ccpo import core_ccpo


def test_dense_returns():
    batch = types.SimpleNamespace(non_tensor_batch={
        # two trajectories, terminal score only, as WebShop reports it
        'task_scores': np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        'traj_uid': np.array(['a', 'a', 'a', 'b', 'b', 'b']),
    })
    out = core_ccpo.dense_step_returns(batch, gamma=0.5, scale=10.0)
    # trajectory a: terminal 8.0 -> return-to-go 2.0, 4.0, 8.0; trajectory b: zeros
    assert np.allclose(out[:3], [2.0, 4.0, 8.0]), out[:3]
    assert np.allclose(out[3:], [0.0, 0.0, 0.0]), out[3:]
    # absent key -> None, which is every benchmark but WebShop
    assert core_ccpo.dense_step_returns(
        types.SimpleNamespace(non_tensor_batch={'traj_uid': np.array(['a'])})) is None
    print("test 1 PASS: dense return-to-go discounts per trajectory and scales to reward units")


def _fixture(dense):
    """One task, 4 trajectories, 6 turns each. Binary return is zero everywhere;
    the dense score separates the trajectories."""
    T, n_traj = 6, 4
    obs, uid, traj, G, scores = [], [], [], [], []
    for t in range(n_traj):
        for s in range(T):
            obs.append(f"page {s % 3}")
            uid.append("task0")
            traj.append(f"tr{t}")
            G.append(0.0)                                    # binary: nobody won
            scores.append(0.0 if s < T - 1 else [0.2, 0.5, 0.9, 0.4][t])
    n = len(obs)
    batch = types.SimpleNamespace(non_tensor_batch={
        'task_scores': np.array(scores, dtype=np.float32), 'traj_uid': np.array(traj)})
    kw = dict(
        step_rewards=torch.tensor(G, dtype=torch.float32),
        response_mask=torch.ones(n, 8),
        anchor_obs=np.array(obs), index=np.array(uid), traj_index=np.array(traj),
        phi=core_ccpo.FrozenPhi(), is_action_valid=np.ones(n, bool),
        return_diag=True,
    )
    if dense:
        kw['dense_scores'] = core_ccpo.dense_step_returns(batch, gamma=0.95)
    return kw


def test_zero_group_gets_a_gradient():
    binary, _ = core_ccpo.ccpo_step_advantage(target="return", **_fixture(dense=False))
    dense, diag = core_ccpo.ccpo_step_advantage(target="score", **_fixture(dense=True))
    binary = np.asarray(binary.detach().cpu() if hasattr(binary, 'detach') else binary, dtype=float)
    dense = np.asarray(dense.detach().cpu() if hasattr(dense, 'detach') else dense, dtype=float)
    assert np.allclose(binary, 0.0), f"binary target should be dead here, got |A| {np.abs(binary).max()}"
    assert np.abs(dense).max() > 1e-6, "dense target produced no gradient either"
    # the credit must still be the attncred estimator's, not a raw score
    assert diag['live_frac'] > 0.0
    # the credibility prior must be live: J=3 siblings, kappa=2 -> lam_k = 0.6
    assert 0.0 < diag['lam_k_mean'] < 1.0, f"lam_k {diag['lam_k_mean']} -- prior inactive"
    assert abs(diag['lam_u_mean'] - 1.0) < 1e-9, "lam_fix=1.0 should pin the attention readout"
    print(f"test 2 PASS: all-zero group -> binary |A|max {np.abs(binary).max():.3f}, "
          f"dense |A|max {np.abs(dense).max():.3f} (lam_k {diag['lam_k_mean']:.3f})")


def test_plumbing_present():
    rl = open(os.path.join(ROOT, 'patches/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py')).read()
    assert "batch.non_tensor_batch['task_scores']" in rl, "rollout loop no longer carries the dense score"
    assert "'task_score' in infos[0]" in rl, "dense score must stay guarded: only WebShop sets it"
    rt = open(os.path.join(ROOT, 'patches/verl-agent/verl/trainer/ppo/ray_trainer.py')).read()
    assert "dense_scores=_dense" in rt, "trainer no longer hands the dense target to the estimator"
    assert "core_ccpo.dense_step_returns" in rt, "trainer no longer builds the dense target"
    src = open(os.path.join(ROOT, 'ccpo/core_ccpo.py')).read()
    assert 'if target == "score":' in src and 'dense_scores is None' in src
    print("test 3 PASS: both plumbing points and the loud failure are in the shipped source")


if __name__ == "__main__":
    test_dense_returns()
    test_zero_group_gets_a_gradient()
    test_plumbing_present()
    print("OK")
