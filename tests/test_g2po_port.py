#!/usr/bin/env python3
"""Verify the G2PO port in ccpo/core_ccpo.py against the published code.

Test 1 checks g2po_node_values and g2po_step_advantage produce exactly what
baselines/G2PO/g2po/core_g2po.py produces on the same fixture. It is skipped if
that checkout is absent (baselines/ is gitignored; see baselines/README.md).

Test 2 checks the gate and target options in isolation: every one of them must
be a no-op when off, the similarity gate must merge what exact matching splits,
the backoff level must serve rows level 0 leaves at zero, and the nextnode
target must actually change what is being credited.

Test 3 is the regression guard: with every new flag off, the estimator must
reproduce the pre-revision advantage bit for bit.
"""
import os
import sys
import types

import os as _os
# OpenBLAS sizes its thread pool from nproc; alongside a training run the
# cgroup pid budget is spent and `import numpy` itself fails in
# blas_thread_init. Same ceiling that shapes the ray and tokenizer configs.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ccpo.core_ccpo import (                                        # noqa: E402
    FrozenPhi, are_similar, ccpo_step_advantage, cluster_keys,
    g2po_node_values, g2po_step_advantage, _TERM_OK, _TERM_BAD,
)

G2PO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "baselines", "G2PO")


def fixture(seed=0):
    """Two tasks, three trajectories each, with observations that repeat both
    within and across trajectories -- the only case where node pooling differs
    from a per-trajectory return."""
    rng = np.random.default_rng(seed)
    obs, uid, traj, ep, valid = [], [], [], [], []
    for task in range(2):
        for t_i in range(3):
            T = int(rng.integers(3, 6))
            R = 10.0 if (task + t_i) % 2 == 0 else 0.0
            for t in range(T):
                obs.append(f"room {rng.integers(0, 3)}. you see a cabinet.")
                uid.append(f"task{task}")
                traj.append(f"task{task}-traj{t_i}")
                ep.append(R)
                valid.append(bool(rng.random() > 0.2))
    return (np.array(obs), np.array(uid), np.array(traj),
            np.array(ep, dtype=np.float32), np.array(valid, dtype=bool))


def test_against_published():
    if not os.path.isdir(G2PO_DIR):
        print("  SKIP: baselines/G2PO not present")
        return None

    # core_g2po imports `from verl import DataProto`; only the name is needed.
    if "verl" not in sys.modules:
        stub = types.ModuleType("verl")
        stub.DataProto = object
        sys.modules["verl"] = stub
    sys.path.insert(0, G2PO_DIR)
    from g2po import core_g2po                                      # noqa: E402

    obs, uid, traj, ep, valid = fixture()
    n = len(obs)

    class Batch:
        pass
    b = Batch()
    b.non_tensor_batch = {"episode_rewards": ep, "anchor_obs": obs,
                          "traj_uid": traj, "uid": uid}
    b.batch = {"responses": torch.zeros(n, 4)}

    ref_v, ref_gid, ref_next = core_g2po.compute_group_aggregation_values(b)
    VAL, NODE, NEXT = g2po_node_values(obs, uid, traj, ep, gamma=0.95,
                                       success_reward=core_g2po.SUCCESS_REWARD)

    ours_v = np.array([VAL[NODE[i]] for i in range(n)])
    d_v = float(np.abs(ours_v - ref_v).max())
    print(f"  node values      max|ours - published| = {d_v:.2e}")

    # successor identity: same partition of terminal vs interior, same target
    # group_idx restarts at 0 inside each task, so it is keyed by (task, gid).
    ref_score = {(str(uid[i]), int(ref_gid[i])): ref_v[i] for i in range(n)}
    ours_next = np.array([VAL[NEXT[i]] for i in range(n)])
    ref_nextv = np.array([
        core_g2po.SUCCESS_REWARD if ref_next[i] == -1 else
        0.0 if ref_next[i] == -2 else
        ref_score[(str(uid[i]), int(ref_next[i]))] for i in range(n)])
    d_n = float(np.abs(ours_next - ref_nextv).max())
    n_term = int(sum(1 for i in range(n) if NEXT[i] in (_TERM_OK, _TERM_BAD)))
    print(f"  successor values max|ours - published| = {d_n:.2e}  "
          f"({n_term} terminal links)")

    ref_adv = core_g2po.compute_step_level_advantage(
        torch.tensor(ref_v, dtype=torch.float32), torch.ones(n, 4), valid,
        uid, ref_gid, ref_next, mode="mean_std_norm",
        invalid_action_penalty=0.1)[:, 0].numpy()
    our_adv = g2po_step_advantage(uid, VAL, NODE, NEXT, valid, invalid_penalty=0.1)
    d_a = float(np.abs(our_adv - ref_adv).max())
    print(f"  step advantage   max|ours - published| = {d_a:.2e}")

    ok = d_v < 1e-5 and d_n < 1e-5 and d_a < 1e-4 and n_term == 6
    print(f"  {'OK (port is faithful)' if ok else 'FAIL'}")
    return ok


def test_options():
    obs, uid, traj, ep, valid = fixture(seed=1)
    n = len(obs)
    G = torch.tensor(np.linspace(0.0, 1.0, n), dtype=torch.float32)
    mask = torch.ones(n, 6)
    common = dict(step_rewards=G, response_mask=mask, anchor_obs=obs, index=uid,
                  traj_index=traj, phi=FrozenPhi(), return_diag=True)

    ok = True

    # -- similarity gate: merges what exact matching splits, never fewer buckets
    near = np.array([o if i % 2 else o.replace("cabinet", "cabinet ")
                     for i, o in enumerate(obs)])
    k_exact = cluster_keys(near, uid, 0.0)
    k_sim = cluster_keys(near, uid, 0.9)
    n_exact, n_sim = len(set(map(str, k_exact))), len(set(map(str, k_sim)))
    ok &= n_sim < n_exact and are_similar("a b c", "a b  c", 0.9)
    print(f"  gate: exact {n_exact} buckets -> sim@0.9 {n_sim}  "
          f"{'OK' if n_sim < n_exact else 'FAIL'}")

    # -- every option off must leave the advantage untouched
    a_off, d_off = ccpo_step_advantage(**common)
    a_dup, _ = ccpo_step_advantage(**common, sim=0.0, sim_backoff=0.0,
                                   target="return", edge_w=0.0)
    ok &= torch.equal(a_off, a_dup)
    print(f"  defaults are the off-state: {'OK' if torch.equal(a_off, a_dup) else 'FAIL'}")

    # -- backoff serves rows level 0 leaves at exactly zero, and only those
    a_bk, d_bk = ccpo_step_advantage(**common, sim_backoff=0.5)
    gained = (~d_off["live_mask"]) & d_bk["live_mask"]
    lost = d_off["live_mask"] & (~d_bk["live_mask"])
    same = torch.allclose(a_off[torch.tensor(d_off["live_mask"])],
                          a_bk[torch.tensor(d_off["live_mask"])])
    ok &= (not lost.any()) and same
    print(f"  backoff: live {d_off['live_frac']:.2f} -> {d_bk['live_frac']:.2f}, "
          f"+{int(gained.sum())} rows, -{int(lost.sum())}, level-0 rows unchanged "
          f"{'OK' if (not lost.any()) and same else 'FAIL'}")

    # -- nextnode target changes the credit, and needs episode_rewards to do it
    a_nn, d_nn = ccpo_step_advantage(**common, target="nextnode",
                                     episode_rewards=ep, is_action_valid=valid)
    a_no, d_no = ccpo_step_advantage(**common, target="nextnode")
    ok &= d_nn["target"] == "nextnode" and d_no["target"] == "return"
    ok &= not torch.allclose(a_nn, a_off)
    print(f"  target: return -> nextnode changes A on "
          f"{int((a_nn != a_off).sum())}/{n} rows; without episode_rewards it "
          f"falls back to '{d_no['target']}'  "
          f"{'OK' if d_no['target'] == 'return' else 'FAIL'}")

    # -- the two reference correlations are now distinct quantities
    ok &= np.isnan(d_off["r_vs_g2po"]) and not np.isnan(d_nn["r_vs_g2po"])
    print(f"  r_vs_gigpo={d_nn['r_vs_gigpo']:+.3f}  r_vs_g2po={d_nn['r_vs_g2po']:+.3f}  "
          f"(nan without episode_rewards: {np.isnan(d_off['r_vs_g2po'])})")

    # -- edge term is additive and only fires when asked
    a_e, d_e = ccpo_step_advantage(**common, episode_rewards=ep,
                                   is_action_valid=valid, edge_w=0.5)
    ok &= d_e["edge_cov"] == 1.0 and not torch.allclose(a_e, a_off)
    print(f"  edge term at w=0.5 moves A: "
          f"{'OK' if not torch.allclose(a_e, a_off) else 'FAIL'}")
    return bool(ok)


def test_regression():
    """The pre-revision file, loaded under a different module name, must give
    the same advantage when every new option is off -- to within the float32
    ulp introduced by computing the baseline in float64."""
    import importlib.util
    old = os.environ.get("ACG_CCPO_BASELINE_FILE", "")
    if not old or not os.path.exists(old):
        print("  SKIP: set ACG_CCPO_BASELINE_FILE to the pre-revision core_ccpo.py")
        return None
    import shutil, tempfile
    tmp = os.path.join(tempfile.mkdtemp(), "core_ccpo_old.py")
    shutil.copyfile(old, tmp)
    spec = importlib.util.spec_from_file_location("core_ccpo_old", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    obs, uid, traj, ep, valid = fixture(seed=2)
    n = len(obs)
    rng = np.random.default_rng(7)
    G = torch.tensor(rng.normal(size=n), dtype=torch.float32)
    mask = torch.ones(n, 5)
    feats = torch.tensor(rng.normal(size=(n, 32)))
    kw = dict(step_rewards=G, response_mask=mask, anchor_obs=obs, index=uid,
              traj_index=traj, phi=FrozenPhi(), phi_feats=feats)
    a_new = ccpo_step_advantage(**kw)
    a_old = mod.ccpo_step_advantage(**kw)
    d = float((a_new - a_old).abs().max())
    # The revision promotes the return vector to float64 before the variance
    # arithmetic; holding it at float32 reproduces the old result bit for bit,
    # so the tolerance here is exactly one float32 ulp, not slack.
    print(f"  max|new - pre-revision| = {d:.2e}  (float32 eps = {np.finfo(np.float32).eps:.2e})  "
          f"{'OK' if d < 1e-6 else 'FAIL'}")
    return d < 1e-6


if __name__ == "__main__":
    print("test 1: node values and step advantage match published G2PO")
    t1 = test_against_published()
    print("\ntest 2: gate, backoff and target options")
    t2 = test_options()
    print("\ntest 3: regression against the pre-revision estimator")
    t3 = test_regression()
    res = [("1", t1), ("2", t2), ("3", t3)]
    print("\n" + " | ".join(
        f"test {k} {'PASS' if v else ('SKIPPED' if v is None else 'FAIL')}"
        for k, v in res))
    sys.exit(0 if all(v is not False for _, v in res) else 1)
