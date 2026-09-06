#!/usr/bin/env python3
"""phi must be a function of BOTH observation and context.

Inside a bucket the observation is constant by construction, so an
observation-only phi provably cannot split anything -- the design calls this E4.
The shipped `hidden` mode does not satisfy it as well as it looks: the reference
policy sees only what the prompt carries, which is `step_count` plus the most
recent `history_length` (2) turns. It can separate "step 5 from step 15" and not
"has this agent already searched here twice". n_unique, revisit and progress
summarise the whole episode and appear nowhere in the prompt unless compaction is
on.

The fixture has the structure the method exists for: siblings reach the same
observation at different points in their episode, and the return depends on WHEN
(early arrivals are on a direct route and succeed; late ones wandered and fail).
The hidden state is pure noise, so only a phi carrying trajectory context can see
the split. A phi that scores no better than the noise-only mode here would not
help on the real benchmark either.
"""
import importlib
import os
import sys

import os as _os
# OpenBLAS sizes its thread pool from nproc; alongside a training run the
# cgroup pid budget is spent and `import numpy` itself fails in
# blas_thread_init. Same ceiling that shapes the ray and tokenizer configs.
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fixture(n_traj=64, T=24, d=256, seed=0):
    rng = np.random.default_rng(seed)
    obs, uid, traj, ep, G = [], [], [], [], []
    for t in range(n_traj):
        early = (t % 2 == 0)
        for s in range(T):
            obs.append("start" if s == 0 else f"You open cabinet {s % 6}. you see a plate.")
            uid.append(f"task{t // 8}")
            traj.append(f"tr{t}")
            ep.append(10.0 if early else 0.0)
            G.append((10.0 if early else 0.0) * (0.95 ** (T - s)) + rng.normal() * 0.3)
    n = len(obs)
    return dict(
        step_rewards=torch.tensor(G, dtype=torch.float32),
        response_mask=torch.ones(n, 64),
        anchor_obs=np.array(obs), index=np.array(uid), traj_index=np.array(traj),
        phi_feats=torch.tensor(rng.normal(size=(n, d)).astype(np.float32)),
        episode_rewards=np.array(ep, dtype=np.float32),
        is_action_valid=np.ones(n, bool),
        shrink="eb_hier", return_diag=True,
    )


def main():
    kw = fixture()
    out = {}
    for mode in ("hidden", "hidden+ctx", "bow"):
        os.environ["ACG_CCPO_PHI"] = mode
        import ccpo.core_ccpo as m
        importlib.reload(m)
        _, dg = m.ccpo_step_advantage(phi=m.FrozenPhi(), **kw)
        out[mode] = dg
        print(f"  {mode:12s} phi={dg['phi_mode']:11s} lam={dg['lam_u_mean']:.4f} "
              f"effect_rel={dg['effect_rel']:.4f}")

    ok = (out["hidden+ctx"]["effect_rel"] > 5 * out["hidden"]["effect_rel"]
          and out["bow"]["effect_rel"] > out["hidden"]["effect_rel"])
    print(f"\n  context-carrying phi finds context-driven structure the hidden state "
          f"misses: {'OK' if ok else 'FAIL'}")
    # the fallback guarantee must survive every phi mode
    for mode in ("hidden", "hidden+ctx", "bow"):
        os.environ["ACG_CCPO_PHI"] = mode
        import ccpo.core_ccpo as m
        importlib.reload(m)
        _, d0 = m.ccpo_step_advantage(phi=m.FrozenPhi(), rho=0.0, **kw)
        assert d0["lam_u_mean"] < 1e-9, mode
    print("  rho=0 -> lam=0 under every phi mode: OK")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
