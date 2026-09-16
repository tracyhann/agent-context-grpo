#!/usr/bin/env python3
"""Guards on the five ablations. CPU only, ~3 s.

    python3 official-repo/ablations/test_ablations.py

1  each ablation differs from its benchmark's control arm in exactly its declared delta
2  ccpo_wmode=cos really weights by clipped cosine, and falls back when all clip away
3  ccpo_lk_fix pins the credibility weight, and does so even at kappa=0

Tests 2-3 exercise the two estimator paths that exist only for these ablations, so a
refactor of core_ccpo cannot leave them silently inert. They run against numpy; if
torch is not installed a minimal stand-in is used for the two tensor touchpoints
core_ccpo has (`step_rewards` in, advantage tensor out).
"""
import importlib
import importlib.util
import os
import sys
import types

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ACG_ROOT") or os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "ccpo"))
sys.path.insert(0, ROOT)

import ablations                                              # noqa: E402
import arms                                                   # noqa: E402


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


er = _load("exp_run", os.path.join(ROOT, "scripts", "exp_run.py"))


# --------------------------------------------------------------------------
# 1. config level
# --------------------------------------------------------------------------
def test_one_delta_each():
    ok = True
    for key, spec in ablations.ABLATIONS.items():
        for bench in ablations.benchmarks_for(key):
            _, ctl = arms.build("attncred", bench, "1.5b")
            _, abl = ablations.build(key, bench)
            ctl = {k: er._coerce(str(v)) for k, v in ctl.items()}
            abl = {k: er._coerce(str(v)) for k, v in abl.items()}
            moved = {k for k in abl if ctl.get(k) != abl[k]}
            want = set(spec["delta"])
            good = moved == want
            ok &= good
            if not good:
                print(f"    {key}/{bench}: moved {sorted(moved)}, declared {sorted(want)}")
        print(f"  {key:18s} delta {list(spec['delta'])} on "
              f"{'+'.join(ablations.benchmarks_for(key))}: {'OK' if good else 'FAIL'}")
    # the WebShop ablations must inherit the dense-score target from their control
    # every WebShop ablation inherits the dense target from its control, EXCEPT the
    # one whose whole point is to change it back.
    dense = all(ablations.build(k, "webshop")[1]["ccpo_target"] == "score"
                for k in ablations.ABLATIONS
                if "webshop" in ablations.benchmarks_for(k)
                and "ccpo_target" not in ablations.ABLATIONS[k]["delta"])
    ok &= dense
    print(f"  webshop ablations keep the dense-score target: {'OK' if dense else 'FAIL'}")
    return bool(ok)


# --------------------------------------------------------------------------
# estimator fixtures
# --------------------------------------------------------------------------
def _ensure_torch():
    """core_ccpo touches torch twice; stand in for it when it is not installed."""
    try:
        import torch                                          # noqa: F401
        return
    except ImportError:
        pass

    class _T:
        def __init__(self, a): self.a = np.asarray(a)
        @property
        def device(self): return "cpu"
        @property
        def dtype(self): return self.a.dtype
        def dim(self): return self.a.ndim
        def detach(self): return self
        def float(self): return _T(self.a.astype(np.float64))
        def cpu(self): return self
        def numpy(self): return self.a
        def __mul__(self, o): return _T(self.a * (o.a if isinstance(o, _T) else o))
        def sum(self, axis=None): return _T(self.a.sum(axis))

    shim = types.ModuleType("torch")
    shim.Tensor = _T
    shim.float32 = np.float32
    shim.tensor = lambda a, dtype=None, device=None: _T(a)
    shim.ones = lambda *shape: _T(np.ones(shape))
    sys.modules["torch"] = shim


def _core(**env):
    """Reload core_ccpo with a controlled environment (its knobs are read at import)."""
    _ensure_torch()
    base = {
        "ACG_CCPO_PHI": "hidden", "ACG_CCPO_WHITEN": "0", "ACG_CCPO_GATE": "hard",
        "ACG_CCPO_LAM_FIX": "1.0", "ACG_CCPO_PRIOR_KAPPA": "0.0", "ACG_CCPO_LK_FIX": "",
        "ACG_CCPO_EDGE_W": "0.0", "ACG_CCPO_BACKOFF_TASK": "0", "ACG_CCPO_JWEIGHT_C": "0.0",
        "ACG_CCPO_STD": "task", "ACG_CCPO_WMODE": "soft", "ACG_CCPO_TARGET": "return",
        "ACG_CCPO_SIM": "0.0", "ACG_CCPO_SIM_BACKOFF": "0.0", "ACG_CCPO_DUMP": "",
    }
    base.update(env)
    for k, v in base.items():
        if v == "":
            os.environ.pop(k, None)
        else:
            os.environ[k] = str(v)
    import ccpo.core_ccpo as m
    return importlib.reload(m)


def _fixture(m, feats, targets, trajs):
    """One task, one shared observation, one row per entry: a single bucket."""
    import torch
    n = len(targets)
    return dict(
        step_rewards=torch.tensor(np.asarray(targets, dtype=np.float64)),
        response_mask=torch.ones(n, 4),
        anchor_obs=np.array(["the same room"] * n),
        index=np.array(["task0"] * n),
        traj_index=np.array([str(t) for t in trajs]),
        phi_feats=np.asarray(feats, dtype=np.float64),
        phi=m.FrozenPhi(), return_diag=True,
    )


# --------------------------------------------------------------------------
# 2. ccpo_wmode=cos
# --------------------------------------------------------------------------
def test_cosine_weights():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(6, 8))
    targets = [1.0, 2.0, 5.0, 8.0, 3.0, 9.0]
    trajs = [0, 1, 2, 3, 4, 5]

    m = _core(ACG_CCPO_WMODE="cos")
    adv, diag = m.ccpo_step_advantage(**_fixture(m, feats, targets, trajs))
    adv = np.asarray(adv.numpy(), dtype=np.float64)

    # Reference, using the module's own whitening so only the WEIGHTING is under test.
    F = m.whiten_feats(feats, k=0)
    T = np.asarray(targets, dtype=np.float64)
    i = 0
    other = [b for b in range(len(T)) if b != i]
    cos = F[other] @ F[i] / (np.linalg.norm(F[i]) * np.linalg.norm(F[other], axis=1))
    w = np.maximum(cos, 0.0)
    want = T[i] - (w * T[other]).sum() / w.sum()

    clipped = int((cos < 0).sum())
    close = abs(adv[i] - want) < 1e-9
    print(f"  cos: {clipped}/{len(other)} neighbours clipped to zero weight")
    print(f"  cos: adv[0]={adv[i]:.6f} vs hand-computed {want:.6f}: "
          f"{'OK' if close else 'FAIL'}")
    # and it must differ from the kernel it replaces
    m2 = _core(ACG_CCPO_WMODE="soft")
    adv_soft = np.asarray(m2.ccpo_step_advantage(**_fixture(m2, feats, targets, trajs))[0].numpy())
    differs = abs(adv_soft[i] - adv[i]) > 1e-6
    print(f"  cos: differs from wmode=soft ({adv_soft[i]:.6f}): {'OK' if differs else 'FAIL'}")
    return bool(close and clipped and differs)


def test_cosine_fallback():
    """All neighbours clipped -> keep the nearest, never drop the baseline."""
    # Two trajectories, one row each. whiten centres, so F[1] = -F[0]: cos = -1.
    m = _core(ACG_CCPO_WMODE="cos")
    adv, _ = m.ccpo_step_advantage(**_fixture(m, [[1.0, 0.0], [0.0, 1.0]], [4.0, 1.0], [0, 1]))
    adv = np.asarray(adv.numpy(), dtype=np.float64)
    want = 4.0 - 1.0                      # the single neighbour, kept by the fallback
    ok = abs(adv[0] - want) < 1e-9
    print(f"  cos fallback: adv[0]={adv[0]:.6f}, expected {want:.6f}: {'OK' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------
# 3. ccpo_lk_fix
# --------------------------------------------------------------------------
def test_lk_fix():
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(8, 6))
    targets = [1.0, 2.0, 5.0, 8.0, 3.0, 9.0, 4.0, 6.0]
    trajs = [0, 0, 1, 1, 2, 2, 3, 3]

    out = {}
    for label, env in (("kappa=2, derived", {"ACG_CCPO_PRIOR_KAPPA": "2.0"}),
                       ("kappa=2, lk=0.5", {"ACG_CCPO_PRIOR_KAPPA": "2.0",
                                            "ACG_CCPO_LK_FIX": "0.5"}),
                       ("kappa=0, lk=0.5", {"ACG_CCPO_PRIOR_KAPPA": "0.0",
                                            "ACG_CCPO_LK_FIX": "0.5"}),
                       ("kappa=0, derived", {"ACG_CCPO_PRIOR_KAPPA": "0.0"})):
        m = _core(**env)
        _, diag = m.ccpo_step_advantage(**_fixture(m, feats, targets, trajs))
        out[label] = float(diag["lam_k_mean"])
        print(f"  lam_k_mean [{label:16s}] = {out[label]:.4f}")

    # J = 3 sibling trajectories, so the derived weight is 3/(3+2) = 0.6
    derived = abs(out["kappa=2, derived"] - 0.6) < 1e-9
    pinned = abs(out["kappa=2, lk=0.5"] - 0.5) < 1e-9
    # the guard change: b_task is computed for the blend even when kappa is 0
    at_zero = abs(out["kappa=0, lk=0.5"] - 0.5) < 1e-9
    # and with neither, the prior never fires
    off = abs(out["kappa=0, derived"] - 1.0) < 1e-9
    for label, good in (("derived weight is J/(J+kappa)=0.6", derived),
                        ("lk_fix pins it to 0.5", pinned),
                        ("lk_fix works at kappa=0", at_zero),
                        ("no prior without either", off)):
        print(f"  {label}: {'OK' if good else 'FAIL'}")
    return bool(derived and pinned and at_zero and off)


def main():
    print("[guard] ablations")
    results = [test_one_delta_each(), test_cosine_weights(), test_cosine_fallback(),
               test_lk_fix()]
    ok = all(results)
    print(f"[guard] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
