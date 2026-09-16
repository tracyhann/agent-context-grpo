#!/usr/bin/env python3
"""Guards on the main-method arms. CPU only, no GPU, ~2 s.

    python3 official-repo/ccpo/test_arms.py

1  every key an arm sets is a real exp_run config key (a typo would exit at launch)
2  attncred/alfworld/1.5b reproduces the historical control's config.json, with
   only the documented deltas
3  the benchmark, backbone and method overlays each change exactly what they claim
4  flash attention is pinned on both sides and the preflight refuses fake builds
5  the trainer still applies ACG_CCPO_EP_W, which is the whole of context-adv-only
6  experiments/experiments.md names the same variants and exp-ids the code builds
7  the repo is self-contained: every path the arms need resolves inside it
"""
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("ACG_ROOT") or os.path.dirname(HERE)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


er = _load("exp_run", os.path.join(ROOT, "scripts", "exp_run.py"))
arms = _load("arms", os.path.join(HERE, "arms.py"))

CONTROL_CFG = os.path.join(ROOT, "experiments", arms.CONTROL, "config.json")
TRAINER = os.path.join(ROOT, "patches", "verl-agent", "verl", "trainer", "ppo", "ray_trainer.py")

# Keys the control run predates: it executed with the core_ccpo defaults these
# reproduce, so they are compared against those values rather than skipped.
IMPLIED = {"ccpo_ep_w": 1.0, "ccpo_ctx_w": 1.0, "ccpo_wmode": "soft", "ccpo_lk_fix": ""}
# The one intended difference from the control, from experiments.md: every run goes
# to 150 steps, so nothing may stop it early.
INTENDED = {"early_stop_patience": (8, 0)}


def _resolved(cfg):
    """cfg as exp_run.main() would see it after --set coercion."""
    return {k: er._coerce(str(v)) for k, v in cfg.items()}


def test_keys_are_real():
    bad = {}
    for m in arms.METHODS:
        for b in arms.benchmarks_for(m):
            for k in arms.BACKBONE:
                _, cfg = arms.build(m, b, k)
                for key in cfg:
                    if key not in er.DEFAULTS:
                        bad.setdefault(key, []).append(f"{m}/{b}/{k}")
    for key, where in bad.items():
        print(f"  unknown exp_run key {key!r} in {where[0]} -- launch would exit")
    ok = not bad
    n = sum(len(arms.BACKBONE) * len(arms.benchmarks_for(m)) for m in arms.METHODS)
    print(f"  every key of all {n} arms is a real config key: {'OK' if ok else 'FAIL'}")
    return ok


def test_matches_control():
    ctl = json.load(open(CONTROL_CFG))["config"]
    name, cfg = arms.build("attncred", "alfworld", "1.5b")
    got = _resolved(cfg)
    diffs = {}
    for k, v in sorted(got.items()):
        want = ctl[k] if k in ctl else IMPLIED.get(k)
        if k not in ctl and k not in IMPLIED:
            continue
        if v != want:
            diffs[k] = (want, v)
    unexpected = {k: v for k, v in diffs.items()
                  if not (k in INTENDED and INTENDED[k] == v)}
    print(f"  {name} vs {arms.CONTROL}: {len(got)} keys compared")
    for k, (a, b) in sorted(diffs.items()):
        tag = "intended" if k in INTENDED else "UNEXPECTED"
        print(f"    {k}: control {a!r} -> arm {b!r}   [{tag}]")
    ok = not unexpected
    print(f"  only the documented delta: {'OK' if ok else 'FAIL'}")
    return ok


def test_overlays():
    """Each overlay must move exactly the keys it claims and nothing else."""
    base = _resolved(arms.build("attncred", "alfworld", "1.5b")[1])
    ws_base = _resolved(arms.build("attncred", "webshop", "1.5b")[1])
    cases = {
        "benchmark webshop": (arms.build("attncred", "webshop", "1.5b")[1],
                              {"env_name", "max_steps", "ccpo_target"}),
        "backbone 7b": (arms.build("attncred", "alfworld", "7b")[1], {"model"}),
        "method context-adv-only": (arms.build("attncred-context-adv-only", "alfworld", "1.5b")[1],
                                    {"ccpo_ep_w"}),
        # M5 is a WebShop arm, so it is judged against the WebShop control, not the
        # ALFWorld one -- otherwise env_name/max_steps show up as "moved" and the two
        # keys that actually define it are buried.
        "method context-adv-only-return (vs the webshop control)": (
            arms.build("attncred-context-adv-only-return", "webshop", "1.5b")[1],
            {"ccpo_ep_w", "ccpo_target"}, ws_base),
    }
    ok = True
    for label, case in cases.items():
        cfg, expect = case[0], case[1]
        against = case[2] if len(case) > 2 else base
        moved = {k for k, v in _resolved(cfg).items() if against.get(k) != v}
        good = moved == expect
        ok &= good
        print(f"  {label}: moved {sorted(moved)} (expected {sorted(expect)}) "
              f"{'OK' if good else 'FAIL'}")
    # the one that matters most: WebShop must carry the dense-score target
    ws = _resolved(arms.build("attncred", "webshop", "1.5b")[1])
    dense = ws["ccpo_target"] == "score"
    ok &= dense
    print(f"  webshop uses the dense score target: {'OK' if dense else 'FAIL'}")
    # and context-adv-only must actually zero the episode term
    ctx = _resolved(arms.build("attncred-context-adv-only", "webshop", "7b")[1])
    zeroed = float(ctx["ccpo_ep_w"]) == 0.0
    ok &= zeroed
    print(f"  context-adv-only sets ccpo_ep_w=0: {'OK' if zeroed else 'FAIL'}")
    return bool(ok)


def test_flash_attn():
    ctl = json.load(open(CONTROL_CFG))["config"]
    _, cfg = arms.build("attncred", "alfworld", "1.5b")
    ok = True
    for k, want in (("vllm_attn_backend", "FLASH_ATTN"), ("remove_padding", True)):
        good = cfg.get(k) == want == ctl[k]
        ok &= good
        print(f"  {k}: arm {cfg.get(k)!r}, control {ctl[k]!r} {'OK' if good else 'FAIL'}")

    tmp = tempfile.mkdtemp(prefix="fa-guard-")
    saved = os.environ.get("PYTHONPATH")
    try:
        shutil.copytree(os.path.join(ROOT, "docker", "fa_stub", "flash_attn"),
                        os.path.join(tmp, "flash_attn"))
        for label, pypath in (("nothing installed", None),
                              ("docker/fa_stub on PYTHONPATH",
                               os.path.join(ROOT, "docker", "fa_stub")),
                              ("stub copied under another name", tmp)):
            if pypath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = pypath
            accepted, reason, _ = arms.flash_attn_status(sys.executable)
            ok &= not accepted
            print(f"  {label}: {'ACCEPTED -- FAIL' if accepted else 'refused'} ({reason})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if saved is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = saved
    return bool(ok)


def test_trainer_wiring():
    """The knobs these arms rely on must still exist where the trainer reads them."""
    src = open(TRAINER).read()
    checks = {
        "trainer reads ACG_CCPO_EP_W":
            re.search(r'_ep_w\s*=\s*float\(os\.environ\.get\(\s*["\']ACG_CCPO_EP_W["\']', src),
        "trainer applies it to episode_adv":
            re.search(r'scores\s*=\s*_ep_w\s*\*\s*episode_adv', src),
        "trainer builds the dense target when ccpo_target=score":
            re.search(r"dense_step_returns\(data", src) and re.search(r"'score'", src),
    }
    env = er.build_env(dict(er.DEFAULTS, gpus="0,1", alfworld_data="/x", hf_home="/x",
                            venv_python="/nonexistent"), "/tmp/exp")
    checks["exp_run plumbs ccpo_lk_fix -> ACG_CCPO_LK_FIX"] = "ACG_CCPO_LK_FIX" in env
    ok = True
    for label, good in checks.items():
        ok &= bool(good)
        print(f"  {label}: {'OK' if good else 'FAIL'}")
    return bool(ok)


def test_doc_matches_code():
    """experiments.md is the spec; a name only it knows is a name nobody can run.

    Every CCPO-* variant name and every ccpo-* exp-id the document mentions must be
    one the registries actually build, and every variant must be documented.
    """
    doc_path = os.path.join(ROOT, "experiments", "experiments.md")
    doc = open(doc_path).read()
    abl = _load("ablations", os.path.join(ROOT, "ablations", "ablations.py"))

    code_names = {arms.variant_name(m, b) for m in arms.METHODS
                  for b in arms.benchmarks_for(m)}
    code_names |= {spec["name"] for spec in abl.ABLATIONS.values()}
    code_ids = {arms.build(m, b, k)[0] for m in arms.METHODS
                for b in arms.benchmarks_for(m) for k in arms.BACKBONE}
    code_ids |= {abl.build(a, b)[0] for a in abl.ABLATIONS
                 for b in abl.benchmarks_for(a)}

    doc_names = set(re.findall(r"CCPO-ATTNCRED[A-Z-]*", doc))
    doc_ids = set(re.findall(r"ccpo-attncred[a-z0-9.\-]*", doc))

    unknown_names = {n for n in doc_names if n not in code_names}
    unknown_ids = {i for i in doc_ids if i not in code_ids and i not in arms.HISTORICAL}
    missing = code_names - doc_names
    for n in sorted(unknown_names):
        print(f"    doc names {n!r}, which no registry builds")
    for i in sorted(unknown_ids):
        print(f"    doc cites exp-id {i!r}, which no registry builds")
    for n in sorted(missing):
        print(f"    {n} is implemented but undocumented")
    ok = not (unknown_names or unknown_ids or missing)
    cited = len(doc_ids & arms.HISTORICAL)
    print(f"  experiments.md: {len(doc_names)} variant names, {len(doc_ids) - cited} arm "
          f"exp-ids ({cited} historical runs cited), all resolve both ways: "
          f"{'OK' if ok else 'FAIL'}")
    return ok


# Nothing is exempt: every vendored file resolves its paths from the repo root.
_EXEMPT_ABS = set()
# Split so this scanner does not match its own source.
_ABS_NEEDLE = "/work" + "space"


def test_self_contained():
    """Everything the planned experiments need must live under ROOT.

    Provisioned at deploy time and deliberately absent from the repo: verl-agent/
    (the trainer checkout, built by scripts/sync_patches.sh), .venv/ and
    .venv-webshop/, and the data and weight directories. Those are runtime, not code.
    """
    needed = {
        "the launcher": os.path.join(ROOT, "scripts", "exp_run.py"),
        "the estimator": os.path.join(ROOT, "ccpo", "core_ccpo.py"),
        "the trainer overlay": os.path.join(ROOT, "patches", "verl-agent", "verl",
                                            "trainer", "ppo", "ray_trainer.py"),
        "the phi hook": os.path.join(ROOT, "patches", "verl-agent", "verl", "workers",
                                     "actor", "dp_actor.py"),
        "the FA2 stub": os.path.join(ROOT, "docker", "fa_stub", "flash_attn"),
        "the control config": CONTROL_CFG,
        "reporting": os.path.join(ROOT, "scripts", "report_results.py"),
        "checkpoint re-scoring": os.path.join(ROOT, "scripts", "chain_eval.py"),
        "environment build": os.path.join(ROOT, "scripts", "setup_env.sh"),
        "patch sync": os.path.join(ROOT, "scripts", "sync_patches.sh"),
        "the spec": os.path.join(ROOT, "experiments", "experiments.md"),
    }
    ok = True
    for label, path in needed.items():
        good = os.path.exists(path)
        ok &= good
        if not good:
            print(f"    MISSING {label}: {os.path.relpath(path, ROOT)}")
    print(f"  all {len(needed)} vendored dependencies present under ROOT: "
          f"{'OK' if ok else 'FAIL'}")

    # and nothing reaches back out to a checkout-specific absolute path
    stray = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "verl-agent", ".venv", ".venv-webshop",
                                    "outputs", "__pycache__"}]
        for fn in filenames:
            if not fn.endswith((".py", ".sh")):
                continue
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, ROOT)
            if rel in _EXEMPT_ABS:
                continue
            try:
                body = open(full, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            if _ABS_NEEDLE in body:
                stray.append(rel)
    for rel in sorted(stray):
        print(f"    {rel} hardcodes a path outside the repo")
    clean = not stray
    print(f"  no vendored file names a path outside the repo: "
          f"{'OK' if clean else 'FAIL'}")
    return bool(ok and clean)


def main():
    print("[guard] main-method arms")
    results = [test_keys_are_real(), test_matches_control(), test_overlays(),
               test_flash_attn(), test_trainer_wiring(), test_doc_matches_code(),
               test_self_contained()]
    ok = all(results)
    print(f"[guard] {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
