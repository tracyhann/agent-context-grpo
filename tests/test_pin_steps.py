#!/usr/bin/env python3
"""Verify ACG_PIN_STEPS keeps a chosen checkpoint out of the rolling pruner.

The pruner in patches/verl-agent/verl/trainer/ppo/ray_trainer.py keeps only the
ACG_KEEP_CKPTS newest global_step dirs. On a 150-step run with save_freq=5 that
deletes the step-100 checkpoint the moment step 105 lands, which is exactly the
checkpoint a long run is usually asked to preserve.

ACG_PIN_STEPS="100" hardlinks global_step_100 to step100-pin at save time, the
same cp -al trick already used for best-*. Two properties make that safe, and
both are asserted here rather than assumed:

  * the pin shares inodes with the original, so it costs no extra disk until the
    rolling copy is pruned -- checkpoints are 26.6 GB on the 1.5B model;
  * the pruner globs "global_step_*", which cannot match "step100-pin". That
    also keeps the pin clear of the int(name.rsplit("_", 1)[1]) sort key, which
    would raise ValueError on a non-numeric suffix.

Test 1 exercises the pin and pruner logic verbatim on a temporary tree.
Test 2 is a source guard: if either snippet is renamed or its glob changed, the
behavioural test above would silently stop testing the shipped code.
"""
import glob
import os
import subprocess as sp
import sys
import tempfile
import shutil

_os = os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

RAY_TRAINER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "patches", "verl-agent", "verl", "trainer", "ppo", "ray_trainer.py")


def _mkckpt(root, step):
    d = os.path.join(root, f"global_step_{step}")
    os.makedirs(os.path.join(d, "actor"))
    with open(os.path.join(d, "actor", "model.bin"), "w") as f:
        f.write("x" * 1024 + str(step))
    return d


def test_pin_survives_pruning():
    root = tempfile.mkdtemp(prefix="pintest_")
    fails = []

    def chk(cond, msg):
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            fails.append(msg)

    try:
        for st in (90, 95, 100, 105):
            _mkckpt(root, st)

        # --- pin logic, verbatim from ray_trainer.py ---
        os.environ["ACG_PIN_STEPS"] = "100"
        for step in (90, 95, 100, 105):
            _cur = os.path.join(root, f"global_step_{step}")
            for _ps in os.environ.get("ACG_PIN_STEPS", "").split(","):
                _ps = _ps.strip()
                if _ps and _ps == str(step):
                    _pin = os.path.join(root, f"step{_ps}-pin")
                    if not os.path.isdir(_pin):
                        sp.run(["cp", "-al", _cur, _pin], check=False)

        chk(os.path.isdir(os.path.join(root, "step100-pin")), "step100-pin created")
        ino_a = os.stat(os.path.join(root, "global_step_100", "actor", "model.bin")).st_ino
        ino_b = os.stat(os.path.join(root, "step100-pin", "actor", "model.bin")).st_ino
        chk(ino_a == ino_b, "pin is a hardlink: same inode, no extra disk")

        # --- pruner, verbatim from ray_trainer.py ---
        _keep = 1
        _gs = sorted(glob.glob(os.path.join(root, "global_step_*")),
                     key=lambda d: int(d.rsplit("_", 1)[1]))
        chk(all("pin" not in os.path.basename(d) for d in _gs),
            "pruner glob never matches step100-pin")
        for _old in _gs[:-_keep]:
            sp.run(["rm", "-rf", _old], check=False)

        chk(not os.path.isdir(os.path.join(root, "global_step_100")),
            "rolling step-100 pruned as expected")
        chk(os.path.isdir(os.path.join(root, "global_step_105")), "newest rolling ckpt kept")
        pinned = os.path.join(root, "step100-pin", "actor", "model.bin")
        chk(os.path.isfile(pinned), "pin survives pruning")
        with open(pinned) as f:
            chk(f.read().endswith("100"), "pinned content intact after the original is deleted")

        # verl asserts "global_step_" is in trainer.resume_from_path, so a pin must be
        # restored under that name before it can be resumed or evaluated from.
        sp.run(["cp", "-al", os.path.join(root, "step100-pin"),
                os.path.join(root, "global_step_100")], check=False)
        chk(os.path.isfile(os.path.join(root, "global_step_100", "actor", "model.bin")),
            "restorable to a resumable global_step_100 path")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return not fails


def test_source_guard():
    fails = []

    def chk(cond, msg):
        print(f"  {'PASS' if cond else 'FAIL'}  {msg}")
        if not cond:
            fails.append(msg)

    if not os.path.isfile(RAY_TRAINER):
        print(f"  SKIP  {RAY_TRAINER} not found")
        return None
    src = open(RAY_TRAINER).read()
    chk('ACG_PIN_STEPS' in src, "ray_trainer.py reads ACG_PIN_STEPS")
    chk('f"step{_ps}-pin"' in src, "pin is named step<N>-pin, outside the pruner glob")
    chk('"cp", "-al", _cur, _pin' in src, "pin is created with cp -al (hardlink)")
    chk('glob.glob(_os.path.join(_root, "global_step_*"))' in src
        or '_glob.glob(_os.path.join(_root, "global_step_*"))' in src,
        "pruner still globs global_step_* only")
    return not fails


if __name__ == "__main__":
    print("test 1: a pinned checkpoint survives rolling pruning")
    t1 = test_pin_survives_pruning()
    print("\ntest 2: source guard on the shipped pin and pruner snippets")
    t2 = test_source_guard()
    res = [("1", t1), ("2", t2)]
    print("\n" + " | ".join(
        f"test {k} {'PASS' if v else ('SKIPPED' if v is None else 'FAIL')}" for k, v in res))
    sys.exit(0 if all(v is not False for _, v in res) else 1)
