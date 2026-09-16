#!/usr/bin/env python3
"""Wait for a training arm to finish, then re-score checkpoints on a fresh draw.

    scripts/chain_eval.py --after experiments/ccpo-attncred-150-20260913 \
        --ckpt experiments/ccpo-attncred-150-20260913/outputs/checkpoints/step125-best \
        --repeats 3 --gpus 4,5

WHY RE-SCORE AT ALL. best.json takes the maximum of a noisy series, so it is biased
upward by ~1.5 sd (H-Z): ccpo-global-ext reported best 85.9 against a mean of 79.69
over nine evaluations -- almost exactly the expected maximum of nine draws. A fresh
pass gives an unbiased number for the checkpoint people will actually quote.

ALIGN_VAL_ON_RESUME=0 IS REQUIRED. The alignment burn advances the validation draw by
global_steps/test_freq resets, so checkpoints saved at different steps would each be
scored on a DIFFERENT draw -- reintroducing the confound this exists to remove. With
it off, every pass here sees the same 128 episodes, which makes repeats a measurement
of decoding noise (T=0.4) and makes multiple checkpoints directly paired.

THE PIN/BEST NAMING TRAP. verl asserts the resume path contains "global_step_", and
the rolling pruner int()s whatever follows it -- which is why best/pin checkpoints are
NOT named that way. This script hardlinks (cp -al, no extra disk) to a conforming name
before resuming, and leaves the link in place.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(msg, fh=None):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


def rows(exp_dir):
    p = os.path.join(exp_dir, "outputs", "metrics.jsonl")
    if not os.path.exists(p):
        return 0
    with open(p) as f:
        return sum(1 for _ in f)


def gpu_busy(gpus, floor_mib=2000):
    """True while any named GPU still holds memory -- the training process exiting is
    what frees it, and that is the signal we can see across PID namespaces."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return False
    want = {g.strip() for g in gpus.split(",")}
    for line in out.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(",")]
        if idx in want and int(mem) > floor_mib:
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", required=True, help="experiment dir to wait on")
    ap.add_argument("--ckpt", action="append", required=True,
                    help="checkpoint dir to score (repeatable)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--gpus", default="4,5")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--no-wait", action="store_true", help="skip waiting; score now")
    a = ap.parse_args()

    after = a.after if os.path.isabs(a.after) else os.path.join(ROOT, a.after)
    cfg = json.load(open(os.path.join(after, "config.json")))["config"]
    total = int(cfg["total_epochs"])
    logf = open(os.path.join(after, "outputs", "chain_eval.log"), "a")

    if not a.no_wait:
        log(f"waiting for {os.path.basename(after)} to reach {total} steps", logf)
        while rows(after) < total:
            time.sleep(a.poll)
        log(f"training reached {rows(after)}/{total}; waiting for GPUs {a.gpus} to free", logf)
        while gpu_busy(a.gpus):
            time.sleep(a.poll)
        time.sleep(60)  # let ray tear down before claiming the cards
    log(f"GPUs {a.gpus} free; starting {len(a.ckpt)} checkpoint(s) x {a.repeats} pass(es)", logf)

    results = {}
    for ck in a.ckpt:
        ck = ck if os.path.isabs(ck) else os.path.join(ROOT, ck)
        name = os.path.basename(ck)
        # verl asserts on "global_step_" in the resume path; hardlink to a conforming
        # name. int() of the suffix must parse, so derive the step from the dir name.
        m = re.search(r"step(\d+)", name)
        if not m:
            log(f"cannot derive a step number from {name}; skipping", logf); continue
        step = m.group(1)
        resume = ck
        if "global_step_" not in name:
            resume = os.path.join(os.path.dirname(ck), f"global_step_{step}")
            if not os.path.isdir(resume):
                r = subprocess.run(["cp", "-al", ck, resume], capture_output=True, text=True)
                if r.returncode != 0:
                    log(f"hardlink {ck} -> {resume} FAILED: {r.stderr.strip()[:200]}", logf); continue
                log(f"hardlinked {name} -> global_step_{step} (no extra disk)", logf)

        for k in range(1, a.repeats + 1):
            tag = f"evalck-{os.path.basename(after).split('-')[1]}{step}-r{k}"
            log(f"pass {k}/{a.repeats} for {name}: {tag}", logf)
            cmd = [sys.executable, os.path.join(ROOT, "scripts", "exp_run.py"),
                   "--name", tag, "--arm", "ccpo", "--no-plot",
                   "--set", f"gpus={a.gpus}", "--set", "val_only=1",
                   "--set", "align_val_on_resume=0", "--set", f"resume_from={resume}",
                   "--set", "total_epochs=1"]
            subprocess.run(cmd, capture_output=True, text=True)
            d = os.path.join(ROOT, "experiments",
                             f"{tag}-{datetime.date.today().strftime('%Y%m%d')}")
            # wait for the eval process to write its one validation row
            score = None
            for _ in range(240):
                time.sleep(15)
                mp = os.path.join(d, "outputs", "metrics.jsonl")
                if os.path.exists(mp):
                    for line in open(mp):
                        r = json.loads(line)
                        if r.get("val/success_rate") is not None:
                            score = r["val/success_rate"] * 100
                if score is not None:
                    break
            if score is None:
                log(f"  {tag}: NO SCORE (see {d}/outputs/train.log)", logf)
            else:
                log(f"  {tag}: {score:.2f}", logf)
                results.setdefault(name, []).append(score)

    log("=== re-evaluation summary (one common draw, repeats = decoding noise) ===", logf)
    for name, ss in results.items():
        m = sum(ss) / len(ss)
        sd = (sum((x - m) ** 2 for x in ss) / max(len(ss) - 1, 1)) ** 0.5
        log(f"  {name}: mean {m:.2f}  sd {sd:.2f}  n {len(ss)}  passes {[round(x,1) for x in ss]}", logf)
    logf.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
