#!/usr/bin/env python3
"""Hardlink a checkpoint out of the rolling window from OUTSIDE the trainer.

    scripts/pin_watch.py --exp experiments/ccpo-attncred-150-20260913 --steps 100

The trainer already does this in-process when ACG_PIN_STEPS is set
(ray_trainer.py, the `[pin]` block). This is a belt-and-braces copy for a run that
was launched before that code was synced into verl-agent/, or when you simply cannot
prove which build the live process loaded -- verl-agent/ is gitignored, so its
contents at launch time are not recoverable after the fact.

Safe to run alongside the in-process pin: if step<N>-pin already exists it does
nothing and exits. `cp -al` costs no disk -- the pin holds the same inodes as the
rolling copy until the pruner removes that.

TIMING. Pruning runs immediately after the NEXT save (keep=ACG_KEEP_CKPTS newest),
so with save_freq=5 and keep_ckpts=1 a step-100 checkpoint is deleted when the
step-105 save completes -- a window of about 5 steps. At 900 s/step that is ~75
minutes, so a 60 s poll has ample margin.

The copy is gated on latest_checkpointed_iteration.txt reaching N, which verl writes
only after the actor and dataloader are on disk. Without that gate a pin could
capture a half-written save.
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def log(exp_dir, msg):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(exp_dir, "outputs", "pin_watch.log"), "a") as fh:
        fh.write(line + "\n")


def marker(ck_dir):
    p = os.path.join(ck_dir, "latest_checkpointed_iteration.txt")
    try:
        return int(open(p).read().strip())
    except (OSError, ValueError):
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--steps", required=True,
                    help="comma-separated step numbers to pin, e.g. 100 or 100,125")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--stale", type=int, default=10800,
                    help="give up if train.log has not advanced for this long (default 3h)")
    a = ap.parse_args()

    exp_dir = a.exp if os.path.isabs(a.exp) else os.path.join(ROOT, a.exp)
    ck_dir = os.path.join(exp_dir, "outputs", "checkpoints")
    if not os.path.isdir(ck_dir):
        sys.exit(f"no checkpoint directory: {ck_dir}")

    cfg = json.load(open(os.path.join(exp_dir, "config.json"))).get("config", {})
    total = int(cfg.get("total_epochs", 0))
    want = sorted({int(s) for s in a.steps.split(",") if s.strip()})
    log(exp_dir, f"pin_watch: {os.path.basename(exp_dir)} steps={want} "
                 f"poll={a.poll}s (in-process ACG_PIN_STEPS may beat us to it; that is fine)")

    while want:
        done = marker(ck_dir)
        for step in list(want):
            pin = os.path.join(ck_dir, f"step{step}-pin")
            src = os.path.join(ck_dir, f"global_step_{step}")
            if os.path.isdir(pin):
                log(exp_dir, f"step {step} already pinned ({pin}); nothing to do")
                want.remove(step)
                continue
            if done < step or not os.path.isdir(src):
                continue
            # cp -al: hardlink tree, so the pin costs no extra disk until the
            # rolling copy is pruned. Name must NOT be global_step_* -- the pruner
            # int()s that suffix and would treat the pin as a rolling checkpoint.
            r = subprocess.run(["cp", "-al", src, pin], capture_output=True, text=True)
            if r.returncode == 0 and os.path.isdir(pin):
                log(exp_dir, f"pinned step {step}: {src} -> {pin}")
                want.remove(step)
            else:
                log(exp_dir, f"pin of step {step} FAILED: {r.stderr.strip()[:200]}")

        if not want:
            break
        # stop chasing steps a finished or dead run will never reach
        rows = 0
        mp = os.path.join(exp_dir, "outputs", "metrics.jsonl")
        if os.path.exists(mp):
            with open(mp) as fh:
                rows = sum(1 for _ in fh)
        lp = os.path.join(exp_dir, "outputs", "train.log")
        age = time.time() - os.path.getmtime(lp) if os.path.exists(lp) else 0
        if total and rows >= total and all(marker(ck_dir) >= s for s in want):
            log(exp_dir, f"run finished at {rows} steps; {want} never appeared. giving up.")
            return 1
        if age > a.stale:
            log(exp_dir, f"train.log stale for {age / 3600:.1f}h and {want} not yet saved. "
                         f"giving up -- restart pin_watch if the run resumes.")
            return 1
        time.sleep(a.poll)

    log(exp_dir, "all requested steps pinned. exiting.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
