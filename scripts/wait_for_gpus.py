#!/usr/bin/env python3
"""Wait until N GPUs are genuinely idle, then launch a job on them.

    scripts/wait_for_gpus.py --need 2 --cmd-template "... --set gpus={gpus} ..."

WHY THIS EXISTS. Nothing arbitrates GPUs on this box -- no scheduler, no lock, no
cgroup device limits -- and other tenants take idle cards within the hour. On
2026-09-15 an arm OOM'd at 04:03, the cards sat empty for ~60 minutes, and by 05:05
all eight were taken. Claiming a card means starting work on it the moment it frees.

TWO GUARDS, both learned from failures here:

* CONSECUTIVE CHECKS. A card reads idle in the seconds between a neighbour's phases.
  `--confirm` samples must ALL be idle before we believe it, spaced `--poll` apart.
* PID BUDGET. This container's `pids.max` is 8192 and one arm needs ~7000, so two
  jobs cannot coexist here at all: a second launch dies with "can't start new
  thread" and can take the first one with it. We refuse to launch while the
  container is busy, regardless of what the GPUs say.

The command template gets {gpus} substituted with the claimed indices, e.g. "4,5".
"""

import argparse
import datetime
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "experiments", "wait_for_gpus.log")


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def idle_gpus(floor_mib):
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        log(f"nvidia-smi failed: {e}")
        return []
    free = []
    for line in out.strip().splitlines():
        idx, mem = [x.strip() for x in line.split(",")]
        if int(mem) <= floor_mib:
            free.append(idx)
    return free


def container_pids():
    try:
        return int(open("/sys/fs/cgroup/pids.current").read().strip())
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--need", type=int, default=2)
    ap.add_argument("--cmd-template", required=True,
                    help="shell command; {gpus} is replaced with the claimed indices")
    ap.add_argument("--floor-mib", type=int, default=600,
                    help="a GPU counts as idle below this (a bare CUDA context is ~300)")
    ap.add_argument("--confirm", type=int, default=3,
                    help="consecutive idle samples required before claiming")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--pids-below", type=int, default=1500,
                    help="refuse to launch while the container is busier than this")
    ap.add_argument("--timeout-hours", type=float, default=72.0)
    a = ap.parse_args()

    log(f"watching for {a.need} idle GPUs (<={a.floor_mib} MiB, {a.confirm} consecutive "
        f"checks {a.poll}s apart); will launch: {a.cmd_template}")
    deadline = time.time() + a.timeout_hours * 3600
    streak, claimed = 0, None

    while time.time() < deadline:
        free = idle_gpus(a.floor_mib)
        pids = container_pids()
        if len(free) >= a.need and pids < a.pids_below:
            cand = ",".join(free[:a.need])
            if cand == claimed:
                streak += 1
            else:
                claimed, streak = cand, 1
            log(f"GPUs {cand} idle ({streak}/{a.confirm}), container pids {pids}")
            if streak >= a.confirm:
                cmd = a.cmd_template.format(gpus=claimed)
                log(f"claiming {claimed}; launching")
                r = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
                log(f"rc={r.returncode} out={r.stdout.strip()[-500:]}")
                if r.stderr.strip():
                    log(f"err={r.stderr.strip()[-300:]}")
                return 0 if r.returncode == 0 else 1
        else:
            if streak:
                log(f"lost the window (free={free}, pids={pids}); resetting")
            streak, claimed = 0, None
        time.sleep(a.poll)

    log(f"gave up after {a.timeout_hours} h without {a.need} free GPUs")
    return 1


if __name__ == "__main__":
    sys.exit(main())
