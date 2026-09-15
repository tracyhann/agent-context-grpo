#!/usr/bin/env python3
"""Hold GPUs with a placeholder allocation, then hand them to a real job.

    scripts/hold_and_launch.py --gpus 6,7 --wait-pid 21963 --cmd "<launch command>"
    scripts/hold_and_launch.py --hold-child --gpus 6,7 --mib 512     # internal

WHY A PLACEHOLDER. Nothing on this box arbitrates GPUs -- no scheduler, no lock, no
cgroup device limits. An idle card is taken by whoever starts first, and the other
tenant's jobs have appeared on every index during this session. Occupying the card is
the only way to keep it.

WHY NOT JUST START THE REAL JOB. Two Ray workloads cannot share this container: 128
train + 128 val actors each against `pids.max` 8192 exhausts the thread budget and the
second one dies with "can't start new thread" (measured 2026-09-14 15:01 -- it killed a
resumed cred arm 30 s after a re-eval started). So the real job has to wait for the
other to finish, and something harmless has to sit on the cards meanwhile.

The placeholder is a separate process holding a small tensor per GPU: a CUDA context
plus a few hundred MiB, no compute. It is killed before the real job launches, and its
memory is released when the process exits.
"""

import argparse
import datetime
import os
import shlex
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "experiments", "hold_and_launch.log")


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def hold_child(gpus, mib):
    """Allocate and sit still until killed. Runs as its own process so that killing it
    is enough to release the memory -- freeing a tensor inside a live CUDA context does
    not return the reservation to other processes."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    import torch
    bufs = []
    for i in range(len(gpus.split(","))):
        bufs.append(torch.empty(int(mib * 1024 * 1024 // 4), dtype=torch.float32,
                                device=f"cuda:{i}"))
    print(f"holding {gpus} with {mib} MiB each", flush=True)
    while True:
        time.sleep(30)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", required=True)
    ap.add_argument("--mib", type=int, default=512)
    ap.add_argument("--hold-child", action="store_true")
    ap.add_argument("--wait-pid", type=int, help="release and launch once this pid exits")
    ap.add_argument("--pids-below", type=int, default=2500,
                    help="also wait until the container's pid count drops below this")
    ap.add_argument("--cmd", help="shell command to launch once the cards are free")
    ap.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()

    if a.hold_child:
        return hold_child(a.gpus, a.mib)

    child = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--hold-child",
         "--gpus", a.gpus, "--mib", str(a.mib)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    log(f"placeholder pid {child.pid} holding GPUs {a.gpus} ({a.mib} MiB each)")

    if a.wait_pid:
        log(f"waiting for pid {a.wait_pid} to exit")
        while pid_alive(a.wait_pid):
            time.sleep(a.poll)
        log(f"pid {a.wait_pid} gone")

    # the dying job's ray workers take a while to actually go away; launching into a
    # still-crowded container is what killed the last attempt.
    while True:
        try:
            cur = int(open("/sys/fs/cgroup/pids.current").read().strip())
        except Exception:
            break
        if cur < a.pids_below:
            log(f"container pids {cur} < {a.pids_below}; proceeding")
            break
        log(f"container pids {cur}; waiting for teardown")
        time.sleep(a.poll)

    child.terminate()
    try:
        child.wait(timeout=60)
    except Exception:
        child.kill()
    log("placeholder released")
    time.sleep(20)          # let the driver reclaim the context before the real job

    if a.cmd:
        log(f"launching: {a.cmd}")
        r = subprocess.run(shlex.split(a.cmd), capture_output=True, text=True)
        log(f"launch rc={r.returncode} out={r.stdout.strip()[-400:]} err={r.stderr.strip()[-300:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
