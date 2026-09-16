#!/usr/bin/env python3
"""Wait for a run to finish, then back up its checkpoints to Hugging Face.

    scripts/hf_backup.py --exp experiments/<exp-id> [--steps 150]

Defaults to ccpo-attncred-150-20260913, the run it was written for.

Uploads exactly three, which is everything worth keeping from a 150-step run:
  step100-pin      -- the pinned comparison point against every 100-step arm
  step<N>-best     -- the best-eval checkpoint named by best.json
  global_step_<N>  -- the final state named by latest_checkpointed_iteration.txt

Deletes nothing. Local cleanup stays a human decision; this only makes it safe to make.
Verifies each upload by comparing file count and total bytes against the local tree
before declaring success, because a partial upload that looks fine is the failure mode
that would actually cost us a checkpoint.
"""
import argparse, json, os, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--exp", default="experiments/ccpo-attncred-150-20260913",
                 help="experiment directory, absolute or relative to the repo root")
_ap.add_argument("--repo", default="tracyhan816/ccpo-variants", help="Hugging Face repo id")
_ap.add_argument("--steps", type=int, default=150, help="steps to wait for before uploading")
_a = _ap.parse_args()

D = _a.exp if os.path.isabs(_a.exp) else os.path.join(ROOT, _a.exp)
OUT, CK = f"{D}/outputs", f"{D}/outputs/checkpoints"
MET, TLOG = f"{OUT}/metrics.jsonl", f"{OUT}/train.log"
BLOG = f"{OUT}/hf_backup.log"
REPO, TARGET_STEPS = _a.repo, _a.steps

def log(m):
    line = f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] {m}'
    print(line, flush=True)
    try:
        with open(BLOG, "a") as f: f.write(line + "\n")
    except Exception: pass

def rows():
    try:
        with open(MET) as f: return sum(1 for _ in f)
    except Exception: return 0

def idle():
    try: return time.time() - os.path.getmtime(TLOG)
    except Exception: return 0.0

def tree(p):
    n = b = 0
    for root, _, files in os.walk(p):
        for fn in files:
            try: b += os.path.getsize(os.path.join(root, fn)); n += 1
            except OSError: pass
    return n, b

log(f"armed; waiting for {TARGET_STEPS} steps (now {rows()})")
deadline = time.time() + 24 * 3600
while time.time() < deadline:
    r, a = rows(), idle()
    if r >= TARGET_STEPS and a > 300:
        log(f"run complete: {r} steps, log idle {a:.0f}s"); break
    if a > 2400 and r > 0:
        log(f"run appears STOPPED EARLY at {r} steps (log idle {a:.0f}s); backing up what exists"); break
    time.sleep(60)
else:
    log("timed out after 24h; backing up whatever exists")

targets = []
pin = f"{CK}/step100-pin"
if os.path.isdir(pin): targets.append(pin)
else: log("WARNING: step100-pin missing")
try:
    b = json.load(open(f"{CK}/best.json"))
    p = f"{CK}/step{b['step']}-best"
    if os.path.isdir(p): targets.append(p)
    log(f"best.json -> step {b['step']} @ {b.get('val_success_rate')}")
except Exception as e:
    log(f"WARNING: best.json unreadable ({type(e).__name__})")
try:
    m = open(f"{CK}/latest_checkpointed_iteration.txt").read().strip()
    p = f"{CK}/global_step_{m}"
    if os.path.isdir(p) and os.path.realpath(p) not in [os.path.realpath(t) for t in targets]:
        targets.append(p)
    log(f"final marker -> global_step_{m}")
except Exception as e:
    log(f"WARNING: marker unreadable ({type(e).__name__})")

if not targets:
    log("nothing to upload; exiting"); sys.exit(1)

os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf"))
from huggingface_hub import HfApi
api = HfApi()
prefix = os.path.basename(D.rstrip("/"))
ok = True
for t in targets:
    name = os.path.basename(t)
    ln, lb = tree(t)
    log(f"uploading {name}: {ln} files, {lb/1e9:.2f} GB -> {REPO}/{prefix}/{name}")
    for attempt in (1, 2, 3):
        try:
            api.upload_folder(folder_path=t, path_in_repo=f"{prefix}/{name}",
                              repo_id=REPO, repo_type="model",
                              commit_message=f"{prefix}: {name}")
            break
        except Exception as e:
            log(f"  attempt {attempt} failed: {type(e).__name__} {str(e)[:160]}")
            if attempt == 3: ok = False
            else: time.sleep(60)
    else:
        continue
    try:
        info = api.repo_info(REPO, repo_type="model", files_metadata=True)
        got = [s for s in info.siblings if s.rfilename.startswith(f"{prefix}/{name}/")]
        rb = sum(s.size or 0 for s in got)
        match = (len(got) == ln and abs(rb - lb) < 1024)
        log(f"  verify {name}: remote {len(got)} files {rb/1e9:.2f} GB vs local {ln} {lb/1e9:.2f} GB -> "
            + ("MATCH" if match else "MISMATCH"))
        ok = ok and match
    except Exception as e:
        log(f"  verify failed: {type(e).__name__} {str(e)[:120]}"); ok = False

log("BACKUP COMPLETE, all verified" if ok else "BACKUP FINISHED WITH PROBLEMS - do not delete local copies")
sys.exit(0 if ok else 2)
