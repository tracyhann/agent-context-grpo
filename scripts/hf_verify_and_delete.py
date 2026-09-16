#!/usr/bin/env python3
"""After the HF upload: prove each checkpoint is RESUMABLE and byte-identical, then delete it.

    scripts/hf_verify_and_delete.py --exp experiments/<exp-id>


Three gates per checkpoint, all must pass before anything is removed:

 1. RESUMABLE STRUCTURE. verl resumes FSDP from per-rank shards, so for world_size W we
    require model_/optim_/extra_state_world_size_W_rank_{0..W-1}.pt to all be present,
    plus data.pt (dataloader state) and actor/huggingface/config.json (model init).
    A checkpoint missing one optim shard loads and then silently resumes without
    optimizer state -- which is why this is checked explicitly rather than by file count.
 2. LOCAL INTEGRITY. Every .pt and .safetensors opens and its directory parses. Torch
    saves are zip archives, so a truncated upload source is caught here rather than
    months later.
 3. REMOTE IDENTITY. Local sha256 == the LFS sha256 HF reports for every file. This is
    end-to-end: it proves what is on HF is what is on disk, without downloading 75 GB.

Only then is the local copy removed. Anything short of all three leaves it in place.
"""
import argparse, hashlib, json, os, re, sys, time, zipfile, shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--exp", default="experiments/ccpo-attncred-150-20260913",
                 help="experiment directory, absolute or relative to the repo root")
_ap.add_argument("--repo", default="tracyhan816/ccpo-variants", help="Hugging Face repo id")
_a = _ap.parse_args()

D   = _a.exp if os.path.isabs(_a.exp) else os.path.join(ROOT, _a.exp)
CK  = f"{D}/outputs/checkpoints"
BLOG= f"{D}/outputs/hf_backup.log"
VLOG= f"{D}/outputs/hf_verify.log"
REPO, PREFIX = _a.repo, os.path.basename(D.rstrip("/"))

def log(m):
    line = f'[{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}] {m}'
    print(line, flush=True)
    try:
        with open(VLOG, "a") as f: f.write(line + "\n")
    except Exception: pass

def sha256(p, buf=8 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b: break
            h.update(b)
    return h.hexdigest()

def structure_ok(d):
    files = {os.path.relpath(os.path.join(r, f), d)
             for r, _, fs in os.walk(d) for f in fs}
    ws = {int(m.group(1)) for f in files for m in [re.search(r"world_size_(\d+)_rank_", f)] if m}
    if not ws:
        log(f"    FAIL: no world_size shards found"); return False
    W = max(ws)
    need = {f"actor/{k}_world_size_{W}_rank_{r}.pt"
            for k in ("model", "optim", "extra_state") for r in range(W)}
    need |= {"data.pt", "actor/huggingface/config.json"}
    missing = sorted(need - files)
    if missing:
        log(f"    FAIL: world_size {W}, missing {missing}"); return False
    log(f"    structure OK: world_size {W}, all model/optim/extra_state shards + data.pt")
    return True

def integrity_ok(d):
    bad = []
    for r, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(r, f)
            if f.endswith(".pt"):
                try:
                    if not zipfile.is_zipfile(p): raise ValueError("not a zip archive")
                    with zipfile.ZipFile(p) as z:
                        if not z.namelist(): raise ValueError("empty archive")
                except Exception as e:
                    bad.append(f"{f}: {type(e).__name__} {e}")
            elif f.endswith(".safetensors"):
                try:
                    with open(p, "rb") as fh:
                        n = int.from_bytes(fh.read(8), "little")
                        json.loads(fh.read(n).decode())
                except Exception as e:
                    bad.append(f"{f}: header unreadable ({type(e).__name__})")
    if bad:
        for b in bad[:5]: log(f"    FAIL integrity: {b}")
        return False
    log("    integrity OK: every .pt archive and .safetensors header parses")
    return True

def remote_ok(d, name, remote):
    local = {os.path.relpath(os.path.join(r, f), d): os.path.join(r, f)
             for r, _, fs in os.walk(d) for f in fs}
    miss = [k for k in local if f"{PREFIX}/{name}/{k}" not in remote]
    if miss:
        log(f"    FAIL: {len(miss)} file(s) absent on HF, e.g. {miss[:3]}"); return False
    bad = 0
    for rel, p in sorted(local.items()):
        meta = remote[f"{PREFIX}/{name}/{rel}"]
        rsha, rsize = meta
        if rsha:
            if sha256(p) != rsha:
                log(f"    FAIL sha256 mismatch: {rel}"); bad += 1
        elif rsize != os.path.getsize(p):
            log(f"    FAIL size mismatch: {rel}"); bad += 1
    if bad: return False
    log(f"    remote identity OK: {len(local)} files match HF by sha256/size")
    return True

# ---- wait for the uploader to finish -------------------------------------------
log("armed; waiting for the upload watcher to exit")
for _ in range(24 * 60):
    # Match only a real python process running the uploader. A substring search over
    # every cmdline also matches shell jobs that merely MENTION the script (a pgrep
    # waiter, this very edit) -- which made the first attempt wait on its own watcher.
    def _uploader_alive():
        me = os.getpid()
        for q in os.listdir("/proc"):
            if not q.isdigit() or int(q) == me:
                continue
            try:
                cl = open(f"/proc/{q}/cmdline", "rb").read().decode("utf8", "ignore").split("\x00")
            except Exception:
                continue
            cl = [c for c in cl if c]
            if len(cl) >= 2 and "python" in os.path.basename(cl[0]) \
               and any(c.endswith("hf_backup.py") for c in cl[1:]):
                return True
        return False
    if not _uploader_alive():
        break
    time.sleep(60)
else:
    log("uploader still running after 24h; aborting"); sys.exit(1)

tail = open(BLOG).read() if os.path.exists(BLOG) else ""
if "BACKUP COMPLETE, all verified" not in tail:
    log("uploader did NOT report clean completion; refusing to verify or delete")
    log("  last line: " + (tail.strip().splitlines() or ["<empty>"])[-1]); sys.exit(2)
log("uploader reported clean completion; verifying")

os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf"))
from huggingface_hub import HfApi
api = HfApi()
info = api.repo_info(REPO, repo_type="model", files_metadata=True)
remote = {}
for s in info.siblings:
    lfs = getattr(s, "lfs", None)
    sha = (lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)) if lfs else None
    remote[s.rfilename] = (sha, s.size)

# Verify everything; delete only what is both verified AND not in use.
# step150-best is held open by another process (operator instruction), and it is
# HARDLINKED to global_step_150 -- same inode -- so removing global_step_150 would free
# nothing while step150-best survives. Both are therefore kept; only the pin is
# reclaimable, and it is the 25 GB that actually matters.
KEEP = {"step150-best", "global_step_150"}
targets = [os.path.join(CK, n) for n in sorted(os.listdir(CK))
           if os.path.isdir(os.path.join(CK, n)) and not os.path.islink(os.path.join(CK, n))]
freed = 0
for t in targets:
    name = os.path.basename(t)
    log(f"verifying {name}")
    if structure_ok(t) and integrity_ok(t) and remote_ok(t, name, remote):
        if name in KEEP:
            log(f"  VERIFIED RESUMABLE -> KEPT (in use / hardlinked twin; deleting it frees nothing)")
            continue
        sz = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(t) for f in fs)
        shutil.rmtree(t)
        freed += sz
        log(f"  VERIFIED RESUMABLE -> deleted local copy ({sz/1e9:.2f} GB freed)")
    else:
        log(f"  NOT verified -> local copy KEPT")
log(f"done; freed {freed/1e9:.2f} GB")
