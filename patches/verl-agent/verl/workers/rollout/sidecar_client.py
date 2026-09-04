# ACG sidecar rollout client: generation offloaded to a per-GPU vLLM server.
# Enabled only when ACG_SIDECAR_HOST is set; any failure falls back to the local
# HF-generate path, so the trainer can never hang or corrupt a batch on a dead
# or stale sidecar. Weight-freshness handshake: the host sync loop writes
# ACG_SIDECAR_STATE ({"step": N}) after (re)starting servers on the newest
# global_step_N/actor/huggingface; the shim waits until that matches
# latest_checkpointed_iteration.txt before generating.

import json
import os
import time
import urllib.request

import torch


def _rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0


def sidecar_enabled() -> bool:
    return bool(os.getenv("ACG_SIDECAR_HOST"))


def _url() -> str:
    host = os.environ["ACG_SIDECAR_HOST"]
    ports = os.environ.get("ACG_SIDECAR_PORTS", "8100,8101,8102,8103,8104,8105").split(",")
    return f"http://{host}:{ports[_rank() % len(ports)].strip()}/v1/completions"


def _weights_fresh(timeout_s: float = 300.0) -> bool:
    state_path = os.getenv("ACG_SIDECAR_STATE")
    iter_path = os.getenv("ACG_SIDECAR_ITER")
    if not state_path or not iter_path:
        return True  # handshake not configured -> trust the server
    if not os.path.exists(state_path):
        return False  # sidecars never synced yet (fresh launch) -> fall back immediately
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with open(iter_path) as f:
                want = int(f.read().strip())
            with open(state_path) as f:
                have = int(json.load(f)["step"])
            if have == want:
                return True
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(3.0)
    return False



def _think_budget(max_tokens):
    """Think-token cap. ACG_FORCE_BUDGET=0 disables forcing (max_tokens unchanged)."""
    fb = int(os.environ.get("ACG_FORCE_BUDGET", "0"))
    return fb if fb > 0 else int(max_tokens)


def _needs_force(text):
    return "<action>" not in text or "</action>" not in text


_FORCE_IDS = None


def _force_ids():
    """Token ids for the injected '</think><action>' prefix.

    No tokenizer exists in this process, so ask the vLLM server once and cache.
    Returns [] on failure, which disables forcing rather than corrupting a batch.
    """
    global _FORCE_IDS
    if _FORCE_IDS is not None:
        return _FORCE_IDS
    text = os.environ.get("ACG_FORCE_STR", "</think><action>")
    try:
        url = _url().replace("/v1/completions", "/tokenize")
        body = {"model": os.environ.get("ACG_SIDECAR_MODEL", "default"),
                "prompt": text, "add_special_tokens": False}
        req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            _FORCE_IDS = list(json.load(r).get("tokens") or [])
    except Exception as e:  # noqa: BLE001
        print(f"[sidecar] tokenize for forcing failed ({type(e).__name__}); forcing disabled")
        _FORCE_IDS = []
    return _FORCE_IDS


def _force_actions(prompts, choices, n_eff, max_tokens):
    """Second pass for responses that never emitted a parseable action.

    Truncation severs the trailing <action> tag: measured 43% of responses at step 71,
    with valid_action_ratio falling to 0.72. Re-issuing the prefix with </think><action>
    appended guarantees the action exists, and caps total length at budget + tail.
    Returns (n_forced, forced_token_counts) and mutates `choices` in place.
    """
    if int(os.environ.get("ACG_FORCE_BUDGET", "0")) <= 0:
        return 0, 0
    tail = int(os.environ.get("ACG_FORCE_TAIL", "32"))
    todo = [k for k, c in enumerate(choices) if _needs_force(c.get("text", ""))]
    if not todo:
        return 0, 0
    fid = _force_ids()
    if not fid:
        return 0, 0
    bodies = []
    for k in todo:
        base = prompts[k // n_eff]
        prev = list(choices[k].get("token_ids") or [])
        # close the think block and open the action tag, then let it finish the action
        bodies.append((k, base, prev + fid))
    body = {
        "model": os.environ.get("ACG_SIDECAR_MODEL", "default"),
        "prompt": [b + p for _, b, p in bodies],
        "n": 1, "max_tokens": tail, "temperature": 0.0,
        "stop": ["</action>"], "include_stop_str_in_output": True,
        "return_token_ids": True,
    }
    try:
        req = urllib.request.Request(_url(), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=float(os.getenv("ACG_SIDECAR_TIMEOUT", "600"))) as r:
            out2 = json.load(r)
    except Exception as e:  # noqa: BLE001 - forcing is best-effort; never break a batch
        print(f"[sidecar] force pass failed ({type(e).__name__}: {e})")
        return 0, 0
    got = sorted(out2.get("choices", []), key=lambda c: c["index"])
    forced_tok = 0
    force_str = os.environ.get("ACG_FORCE_STR", "</think><action>")
    for (k, _, prev), c in zip(bodies, got):
        add = list(c.get("token_ids") or [])
        choices[k]["token_ids"] = prev + add          # prev already carries the forced ids
        choices[k]["text"] = choices[k].get("text", "") + force_str + c.get("text", "")
        forced_tok += len(add)
    return len(todo), forced_tok


def generate_via_sidecar(idx, attention_mask, eos_token_id, pad_token_id,
                         n, max_tokens, temperature, top_p, top_k, greedy):
    """Returns response tensor (bs*n, max_tokens) right-padded, or None on any failure."""
    if not _weights_fresh():
        print("[sidecar] weights not fresh within timeout; falling back to local generate")
        return None
    prompts = []
    for row, mask in zip(idx.tolist(), attention_mask.tolist()):
        prompts.append([t for t, m in zip(row, mask) if m == 1])  # strip left padding
    body = {
        "model": os.environ.get("ACG_SIDECAR_MODEL", "default"),
        "prompt": prompts,
        "n": 1 if greedy else int(n),
        # Budget forcing: cap the think block well below max_tokens so there is
        # always room to append an action; stop as soon as the action closes.
        "max_tokens": int(_think_budget(max_tokens)),
        "stop": ["</action>"],
        "include_stop_str_in_output": True,
        "temperature": 0.0 if greedy else float(temperature),
        "top_p": 1.0 if greedy else float(top_p),
        "top_k": -1 if greedy or int(top_k) <= 0 else int(top_k),
        "return_token_ids": True,
    }
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
    try:
        t0 = time.time()
        req = urllib.request.Request(_url(), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=float(os.getenv("ACG_SIDECAR_TIMEOUT", "600"))) as r:
            out = json.load(r)
        n_eff = body["n"]
        expect = len(prompts) * n_eff
        choices = sorted(out["choices"], key=lambda c: c["index"])
        _nf, _ftok = _force_actions(prompts, choices, n_eff, max_tokens)
        if _nf:
            print(f"[sidecar] forced action on {_nf}/{len(choices)} responses (+{_ftok} toks)")
        if len(choices) != expect:
            print(f"[sidecar] choice count {len(choices)} != expected {expect}; falling back")
            return None
        rows = []
        for c in choices:
            ids = list(c.get("token_ids") or [])
            if not ids and c.get("text"):
                print("[sidecar] server returned no token_ids; falling back")
                return None
            # guarantee eos before padding when the server stopped naturally
            if c.get("finish_reason") == "stop" and (not ids or ids[-1] not in eos_ids):
                ids.append(eos_ids[0])
            ids = ids[:max_tokens]
            rows.append(ids + [pad_token_id] * (max_tokens - len(ids)))
        gen_tok = sum(len([t for t in r_ if t != pad_token_id]) for r_ in rows)
        print(f"[sidecar] rank={_rank()} {len(prompts)}x{n_eff} -> {gen_tok} toks in {time.time()-t0:.1f}s")
        return torch.tensor(rows, dtype=idx.dtype, device=idx.device)
    except Exception as e:  # noqa: BLE001 - any sidecar failure must fall back, never crash training
        print(f"[sidecar] request failed ({type(e).__name__}: {e}); falling back to local generate")
        return None
