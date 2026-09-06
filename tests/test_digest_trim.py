#!/usr/bin/env python3
"""Unit test for the digest delimiters and the prompt trim loop.

The trim loop itself lives in TrajectoryCollector (rollout_loop.py) where it
needs a full trainer config; this reproduces its string surgery verbatim on a
real digest with a real tokenizer, which is the part that can silently corrupt
the prompt. Run on CPU inside acg_persist.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "verl-agent"))
sys.path.insert(0, os.path.join(ROOT, "docker", "fa_stub"))
# The Rust tokenizer builds a rayon pool sized from nproc; on a box whose cgroup
# pid budget is already spent by a training run that fails outright.
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from agent_system.memory.compact import (DIGEST_FOOTER, DIGEST_HEADER,  # noqa: E402
                                         build_digest)

def _model_path():
    """The base model this repo trains, or ACG_TEST_CKPT to point elsewhere."""
    env = os.environ.get("ACG_TEST_CKPT")
    if env:
        return env
    snap = os.path.join(os.environ.get("HF_HOME", os.path.join(ROOT, "hf")), "hub",
                        "models--Qwen--Qwen2.5-1.5B-Instruct", "snapshots")
    if os.path.isdir(snap):
        return os.path.join(snap, sorted(os.listdir(snap))[0])
    return "Qwen/Qwen2.5-1.5B-Instruct"


MODEL = _model_path()


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)

    records = []
    for i in range(40):
        records.append({"text_obs": f"You are in the middle of a room {i%4}. "
                                    f"You see a cabinet {i}, a drawer {i}.",
                        "action": f"open drawer {i}"})
    digest = build_digest(records, budget_tokens=512, tokenizer=tok)
    assert digest.startswith(DIGEST_HEADER), "header missing"
    assert digest.rstrip().endswith(DIGEST_FOOTER), "footer missing"
    n0 = len(digest.split("\n"))
    print(f"digest built: {len(tok(digest).input_ids)} tokens, {n0} lines")

    obs_content = ("Task: put a clean apple in the fridge.\n" + digest +
                   "\nRecent: you opened drawer 39.\nCurrent observation: "
                   "the drawer is empty.\nAdmissible: 'go to fridge 1'")
    base = len(tok(obs_content).input_ids)

    for cap in (base - 5, base // 2, 60):
        content, lines = obs_content, None
        pre, tail = content.split(DIGEST_HEADER, 1)
        body, post = tail.split(DIGEST_FOOTER, 1)
        lines = [l for l in body.split("\n") if l]
        while lines and len(tok(content).input_ids) > cap:
            lines.pop()
            blk = (DIGEST_HEADER + "\n".join(lines) + "\n" + DIGEST_FOOTER
                   if lines else "")
            content = pre + blk + post
        got = len(tok(content).input_ids)
        # the parts that must never be trimmed
        for must in ("Task: put a clean apple in the fridge.",
                     "Current observation: the drawer is empty.",
                     "Admissible: 'go to fridge 1'"):
            assert must in content, f"trimming destroyed: {must!r}"
        print(f"cap {cap:>5}: {base} -> {got} tokens, digest {n0} -> "
              f"{len(lines)} lines, task+obs+admissible intact "
              f"{'(floor reached, digest empty)' if not lines else ''}")

    print("\nPASS: delimiters hold and the trim loop never touches "
          "task, current observation or admissible actions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
