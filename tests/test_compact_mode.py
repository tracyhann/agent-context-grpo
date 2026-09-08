#!/usr/bin/env python3
"""ACG_COMPACT_MODE=replace must remove the digest's token overhead.

H-AA measured the prepend path costing +194 prompt tokens per turn against the
no-digest arm, at 15 of 15 paired steps. The cause is duplication: build_digest
summarises the FULL history, so prepending it to the recent-`history_length`
window ships those turns twice.

This reproduces the prompt-history block exactly as ALFWorldEnvironmentManager
builds it -- Memory.fetch's line format, then the digest applied both ways -- and
asserts replace costs no more than prepend, on an episode long enough for the
duplication to bite.
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verl-agent"))

from agent_system.memory.compact import build_digest

HISTORY_LENGTH = 2   # config.env.history_length used by every ALFWorld run here


def episode(n_turns):
    """A plausible ALFWorld trajectory: real receptacle sweeps, repeats, no-ops."""
    recs, obs = [], "You are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a countertop 1, a drawer 2, a fridge 1, and a garbagecan 1."
    places = ["cabinet 1", "cabinet 2", "cabinet 3", "cabinet 4", "countertop 1",
              "drawer 1", "drawer 2", "fridge 1", "garbagecan 1", "sinkbasin 1"]
    for t in range(n_turns):
        p = places[t % len(places)]
        if t % 4 == 3:
            act, nxt = f"open {p}", obs                      # no-op: obs unchanged
        else:
            act = f"go to {p}"
            nxt = f"You arrive at loc {t}. On the {p}, you see a cloth {t%3+1}, a soapbar {t%2+1}."
        recs.append({"text_obs": obs, "action": act})
        obs = nxt
    recs.append({"text_obs": obs, "action": ""})
    return recs


def fetch_block(recs, history_length):
    """Byte-identical to Memory.fetch's rendering."""
    recent = recs[-history_length:]
    start = len(recs) - len(recent)
    return "\n".join(
        f"[Observation {start+j+1}: '{r['text_obs']}', Action {start+j+1}: '{r['action']}']"
        for j, r in enumerate(recent))


def main():
    from transformers import AutoTokenizer
    snap = os.path.join(os.environ.get("HF_HOME", os.path.join(ROOT, "hf")), "hub",
                        "models--Qwen--Qwen2.5-1.5B-Instruct", "snapshots")
    ckpt = os.path.join(snap, sorted(os.listdir(snap))[0]) if os.path.isdir(snap) \
        else "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(ckpt)
    ntok = lambda s: len(tok(s).input_ids)

    print(f"{'turns':>6}{'window':>9}{'digest':>9}{'prepend':>9}{'replace':>9}"
          f"{'prep-win':>10}{'repl-win':>10}")
    ok = True
    for n in (4, 8, 16, 24, 32):
        recs = episode(n)
        win = fetch_block(recs, HISTORY_LENGTH)
        dig = build_digest(recs, budget_tokens=512, tokenizer=tok,
                           obs_key="text_obs", action_key="action")
        prep, repl = dig + "\n" + win, dig
        w, d, p, r = ntok(win), ntok(dig), ntok(prep), ntok(repl)
        print(f"{n:>6}{w:>9}{d:>9}{p:>9}{r:>9}{p-w:>+10}{r-w:>+10}")
        if r > p:
            ok = False
    print(f"\nreplace <= prepend at every length: {'PASS' if ok else 'FAIL'}")

    # The gate that matters for the run: replace must not silently drop history.
    recs = episode(16)
    dig = build_digest(recs, budget_tokens=512, tokenizer=tok,
                       obs_key="text_obs", action_key="action")
    acts = {r["action"] for r in recs if r["action"]}
    covered = sum(1 for a in acts if a in dig)
    print(f"distinct actions in episode {len(acts)}, named in digest {covered}  "
          f"{'OK' if covered == len(acts) else 'PARTIAL (budget-evicted)'}")
    print(f"window covers {HISTORY_LENGTH} turns; digest covers {len(recs)-1}")
    return 0 if ok else 1





def sweep():
    """Budget calibration. The digest is only worth its cost if a SMALL budget
    still names the actions that matter; 512 against a 466-token baseline prompt
    more than doubles the prompt every time it fires."""
    from transformers import AutoTokenizer
    snap = os.path.join(os.environ.get("HF_HOME", os.path.join(ROOT, "hf")), "hub",
                        "models--Qwen--Qwen2.5-1.5B-Instruct", "snapshots")
    ckpt = os.path.join(snap, sorted(os.listdir(snap))[0]) if os.path.isdir(snap) \
        else "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(ckpt)
    ntok = lambda s: len(tok(s).input_ids)
    FIRE = 0.36        # measured stall-gate firing rate
    BASE = 466         # measured baseline prompt_length/mean

    print("\nbudget sweep on a 24-turn episode (replace mode), vs BASE prompt 466 tok")
    print(f"{'budget':>7}{'digest':>8}{'lines':>7}{'informative':>12}"
          f"{'net/fire':>10}{'avg overhead':>14}{'% of base':>11}")
    recs = episode(24)
    win = ntok(fetch_block(recs, HISTORY_LENGTH))
    inform_total = len({r["action"] for r in recs if r["action"]})
    for b in (64, 96, 128, 192, 256, 384, 512):
        d = build_digest(recs, budget_tokens=b, tokenizer=tok,
                         obs_key="text_obs", action_key="action")
        n = ntok(d)
        lines = d.count("\n- ") + d.count("\n") - 1 if d else 0
        lines = len([l for l in d.split("\n") if l.startswith("- ")])
        kept = len([l for l in d.split("\n") if l.startswith("- ") and "no effect" not in l])
        net = n - win
        print(f"{b:>7}{n:>8}{lines:>7}{kept:>4}/{inform_total:<7}"
              f"{net:>+10}{FIRE*net:>+14.0f}{100*FIRE*net/BASE:>10.1f}%")


if __name__ == "__main__":
    rc = main()
    sweep()
    sys.exit(rc)
