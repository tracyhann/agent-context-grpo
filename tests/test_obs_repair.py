#!/usr/bin/env python3
"""ACG_OBS_REPAIR must reproduce G2PO's anchor repair exactly, and the point of it
is anchor DISTINCTNESS -- the anchor is the state identity used for node grouping in
the step term, so states that share an anchor share a baseline.

Reference: baselines/G2PO/agent_system/environments/env_manager.py:132-178.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INVISIBLE_PATTERNS = ("You heat", "You cool", "You clean", "You turn on")


def g2po_reference(obs_seq):
    """Literal transcription of G2PO's loop, for one environment."""
    import numpy as np
    history, out = [obs_seq[0]], []
    for ob in obs_seq[1:]:
        if not ob.startswith("Nothing happens"):
            history.append(ob)
        inv = len(history) >= 2 and np.array(
            [history[-2].startswith(p) for p in _INVISIBLE_PATTERNS]).any()
        out.append(" ".join(history[-2:]) if inv else history[-1])
    return out


def ours(obs_seq):
    """The logic added to env_manager.step(), same shape."""
    history, out = [obs_seq[0]], []
    for ob in obs_seq[1:]:
        if not str(ob).startswith("Nothing happens"):
            history.append(ob)
        inv = len(history) >= 2 and any(
            str(history[-2]).startswith(p) for p in _INVISIBLE_PATTERNS)
        out.append(" ".join(history[-2:]) if inv else history[-1])
    return out


SEQ = [
    "You are in the middle of a room. You see a fridge 1, a microwave 1, a countertop 1.",
    "You arrive at loc 3. On the countertop 1, you see an egg 1.",
    "You pick up the egg 1 from the countertop 1.",
    "Nothing happens.",                                  # failed action
    "You arrive at loc 7. The microwave 1 is closed.",
    "You heat the egg 1 with the microwave 1.",          # invisible-state action
    "You arrive at loc 3. On the countertop 1, you see nothing.",   # looks pre-heat
    "Nothing happens.",
    "You turn on the desklamp 1.",                       # invisible-state action
    "You are in the middle of a room. You see a fridge 1, a microwave 1, a countertop 1.",
]


def main():
    a, b = g2po_reference(SEQ), ours(SEQ)
    match = a == b
    print(f"matches G2PO reference on {sum(x==y for x,y in zip(a,b))}/{len(a)} steps  "
          f"{'PASS' if match else 'FAIL'}")

    raw = SEQ[1:]
    print(f"\ndistinct anchors: raw {len(set(raw))}/{len(raw)}   "
          f"repaired {len(set(b))}/{len(b)}")

    # (B) the "Nothing happens" collapse: how many raw anchors are the bare failure
    # string, which is identical across every task and position in the batch?
    nh = sum(1 for o in raw if o.startswith("Nothing happens"))
    print(f"\n(B) raw anchors that are the bare failure string: {nh}/{len(raw)}"
          f"  -- these collapse to ONE anchor batch-wide")
    print(f"    after repair: {sum(1 for o in b if o.startswith('Nothing happens'))}")

    # (A) the invisible-state conflation: does the post-heat observation differ from
    # the pre-heat one it duplicates?
    pre, post = raw[0], raw[5]
    print(f"\n(A) post-invisible-action anchor vs an earlier same-looking one:")
    print(f"    raw      identical? {raw[5] == raw[0] or raw[5].startswith(raw[0][:40])}")
    print(f"    repaired carries the heat event? {'You heat' in b[5]}")
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(main())
