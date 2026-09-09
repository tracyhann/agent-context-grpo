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
    return match





def test_live_path():
    """Exercise the REAL AlfWorldEnvironmentManager.reset/step with a stub env.

    ast.parse cannot catch an ordering bug -- `self.history_obs = [[o] for o in
    full_text_obs]` placed before `full_text_obs` is assigned parses fine and raises
    NameError only at runtime. This calls the actual code path so that class of bug
    fails here instead of 8 minutes into a GPU run.
    """
    sys.path.insert(0, "/workspace"); sys.path.insert(0, "/workspace/verl-agent")
    sys.path.insert(0, "/workspace/docker/fa_stub")
    from agent_system.environments.env_manager import AlfWorldEnvironmentManager

    ADMISSIBLE = [["go to cabinet 1", "open fridge 1", "help"]]
    SEQ = ["You are in a room. Your task is to: heat an egg.",
           "You arrive at loc 1. On the cabinet 1, you see an egg 1.",
           "Nothing happens.",
           "You heat the egg 1 with the microwave 1.",
           "You arrive at loc 1. On the cabinet 1, you see an egg 1."]

    class StubEnvs:
        def __init__(self): self.t = 0
        @property
        def get_admissible_commands(self): return list(ADMISSIBLE)
        def reset(self):
            self.t = 0
            return [SEQ[0]], [None], [{"extra.gamefile": "g"}]
        def step(self, actions):
            self.t += 1
            return ([SEQ[min(self.t, len(SEQ)-1)]], [None], [0.0], [False],
                    [{"extra.gamefile": "g", "won": False}])

    class Cfg(dict):
        def __getattr__(self, k): return self[k]
    cfg = Cfg(env=Cfg(history_length=2, rollout=Cfg(n=1)), data=Cfg())

    mgr = AlfWorldEnvironmentManager.__new__(AlfWorldEnvironmentManager)
    mgr.envs, mgr.config = StubEnvs(), cfg
    mgr.projection_f = lambda acts, adm: (list(acts), [1] * len(acts))
    from agent_system.memory.memory import SimpleMemory
    mgr.memory = SimpleMemory()

    results = {}
    for flags in (("0", "0"), ("1", "0"), ("1", "1")):
        os.environ["ACG_OBS_REPAIR"], os.environ["ACG_ANCHOR_AFF"] = flags
        obs, _ = mgr.reset(None)
        anchors = []
        for _ in range(4):
            nxt, _r, _d, _i = mgr.step(["go to cabinet 1"])
            anchors.append(nxt["anchor"][0])
        results[flags] = anchors
        print(f"  OBS_REPAIR={flags[0]} ANCHOR_AFF={flags[1]}: "
              f"{len(set(anchors))} distinct anchors over 4 steps")

    base, rep, both = results[("0","0")], results[("1","0")], results[("1","1")]
    ok = True
    nh = sum(1 for a in base if str(a).startswith("Nothing happens"))
    print(f"  baseline anchors that are the bare failure string: {nh}")
    print(f"  with repair:                                       "
          f"{sum(1 for a in rep if str(a).startswith('Nothing happens'))}")
    if nh == 0: print("  (stub produced no failure turn -- weak check)")
    if sum(1 for a in rep if str(a).startswith("Nothing happens")) > 0:
        ok = False; print("  FAIL: repair left a bare failure anchor")
    if not any("Admissible actions" in str(a) for a in both):
        ok = False; print("  FAIL: ANCHOR_AFF did not fold admissible actions in")
    if any("Admissible actions" in str(a) for a in rep):
        ok = False; print("  FAIL: admissible actions leaked in without ANCHOR_AFF")
    for k in ("ACG_OBS_REPAIR", "ACG_ANCHOR_AFF"): os.environ.pop(k, None)
    print(f"  live-path test {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ok1 = main()
    print("\nlive code path (stub env):")
    ok2 = test_live_path()
    sys.exit(0 if (ok1 and ok2) else 1)
