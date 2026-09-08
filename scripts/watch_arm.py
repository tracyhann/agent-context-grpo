#!/usr/bin/env python3
"""Emit one line per NEW held-out evaluation, scored STEP-MATCHED against the
79.7% arm rather than against its converged endpoint.

Scoring an early step against a converged reference produces meaningless -70
deltas; the only fair comparison at step N is the base arm's own step N. Where
the base arm has no evaluation at that step, no delta is printed.

The deciding readout is the per-type split (H-Y'), not overall success, whose SE
at n=128 is ~3.6 points -- too wide to separate the hypotheses on its own.
"""
import json, math, os, sys

BASE = "/workspace/experiments/ccpo-global-20260907/outputs/metrics.jsonl"
MEMORY_TYPES = ("pick_two_obj_and_place", "look_at_obj_in_light")


def evals(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if '"val/success_rate"' in l]


def se(p, n=128):
    return 100 * math.sqrt(max(p * (1 - p), 1e-9) / n)


path, seen_file = sys.argv[1], sys.argv[2]
base = {r.get("step"): r for r in evals(BASE)}
seen = int(open(seen_file).read()) if os.path.exists(seen_file) else 0
V = evals(path)

for r in V[seen:]:
    st = r.get("step", 0)
    o = 100 * r["val/success_rate"]
    b = base.get(st)
    if b is None:
        head = f"step {st}: OVERALL {o:.1f}%  (no step-matched base eval)"
    else:
        ob = 100 * b["val/success_rate"]
        d = o - ob
        # SE of a difference of two independent proportions at n=128 each.
        sd = math.hypot(se(r["val/success_rate"]), se(b["val/success_rate"]))
        verdict = "within noise" if abs(d) < 2 * sd else "SIGNIFICANT"
        head = (f"step {st}: OVERALL {o:.1f}% vs base {ob:.1f}% "
                f"({d:+.1f}, +/-{2*sd:.1f} 2SE -> {verdict})")
    print(f"{head} | valid_act {r.get('episode/valid_action_ratio', float('nan')):.4f} "
          f"| prompt_len {r.get('prompt_length/mean', 0):.0f} "
          f"| train {100*r.get('episode/success_rate', float('nan')):.1f}%")

    if b is not None:
        parts = []
        for t in sorted(MEMORY_TYPES) + sorted(
                k[4:-13] for k in r
                if k.startswith("val/") and k.endswith("_success_rate")
                and k != "val/success_rate" and "best" not in k
                and k[4:-13] not in MEMORY_TYPES):
            k = f"val/{t}_success_rate"
            if k in r and k in b:
                tag = "*" if t in MEMORY_TYPES else " "
                parts.append(f"{tag}{t[:22]} {100*r[k]:.0f} vs {100*b[k]:.0f}"
                             f" ({100*(r[k]-b[k]):+.0f})")
        print("   per-type vs step-matched base (* = memory-demanding, the "
              "pre-registered ones): " + "; ".join(parts))
open(seen_file, "w").write(str(len(V)))
