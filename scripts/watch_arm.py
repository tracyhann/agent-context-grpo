#!/usr/bin/env python3
"""Emit one line per NEW held-out evaluation of a run, scored against the
pre-registered per-type baseline of the 79.7% arm (H-Y' in hypothesis.md).

Overall success is the weakest readout; the deciding signal is whether any gain
concentrates on the two memory-demanding types.
"""
import json, os, sys

# ccpo-global-ext, steps 105-145 pooled (9 evals). SE in parentheses.
REF = {"pick_two_obj_and_place": 69.6, "look_at_obj_in_light": 65.6,
       "pick_cool_then_place_in_recep": 78.2, "pick_clean_then_place_in_recep": 80.4,
       "pick_heat_then_place_in_recep": 87.1, "pick_and_place": 87.6}
MEMORY_TYPES = ("pick_two_obj_and_place", "look_at_obj_in_light")

path, seen_file = sys.argv[1], sys.argv[2]
seen = int(open(seen_file).read()) if os.path.exists(seen_file) else 0
rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
V = [r for r in rows if "val/success_rate" in r]
for r in V[seen:]:
    st = r.get("step", 0)
    o = 100 * r["val/success_rate"]
    parts = []
    for t, ref in sorted(REF.items(), key=lambda kv: kv[1]):
        k = f"val/{t}_success_rate"
        if k not in r:
            continue
        d = 100 * r[k] - ref
        tag = "*" if t in MEMORY_TYPES else " "
        parts.append(f"{tag}{t[:22]} {100*r[k]:.0f} ({d:+.0f})")
    print(f"step {st}: OVERALL {o:.1f}% ({o-79.7:+.1f} vs 79.7 arm) | "
          f"valid_act {r.get('episode/valid_action_ratio', float('nan')):.4f} | "
          f"prompt_len {r.get('prompt_length/mean', 0):.0f}")
    print("   per-type vs pre-registered ref (* = memory-demanding): " + "; ".join(parts))
open(seen_file, "w").write(str(len(V)))
