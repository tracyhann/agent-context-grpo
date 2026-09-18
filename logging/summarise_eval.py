#!/usr/bin/env python3
"""Print the per-seed table and mean ± std for an evaluation directory."""
import sys, os, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_val import parse
KEYS = [("Pick","pick_and_place"),("Look","look_at_obj_in_light"),("Clean","pick_clean_then_place_in_recep"),
        ("Heat","pick_heat_then_place_in_recep"),("Cool","pick_cool_then_place_in_recep"),
        ("Pick2","pick_two_obj_and_place")]
root = sys.argv[1]
seeds = sorted(d for d in os.listdir(root) if d.startswith("seed-")) if os.path.isdir(root) else []
cols = {k: [] for k, _ in KEYS}; alls = []; rows = []
for s in seeds:
    log = os.path.join(root, s, "outputs", "train.log")
    d = parse(log) if os.path.exists(log) else {}
    if "success_rate" not in d: rows.append((s.replace("seed-",""), ["n/a"]*7)); continue
    cells = []
    for k, key in KEYS:
        v = d.get(f"{key}_success_rate")
        if v is None: cells.append("n/a")
        else: cols[k].append(100*v); cells.append(f"{100*v:.2f}")
    alls.append(100*d["success_rate"]); cells.append(f"{100*d['success_rate']:.2f}")
    rows.append((s.replace("seed-",""), cells))
ms = lambda xs: f"{st.mean(xs):.2f} ± {st.stdev(xs):.2f}" if len(xs) > 1 else (f"{xs[0]:.2f}" if xs else "—")
hdr = ["seed"] + [k for k, _ in KEYS] + ["All"]
w = 13
sep = lambda l,m,r: print(l + m.join("─"*(w+2) for _ in hdr) + r)
row = lambda c: print("│ " + " │ ".join(f"{x:<{w}}" for x in c) + " │")
print(f"\n{os.path.basename(root)}")
sep("┌","┬","┐"); row(hdr); sep("├","┼","┤")
for s, c in rows: row([s] + c)
sep("├","┼","┤"); row(["mean ± std"] + [ms(cols[k]) for k, _ in KEYS] + [ms(alls)]); sep("└","┴","┘")
print("\nQuote the All mean ± std. Per-task columns cover a handful of the 128 games each")
print("and swing by ±10 between seeds; they are indicative only.")
