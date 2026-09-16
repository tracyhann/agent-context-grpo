#!/usr/bin/env python3
"""Per-task-type held-out results in the published-table layout.

    scripts/report_results.py --exp experiments/ccpo-attncred-20260912
    scripts/report_results.py --exp <a> --exp <b> --markdown
    scripts/report_results.py --all --rows best,last,100

Prints one block per arm with a row at the BEST evaluation, the LAST evaluation, and
step 100, so an arm can be read against the published tables (G2PO / GiGPO / HGPO),
which quote a single number per task type.

READING RULES, because a single row here is weaker than it looks:

* Held-out evaluation is stochastic by design (T=0.4, 128 of 140 games), so one
  evaluation of the whole split carries SE ~ +/-3.3 points.
* Each task type is only ~20 of those 128 games. One or two episodes move a type by
  5 points. Per-type rows are texture, not findings.
* BEST is the maximum of a noisy series and is biased upward by roughly 1.5 sd
  (H-Z). It is reported because the published tables do, not because it is this
  project's preferred statistic -- that is the 70-100 window mean, printed under
  each block.
* `All` is val/success_rate, the trainer's own overall. The per-type columns are
  means over validation batches rather than a pooled per-game rate, so they need not
  average exactly to `All`.
"""

import argparse
import glob
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (column header, metrics.jsonl suffix) in the order the published tables use
TYPES = [
    ("Pick", "pick_and_place"),
    ("Look", "look_at_obj_in_light"),
    ("Clean", "pick_clean_then_place_in_recep"),
    ("Heat", "pick_heat_then_place_in_recep"),
    ("Cool", "pick_cool_then_place_in_recep"),
    ("Pick2", "pick_two_obj_and_place"),
]


# Behaviour columns: (header, metrics key, scale). These are TRAINING rollouts --
# the trainer logs none of them on the validation path today, so they describe how
# the agent behaves while learning, not on the held-out split. Keys that do not
# exist yet print "--" and start populating on their own once they are logged:
#   admissible_action_ratio / inadmissible_unpenalised_ratio  landed 2026-09-13
#     (commit 4bbd026), so runs launched before that show "--"
#   noop_ratio / revisit_ratio / unique_per_turn / term_capped  not yet implemented
BEHAV = [
    ("Turns", "episode/length/mean", 1),
    ("TrainSuc", "episode/success_rate", 100),
    ("Valid%", "episode/valid_action_ratio", 100),
    ("Adm%", "episode/admissible_action_ratio", 100),
    ("NoOp%", "episode/noop_ratio", 100),
    ("Revisit%", "episode/revisit_ratio", 100),
    ("RespTok", "response_length/mean", 1),
    ("Trunc%", "response_length/clip_ratio", 100),
]


# ---------------------------------------------------------------------------
# Paper table: the same columns for TRAINING rollouts and the HELD-OUT split, at
# step 100 / best / final. experiments/experiments.md section 5 is the spec.
#
# Every key below is already logged by the trainer; nothing here derives a number.
# The only one that is new is val/length/mean (held-out turns) -- runs launched
# before it landed print "--" for that cell.
# ---------------------------------------------------------------------------
ALFWORLD_COLS = [(h, f"{k}_success_rate", 100) for h, k in TYPES] + [
    ("All", "success_rate", 100),
    ("Turns", "length/mean", 1),
]
WEBSHOP_COLS = [
    ("Success", "success_rate", 100),
    ("Score", "webshop_task_score (not success_rate)", 100),
    ("Turns", "length/mean", 1),
]


def benchmark_of(keys):
    """Which table to print, read off the metrics the run actually logged."""
    if any("webshop_task_score" in k for k in keys):
        return "webshop"
    if any(k.startswith("val/pick_and_place") for k in keys):
        return "alfworld"
    return None


def load_paper(exp_dir):
    """(per-step train rows, per-step held-out rows, benchmark)."""
    p = os.path.join(exp_dir, "outputs", "metrics.jsonl")
    if not os.path.exists(p):
        return {}, {}, None
    train, held, keys = {}, {}, set()
    with open(p) as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            keys |= set(r)
            step = i + 1
            train[step] = {k[len("episode/"):]: v for k, v in r.items()
                           if k.startswith("episode/")}
            if r.get("val/success_rate") is not None:
                held[step] = {k[len("val/"):]: v for k, v in r.items()
                              if k.startswith("val/")}
    return train, held, benchmark_of(keys)


def paper_table(exp_dir, wanted, markdown=False):
    name = os.path.basename(exp_dir.rstrip("/"))
    train, held, bench = load_paper(exp_dir)
    if not held:
        print(f"{name}: no validation rows in metrics.jsonl\n")
        return
    if bench is None:
        print(f"{name}: cannot tell which benchmark this is from its metrics\n")
        return
    cols = ALFWORLD_COLS if bench == "alfworld" else WEBSHOP_COLS
    evals = [(s, {"All": held[s]["success_rate"] * 100}) for s in sorted(held)]
    picked = pick_rows(evals, wanted)

    def cell(row, key, scale):
        v = None if row is None else row.get(key)
        return "--" if v is None else f"{v * scale:.1f}"

    heads = [h for h, _, _ in cols]
    title = "ALFWorld, held-out `valid_seen`" if bench == "alfworld" else "WebShop"
    lines = []
    for label, step, _ in picked:
        label = label.replace("last @", "final @")
        for split, src in (("train", train.get(step)), ("held-out", held.get(step))):
            lines.append((label, split, [cell(src, k, sc) for _, k, sc in cols]))

    if markdown:
        print(f"**{name}** — {title}\n")
        print("| checkpoint | split | " + " | ".join(heads) + " |")
        print("|---" * (len(heads) + 2) + "|")
        for label, split, cells in lines:
            print(f"| {label} | {split} | " + " | ".join(cells) + " |")
        print()
    else:
        print(f"{name}  —  {title}")
        print(" " * 24 + "".join(h.rjust(8) for h in heads))
        for label, split, cells in lines:
            print(f"  {label:<13}{split:<9}" + "".join(c.rjust(8) for c in cells))
        print()
    note = ("train rows are that step's rollout batch (16 tasks x 8), not a checkpoint "
            "evaluation;\nheld-out rows are the 128-episode validation draw at the same step.")
    print(note if not markdown else f"_{note}_\n")


def load(exp_dir):
    """(evaluations, behaviour) keyed by 1-based step, as the trainer counts them."""
    p = os.path.join(exp_dir, "outputs", "metrics.jsonl")
    if not os.path.exists(p):
        return [], {}
    evals, behav = [], {}
    with open(p) as fh:
        for i, line in enumerate(fh):
            r = json.loads(line)
            step = i + 1
            behav[step] = {h: (r[k] * s if r.get(k) is not None else None)
                           for h, k, s in BEHAV}
            if r.get("val/success_rate") is None:
                continue
            row = {h: r.get(f"val/{k}_success_rate") for h, k in TYPES}
            row["All"] = r["val/success_rate"]
            evals.append((step, {k: (v * 100 if v is not None else None)
                                 for k, v in row.items()}))
    return evals, behav


def fmt(row, cols, width=8):
    cells = []
    for c in cols:
        v = row.get(c)
        cells.append("--".rjust(width) if v is None else f"{v:.1f}".rjust(width))
    return "".join(cells)


def pick_rows(evals, wanted):
    """[(label, step, row)] for the requested selectors."""
    by_step = dict(evals)
    out = []
    for w in wanted:
        if w == "best":
            step, row = max(evals, key=lambda e: (e[1]["All"], -e[0]))
            out.append((f"best @{step}", step, row))
        elif w == "last":
            step, row = evals[-1]
            out.append((f"last @{step}", step, row))
        else:
            n = int(w)
            if n in by_step:
                out.append((f"step {n}", n, by_step[n]))
            else:
                out.append((f"step {n}", n, None))
    return out


def behav_cell(v, header):
    if v is None:
        return "--"
    return f"{v:.0f}" if header == "RespTok" else f"{v:.1f}"


def report(exp_dir, wanted, markdown=False, behaviour=True):
    name = os.path.basename(exp_dir.rstrip("/"))
    evals, behav = load(exp_dir)
    if not evals:
        print(f"{name}: no validation rows in metrics.jsonl\n")
        return
    cols = [h for h, _ in TYPES] + ["All"]
    rows = pick_rows(evals, wanted)
    window = [r["All"] for s, r in evals if s >= 70]
    bcols = [h for h, _, _ in BEHAV]

    if markdown:
        print(f"**{name}** — ALFWorld, held-out `valid_seen`\n")
        print("| | " + " | ".join(cols) + " |")
        print("|---" * (len(cols) + 1) + "|")
        for label, _, row in rows:
            if row is None:
                print(f"| {label} | " + " | ".join(["n/a"] * len(cols)) + " |")
            else:
                print(f"| {label} | " + " | ".join(
                    "--" if row[c] is None else f"{row[c]:.1f}" for c in cols) + " |")
        print()
    else:
        head = "".join(c.rjust(8) for c in cols)
        print(f"{name}")
        print(" " * 14 + "ALFWorld".center(len(head)).rstrip())
        print(" " * 14 + head)
        for label, _, row in rows:
            if row is None:
                print(f"  {label:<12}" + "".join("n/a".rjust(8) for _ in cols))
            else:
                print(f"  {label:<12}" + fmt(row, cols))
        print()

    if behaviour:
        if markdown:
            print("behaviour at the same steps (training rollouts, not held-out)\n")
            print("| | " + " | ".join(bcols) + " |")
            print("|---" * (len(bcols) + 1) + "|")
            for label, step, _ in rows:
                b = behav.get(step)
                cells = (["n/a"] * len(bcols) if b is None
                         else [behav_cell(b[c], c) for c in bcols])
                print(f"| {label} | " + " | ".join(cells) + " |")
            print()
        else:
            print("  behaviour at the same steps (training rollouts, not held-out)")
            print(" " * 14 + "".join(c.rjust(9) for c in bcols))
            for label, step, _ in rows:
                b = behav.get(step)
                if b is None:
                    print(f"  {label:<12}" + "".join("n/a".rjust(9) for _ in bcols))
                else:
                    print(f"  {label:<12}" + "".join(
                        behav_cell(b[c], c).rjust(9) for c in bcols))
            print()

    if window:
        extra = (f"  window 70-100 mean {st.mean(window):.2f} over {len(window)} evals"
                 f"   |   all {len(evals)} evals {st.mean([r['All'] for _, r in evals]):.2f}")
        print(extra if not markdown else f"_{extra.strip()}_\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", action="append", default=[], help="experiment directory (repeatable)")
    ap.add_argument("--all", action="store_true", help="every experiment with >=1 evaluation")
    ap.add_argument("--rows", default="best,last,100",
                    help="comma-separated: best, last, or a step number (default best,last,100)")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--no-behaviour", action="store_true",
                    help="success table only; omit the behaviour block")
    ap.add_argument("--paper-table", action="store_true",
                    help="train and held-out, same columns, at step 100 / best / final "
                         "-- the layout experiments.md section 5 asks for")
    a = ap.parse_args()

    exps = [e if os.path.isabs(e) else os.path.join(ROOT, e) for e in a.exp]
    if a.all:
        exps = sorted(os.path.dirname(os.path.dirname(p))
                      for p in glob.glob(os.path.join(ROOT, "experiments", "*", "outputs", "metrics.jsonl")))
    if not exps:
        sys.exit("pass --exp <dir> (repeatable) or --all")

    wanted = [w.strip() for w in a.rows.split(",") if w.strip()]
    if a.paper_table:
        if a.rows == ap.get_default("rows"):
            wanted = ["100", "best", "last"]
        for e in exps:
            paper_table(e, wanted, a.markdown)
        return

    for e in exps:
        report(e, wanted, a.markdown, behaviour=not a.no_behaviour)

    print("best is the max of a noisy series (biased up ~1.5 sd, H-Z); one evaluation of the"
          "\nfull split carries SE ~+/-3.3, and each task type is only ~20 of the 128 games."
          "\nbehaviour columns are TRAINING rollouts -- the trainer logs none of them on the"
          "\nvalidation path, so they do not describe the held-out episodes above. '--' means"
          "\nthe key was not logged by that run; see BEHAV in this file for when each landed.")


if __name__ == "__main__":
    sys.exit(main())
