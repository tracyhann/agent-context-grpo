#!/usr/bin/env python3
"""Convert the G2PO reference run's console log into our metrics.jsonl format.

Their tree has no ACG instrumentation, so it never writes metrics.jsonl. verl prints one
`step:N - key:value - key:value ...` line per iteration, which carries everything we need
for a baseline curve. This makes the reference arm readable by the same tooling as ours
(scripts/watch_arm.py, the plot watcher) instead of a bespoke path.

Usage: parse_g2po_log.py <train.log> <metrics.jsonl>
"""
import json
import re
import sys


def parse(line):
    if not line.startswith("step:"):
        return None
    out = {}
    for part in line.strip().split(" - "):
        k, _, v = part.partition(":")
        k, v = k.strip(), v.strip()
        if not k or not v:
            continue
        if k == "step":
            try:
                out["step"] = int(float(v))
            except ValueError:
                return None
            continue
        try:
            out[k] = float(v)
        except ValueError:
            pass          # non-numeric metric, not needed for the curve
    return out if "step" in out else None


def main():
    src, dst = sys.argv[1], sys.argv[2]
    rows, seen = [], set()
    with open(src, errors="replace") as f:
        for line in f:
            # ray prefixes worker output; the step line may be embedded
            i = line.find("step:")
            if i < 0:
                continue
            r = parse(line[i:])
            # keep the LAST record per step: verl prints the val metrics on the
            # same line as the train ones, but a step can appear more than once
            # in the log when ray echoes it.
            if r and (r["step"] not in seen or len(r) > 3):
                seen.add(r["step"])
                rows = [x for x in rows if x["step"] != r["step"]] + [r]
    rows.sort(key=lambda r: r["step"])
    with open(dst, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    vals = [r for r in rows if "val/success_rate" in r]
    print(f"{len(rows)} steps, {len(vals)} evaluations -> {dst}")
    if vals:
        print("  last:", {k: v for k, v in vals[-1].items()
                          if k == "step" or "success" in k})


if __name__ == "__main__":
    main()
