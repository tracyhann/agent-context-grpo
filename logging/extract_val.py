#!/usr/bin/env python3
"""Pull val/* metrics out of a verl driver log.

verl pretty-prints the metrics dict as a WRAPPED STRING, so a key can sit on one line and
its value on the next, with stray quote characters at the line edges. A naive grep finds
only some keys. Join the lines first, then match.
"""
import re, sys
def parse(path):
    out = []
    for line in open(path, errors="ignore"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        line = re.sub(r"^\s*\(TaskRunner pid=\d+\)\s*", "", line).strip().strip('"').strip("'")
        out.append(line)
    txt = "".join(out)
    return {m.group(1): float(m.group(2))
            for m in re.finditer(r"val/([A-Za-z0-9_/]+)':?\s*np\.float64\(([0-9.]+)\)", txt)}
if __name__ == "__main__":
    d = parse(sys.argv[1])
    if len(sys.argv) > 2:
        for k in sorted(d): print(f"{k}\t{d[k]}")
    elif "success_rate" in d: print(d["success_rate"])
