#!/usr/bin/env python3
"""Build a validation parquet sized so one pass covers the split exactly once.

    census/make_val_parquet.py alfworld <src_dir> <out_dir>    # 140 rows
    census/make_val_parquet.py webshop  <src_dir> <out_dir>    # 500 rows

verl derives the number of validation resets from len(val_dataset)/val_batch_size.
The rows themselves are placeholders for these agent environments -- the prompt
column is empty and the task comes from the env -- so only the COUNT matters, and
it has to equal the number of held-out tasks:

  ALFWorld  140 games in eval_in_distribution, val_batch_size=140 -> 1 reset
  WebShop   500 goals (envs.py: indices 0-499), val_batch_size=125 -> 4 resets

Pair this with ACG_ALF_CENSUS / ACG_WS_CENSUS, which pin WHICH task each env gets;
this file only fixes HOW MANY are played.
"""
import os
import shutil
import sys

import pandas as pd

SIZES = {"alfworld": 140, "webshop": 500}


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in SIZES:
        raise SystemExit(__doc__)
    bench, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    n = SIZES[bench]
    os.makedirs(out, exist_ok=True)
    d = pd.read_parquet(os.path.join(src, "test.parquet"))
    reps = -(-n // len(d))                      # ceil
    pd.concat([d] * reps, ignore_index=True).head(n).to_parquet(
        os.path.join(out, "test.parquet"))
    shutil.copy(os.path.join(src, "train.parquet"), os.path.join(out, "train.parquet"))
    print(f"  {out}/test.parquet: {n} rows (from {len(d)}), train.parquet copied")


if __name__ == "__main__":
    main()
