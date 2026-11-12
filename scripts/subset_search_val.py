#!/usr/bin/env python3
"""Write a per-dataset subset of the Search-R1 test parquet for in-training validation.

The processed test split holds 51,713 questions from seven datasets, and the
trainer validates on the whole file every `test_freq` steps. This keeps the first
N rows of every `data_source`, in file order, so periodic validation stays cheap.
Score the final checkpoint on the full file.

    python3 scripts/subset_search_val.py --per-source 500
    python3 scripts/prepare_search_r1_ccpo.py ... --val-files envdata/searchr1/test_500_per_source.parquet
"""
import argparse
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def subset(source, destination, per_source):
    import pyarrow as pa
    import pyarrow.parquet as pq

    if per_source < 1:
        raise ValueError('per_source must be positive')
    table = pq.read_table(source)
    if 'data_source' not in table.column_names:
        raise ValueError(f'{source} has no data_source column')
    kept, seen = [], Counter()
    for row, name in enumerate(table.column('data_source').to_pylist()):
        if seen[name] < per_source:
            seen[name] += 1
            kept.append(row)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    pq.write_table(table.take(pa.array(kept, type=pa.int64())), destination)
    return dict(seen), table.num_rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', default=str(ROOT/'envdata/searchr1/test.parquet'))
    p.add_argument('--per-source', type=int, required=True)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    out = a.out or str(Path(a.source).with_name(f'test_{a.per_source}_per_source.parquet'))
    counts, total = subset(a.source, out, a.per_source)
    for name, count in sorted(counts.items()):
        print(f'  {name}: {count}')
    print(f'{sum(counts.values())} of {total} rows -> {out}')


if __name__ == '__main__':
    main()
