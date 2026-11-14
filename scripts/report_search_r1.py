#!/usr/bin/env python3
"""Search-R1 exact match per dataset, read from a run's validation metrics.

    python3 scripts/report_search_r1.py experiments/search-ccpo-... experiments/search-grpo-...
    python3 scripts/report_search_r1.py <run> <run> --step 200 --markdown

One row per run at one evaluation (the last by default): exact match in percent on
every dataset, `Avg` and `All`.

* Every number is the trainer's rate over questions, `val/<dataset>_success_rate/pooled`.
  `val/<dataset>_success_rate` is not read: it is the mean of the validation batches'
  means, so the few questions of a dataset that share a batch with another dataset
  weigh as much as a full batch.
* `Avg` is the unweighted mean over the datasets. The average column of Search-R1's
  tables equals that mean of its dataset columns; neither that paper nor GiGPO's
  defines the column. `All` is the rate over every question; the large datasets
  dominate it.
* Runs are compared only when they were validated on the same number of questions
  per dataset.
"""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
# data_source values of PeterJinGo/nq_hotpotqa_train's test split, in the column
# order of the published tables. Any other data source follows, under its own name.
DATASETS = [('nq', 'NQ'), ('triviaqa', 'TriviaQA'), ('popqa', 'PopQA'), ('hotpotqa', 'HotpotQA'),
            ('2wikimultihopqa', '2Wiki'), ('musique', 'MuSiQue'), ('bamboogle', 'Bamboogle')]
POOLED = re.compile(r'^val/(.+)_success_rate/pooled$')
OVERALL = 'val/success_rate'


def evaluations(run):
    """{step: metrics row} for the steps that logged pooled validation rates."""
    path = Path(run)/'outputs/metrics.jsonl'
    if not path.is_file():
        raise FileNotFoundError(f'No metrics at {path}')
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    found = {int(r['step']): r for r in rows if r.get(f'{OVERALL}/pooled') is not None}
    if not found:
        raise ValueError(f'{path} has no pooled validation rates. They are logged by the current '
                         'trainer overlay; run scripts/sync_patches.sh before training.')
    return found


def summarize(row):
    """Per-dataset rate and question count, their unweighted mean, and the overall rate."""
    rates, questions = {}, {}
    for key, value in row.items():
        match = POOLED.match(key)
        if match and value is not None:
            rates[match.group(1)] = float(value)
            questions[match.group(1)] = int(row[key[:-len('pooled')] + 'episodes'])
    if not rates:
        raise ValueError('The evaluation has no per-dataset rates')
    known = [name for name, _ in DATASETS if name in rates]
    order = known + sorted(set(rates) - set(known))
    return dict(order=order, rates=rates, questions=questions,
                avg=sum(rates.values()) / len(rates), all=float(row[f'{OVERALL}/pooled']),
                all_questions=int(row[f'{OVERALL}/episodes']))


def collect(runs, step=None):
    """[(run name, step, summary)], refusing runs validated on different question sets."""
    out = []
    for run in runs:
        found = evaluations(run)
        chosen = max(found) if step is None else step
        if chosen not in found:
            raise ValueError(f'{run} has no evaluation at step {chosen}; it has {sorted(found)}')
        out.append((Path(run).name, chosen, summarize(found[chosen])))
    first = out[0]
    for name, _, summary in out[1:]:
        if summary['questions'] != first[2]['questions']:
            raise ValueError(f'{name} and {first[0]} were validated on different questions: '
                             f'{summary["questions"]} against {first[2]["questions"]}')
    return out


def render(results, markdown=False):
    summary = results[0][2]
    labels = dict(DATASETS)
    heads = [labels.get(name, name) for name in summary['order']] + ['Avg', 'All']
    lines = [(f'{name} @{step}', [f'{100 * s["rates"][d]:.1f}' for d in s['order']]
              + [f'{100 * s["avg"]:.1f}', f'{100 * s["all"]:.1f}']) for name, step, s in results]
    lines.append(('questions', [str(summary['questions'][d]) for d in summary['order']]
                  + ['', str(summary['all_questions'])]))
    if markdown:
        text = ['| run | ' + ' | '.join(heads) + ' |', '|---' * (len(heads) + 1) + '|']
        text += [f'| {label} | ' + ' | '.join(cells) + ' |' for label, cells in lines]
    else:
        width = max(len(label) for label, _ in lines) + 2
        text = [' ' * width + ''.join(h.rjust(11) for h in heads)]
        text += [label.ljust(width) + ''.join(c.rjust(11) for c in cells) for label, cells in lines]
    return '\n'.join(text) + '\n'


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+', help='experiment directories')
    p.add_argument('--step', type=int, default=None, help='evaluation step; default: each run\'s last')
    p.add_argument('--markdown', action='store_true')
    a = p.parse_args(argv)
    runs = [r if Path(r).is_absolute() else ROOT/r for r in a.runs]
    try:
        print(render(collect(runs, a.step), a.markdown), end='')
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))


if __name__ == '__main__':
    main()
