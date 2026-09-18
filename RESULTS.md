# Results

**Nothing in this repo has been run yet.** Fill the table in as runs complete on this
machine. Every entry must be a multi-seed mean ± std from `scripts/evaluate.sh`, never a
single evaluation and never a training-time endpoint — see §3 of
`experiments/experiments.md` for why.

## To fill in

ALFWorld and WebShop, 150 steps, 7B, held-out split, 5 seeds (997/101/3173/869/2917).

| Method | Benchmark | Steps | All (mean ± std) | Best seed | Date | Checkpoint |
|---|---|---|---|---|---|---|
| GRPO  | ALFWorld | 150 | | | | |
| GiGPO | ALFWorld | 150 | | | | |
| HGPO  | ALFWorld | 150 | | | | |
| G2PO  | ALFWorld | 150 | | | | |
| GRPO  | WebShop  | 150 | | | | |
| GiGPO | WebShop  | 150 | | | | |
| HGPO  | WebShop  | 150 | | | | |
| G2PO  | WebShop  | 150 | | | | |

Per-task columns (Pick / Look / Clean / Heat / Cool / Pick2) come out of
`logging/summarise_eval.py`. Each covers a handful of the 128 games and swings by ±10
between seeds while `All` moves ±1–3, so print them with that caveat or not at all.

## Published targets

From the HGPO paper, Table 1 (K=2, ALFWorld in-distribution, mean of 3 seeds):

| Method | 1.5B | 7B |
|---|---|---|
| HGPO  | 92.77 | 95.44 |
| GiGPO | 90.16 | 93.29 |
| GRPO  | 72.8  | 78.64 |
| G2PO  | 95.0  | — |

That paper's Table 7 reports the same methods on an **older verl-agent version** with
markedly lower values (HGPO 7B K=2: 91.15 vs 95.44). Framework version moves these results
by several points, so always state which table a target came from.

## Sanity checks from a different machine — NOT results of this repo

These were produced on another host, at **other step budgets**, with 3 seeds rather than 5.
They are here only so a run on this machine can be recognised as plausible or obviously
broken. Do not copy them into the table above, and do not present them as outputs of this
repo.

| Method | Size | Steps | Held-out (3 seeds) |
|---|---|---|---|
| HGPO  | 7B   | 160 | 96.09 ± 0.78 |
| HGPO  | 1.5B | 160 | 91.93 ± 0.90 |
| GiGPO | 1.5B | 150 | 86.46 ± 2.74 |
| G2PO  | 1.5B | 150 | 93.8 (single evaluation, no error bar) |

Two things they demonstrate that apply here:

- **Evaluation noise is large.** The same *frozen* GiGPO checkpoint scored 83.6 / 86.7 /
  89.1 across three seeds — a 5.5-point spread with no training variance at all.
- **A training-time endpoint can mislead badly.** HGPO 7B's own final evaluation read 92.2
  while the multi-seed mean of that same checkpoint was 96.09 — the endpoint understated
  the run by 3.9 points. Always score the checkpoint with `scripts/evaluate.sh`.
