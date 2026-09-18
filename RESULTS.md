# Reference results measured on this host

ALFWorld, in-distribution held-out split (`eval_in_distribution`), 128 games, T=0.4 with
sampling, `history_length=2` (K=2). Percentages are overall success rate.

## Multi-seed evaluations of frozen checkpoints

Seeds 997 / 101 / 3173, evaluated with `trainer.val_only=True` on the final checkpoint.
These are the numbers to quote: a single evaluation is far too noisy.

| Run | Step | seed 997 | seed 101 | seed 3173 | **mean ± std** | Published (K=2) |
|---|---|---|---|---|---|---|
| HGPO 7B    | 160 | 96.09 | 95.31 | 96.88 | **96.09 ± 0.78** | 95.44 |
| HGPO 1.5B  | 160 | 91.41 | 92.97 | 91.41 | **91.93 ± 0.90** | 92.77 |
| GiGPO 1.5B | 150 | 86.72 | 89.06 | 83.59 | **86.46 ± 2.74** | 90.16 |

Per-task means (same evaluations):

| Run | Pick | Look | Clean | Heat | Cool | Pick2 |
|---|---|---|---|---|---|---|
| HGPO 7B    | 96.39 | 98.33 | 98.48 | 97.62 | 92.41 | 95.30 |
| HGPO 1.5B  | 91.37 | 87.06 | 100.00 | 100.00 | 87.57 | 83.20 |
| GiGPO 1.5B | 87.89 | 82.90 | 96.24 | 92.50 | 83.04 | 71.46 |

Per-task columns cover only a handful of the 128 games each, so they swing by ±10 or more
between seeds while `All` varies by ±1–3. Treat them as indicative only.

## Training-time endpoints (single evaluation — noisy)

| Run | Budget | Final | Best | Window (last 4) | Step 100 |
|---|---|---|---|---|---|
| HGPO 7B    | 160 | 92.2 | 96.9 @135 | 95.1 | 91.4 |
| HGPO 1.5B  | 160 | 93.0 | 93.8 @155 | 91.0 | 79.7 |
| GiGPO 1.5B | 150 | 86.7 | 88.3 @130 | 85.3 | 76.6 |
| G2PO 1.5B  | 100 | 91.4 | 95.3 @95  | 90.8 | 91.4 |

**HGPO 7B's endpoint is the cautionary case**: its single step-160 evaluation read 92.2,
but the 3-seed mean of that same checkpoint is 96.09 — the endpoint was a low draw, and
quoting it would have understated the run by 3.9 points and missed that it beats the
published figure.

## Budget matters

G2PO was only ever run to 100 steps, GiGPO to 150, HGPO to 160, so cross-method
comparisons at their endpoints are not budget-matched. At the common 100-step budget:

| Method (1.5B) | All @100 |
|---|---|
| G2PO  | 91.4 |
| HGPO  | 79.7 |
| GiGPO | 76.6 |

HGPO's 11.7-point deficit to G2PO at step 100 closes entirely by step 160. Any table
mixing endpoints must state each method's budget.

## Published targets (HGPO paper, Table 1, K=2, in-distribution)

| Method | 1.5B | 7B |
|---|---|---|
| HGPO  | 92.77 | 95.44 |
| GiGPO | 90.16 | 93.29 |
| GRPO  | 72.8  | 78.64 |
| G2PO  | 95.0  | — |

The paper averages **three random seeds**; single-seed runs here should not be compared to
those numbers without error bars. Note also that the paper's Table 7 reports the same
methods on an older verl-agent version with markedly lower numbers (HGPO 7B K=2: 91.15),
so framework version moves these results by several points.

## Not yet measured

- Every WebShop number — the environment is not installed (see `env/setup_webshop.sh`).
- GRPO at any size on either benchmark — no GRPO arm has been run on this host.
- GiGPO 7B, G2PO 7B — only 1.5B versions exist so far.
