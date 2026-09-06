# Related works

The four ancestors CCPO is positioned against in [`docs/method.html`](../docs/method.html)
(§ "Against the GRPO family" and the grouping-key ladder). arXiv IDs are the ones
cited inline in that page; PDFs pulled from `arxiv.org/pdf/<id>`.

| file | method | title | first author | arXiv | v | date |
|---|---|---|---|---|---|---|
| [`grpo_2402.03300.pdf`](grpo_2402.03300.pdf) | **GRPO** | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | Zhihong Shao | [2402.03300](https://arxiv.org/abs/2402.03300) | v3 | 2024-02-05 |
| [`gigpo_2505.10978.pdf`](gigpo_2505.10978.pdf) | **GiGPO** | Group-in-Group Policy Optimization for LLM Agent Training | Lang Feng | [2505.10978](https://arxiv.org/abs/2505.10978) | v3 | 2025-05-16 |
| [`hgpo_2602.22817.pdf`](hgpo_2602.22817.pdf) | **HGPO** | Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks | Shuo He | [2602.22817](https://arxiv.org/abs/2602.22817) | v1 | 2026-02-26 |
| [`g2po_2606.22995.pdf`](g2po_2606.22995.pdf) | **G²PO** | Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning | Yunan Wang | [2606.22995](https://arxiv.org/abs/2606.22995) | v1 | 2026-06-22 |

## Why each one is here

**GRPO** is the base the whole family sits on and the arm CCPO is meant to be
compared against head-to-head: group-relative advantage over a task's siblings,
no step baseline at all. `A_EP` in `ccpo/core_ccpo.py` is unchanged from it.

**GiGPO** introduces the anchor-state step group — bucket a rollout group's steps
by exact-match current state, normalise inside each bucket. CCPO keeps that gate
and replaces the hard cluster with a weighted leave-one-out baseline.

**HGPO** groups by exact k-step history instead, nesting buckets by depth. It is
the contrast case for why CCPO conditions on a continuous φ: exact-suffix support
decays fast in k.

**G²PO** is the direct comparison target throughout the repo — its node key is the
same raw observation text, and its node value includes the trajectory being
scored (CCPO's does not). `r_vs_g2po` in the estimator's diagnostics, and the
0.956 correlation reported in the top-level README, are measured against its step
credit. `scripts/run_verl_ccpo.sh` mirrors its ALFWorld reference config.

## Note

`docs/method.html` cites published G²PO/GiGPO numbers as *regime context*, not as
a claimed parity comparison — see the "full comparison" table's own caveat. The
only internally exact comparison in this repo is CCPO vs GRPO under an identical
config, and that baseline does not exist yet (see the top-level README).
