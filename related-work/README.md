# Baseline papers

The four methods this repo reproduces, with the paper each one is defined by.

| File | Method | Paper | arXiv |
|---|---|---|---|
| `grpo_2402.03300.pdf`  | **GRPO**  | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | [2402.03300](https://arxiv.org/abs/2402.03300) |
| `gigpo_2505.10978.pdf` | **GiGPO** | Group-in-Group Policy Optimization for LLM Agent Training | [2505.10978](https://arxiv.org/abs/2505.10978) |
| `g2po_2606.22995.pdf`  | **G2PO**  | Group-Graph Policy Optimization for Long-Horizon Agentic Reinforcement Learning | [2606.22995](https://arxiv.org/abs/2606.22995) |
| `hgpo_2602.22817.pdf`  | **HGPO**  | Hierarchy-of-Groups Policy Optimization for Long-Horizon Agentic Tasks (ICLR 2026) | [2602.22817](https://arxiv.org/abs/2602.22817) |

## What each one changes about the advantage

- **GRPO** — trajectory-level only. One advantage per rollout, standardised within a group
  of rollouts sharing a prompt. No step-level term.
- **GiGPO** — episode advantage **+** a step advantage, grouped by the *current observation*.
  Combined with `step_advantage_w` (1.0 in every run here).
- **G2PO** — episode **+** step, but the step grouping key is a graph/anchor over states
  rather than the raw observation.
- **HGPO** — **step-level only**, and the step term is itself a hierarchy: each step joins
  one group per history depth k = 1…K+1 (suffix-matched observation sequences), each group
  standardises the discounted return-to-go, and the per-depth advantages are combined by a
  length weight `w_k ∝ (k+1)^α`. There is no separate episode term unless `base_group=True`
  (off in every published ALFWorld script). The trajectory signal instead reaches each step
  through the γ-discounted return.

## Numbers quoted in `../RESULTS.md`

Published targets come from HGPO's Table 1 (K=2, in-distribution, mean of 3 seeds).
That paper's Table 7 reports the same methods on an **older verl-agent version** with
markedly lower values (HGPO 7B K=2: 91.15 vs 95.44), so always state which table a
published number came from — the framework version moves results by several points.
