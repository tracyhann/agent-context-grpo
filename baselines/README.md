# Baselines

Reference checkouts of the four ancestors CCPO is positioned against, plus the
GRPO implementation actually in use. Shallow clones (`--depth 1`), pulled for
reading — not wired into anything, and **gitignored**: see the root `.gitignore`.

Papers for all four are in [`../related-works/`](../related-works/).

## Where each estimator actually lives

| method | file | entry point |
|---|---|---|
| **GRPO** | `verl/verl/trainer/ppo/core_algos.py` | `compute_grpo_outcome_advantage` (L268) |
| **GiGPO** | `verl-agent/gigpo/core_gigpo.py` | `compute_gigpo_outcome_advantage` (L138); grouping in `build_step_group` (L243) |
| **HGPO** | `verl-agent/recipe/hgpo/core_hgpo.py` | `hgpo_advantage_estimate` (L146), `compute_hgpo_outcome_advantage` (L66) |
| **G²PO** | `G2PO/g2po/core_g2po.py` | `compute_g2po_outcome_advantage` (L216); node values in `compute_group_aggregation_values` (L65) |

## Repos

| dir | source | HEAD | note |
|---|---|---|---|
| `verl-agent/` | [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent) | `20bd331` 2026-06-09 | GiGPO's official repo **and** HGPO's (`recipe/hgpo`). This is the codebase `../patches/` mirrors. |
| `G2PO/` | [Nala-YN/G2PO](https://github.com/Nala-YN/G2PO) | `b8ffaf5` 2026-06-29 | Standalone verl fork; vendors its own `verl/` and `agent_system/`. |
| `verl/` | [verl-project/verl](https://github.com/verl-project/verl) | `23af6a7` 2026-09-04 | Upstream trainer. `volcengine/verl` now redirects here. |
| `DeepSeek-Math/` | [deepseek-ai/DeepSeek-Math](https://github.com/deepseek-ai/DeepSeek-Math) | `b8b0f8c` 2024-04-15 | GRPO's origin paper repo — **contains no GRPO training code**, only a 98 MB math eval harness. Kept for provenance; read `verl` for the actual algorithm. |

## Two things worth knowing

**HGPO ships inside verl-agent, not a separate repo** — `recipe/hgpo`, with its own
ray trainer, env manager, config and the ALFWorld/WebShop run scripts at
Qwen2.5-1.5B and 7B. Since `../patches/` already mirrors this same repo, HGPO is
the ancestor whose code is *closest* to being drop-in comparable here.

**`verl-agent/recipe/GraphGPO/` is a fifth relevant method**, not one of the four
above: *Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for
Agentic RL* ([2605.26684](https://arxiv.org/abs/2605.26684), Cheng, He, Feng, …,
An — same group as GiGPO/HGPO). It extends GiGPO with a state-transition graph
and shortest-path-to-goal step returns. Distinct from G²PO (PKU/Microsoft),
which is also graph-based. It ships a `compare_advantages.py` that diffs
estimators on the same rollouts — directly reusable for the kind of head-to-head
this project needs.
