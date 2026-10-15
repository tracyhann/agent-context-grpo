# Contextual Credit Policy Optimization

[Project page](https://agent-ccpo.github.io) | [Paper](https://agent-ccpo.github.io/static/paper/ccpo.pdf)

Official implementation of **Contextual Credit Policy Optimization (CCPO)**, a critic-free method that assigns step-level credit to LLM agents trained with group-based reinforcement learning on long-horizon tasks.

![Overview of CCPO](assets/teaser.png)

*Overview of CCPO. CCPO combines contextually weighted peer comparisons and short-horizon future progress for critic-free step-level credit assignment.*

## Method

Recent methods derive step-level credit by comparing observation-matched steps across trajectories. However, identical observations can occur in different contexts, and favorable terminal outcomes can hide redundant actions within successful trajectories. CCPO weights observation-matched peers by contextual similarity and pairs historical credit with future credit, which measures the change in contextual outcome value over a short realized continuation. Both credits are computed in hindsight from the rollouts already collected for training.

![CCPO pipeline](assets/pipeline.png)

*Contextual similarity weights observation-matched peers. A history-conditioned return residual and the estimated outcome-value change over a short future window yield step-level credit without an auxiliary value critic.*

For step $t$ of trajectory $i$ on task $x_i$, peers $\mathcal P_{i,t}$ are steps of other trajectories on the same task with the same observation, or all steps of other same-task trajectories when none match. Each step has a context feature $\phi_{i,t}$ that joins the frozen model's hidden state at the last prompt token with trajectory statistics. The contextual readout of a target $r$ is

```math
\mathcal B_{i,t}[r]=\sum_{(j,s)\in\mathcal P_{i,t}}\bar w_{(i,t),(j,s)}\,r_{j,s},
\qquad
\bar w_{(i,t),(j,s)}\propto\exp\bigl(-\lVert\phi_{i,t}-\phi_{j,s}\rVert_2/\tau_B\bigr).
```

Historical credit compares the return $Y_{i,t}$ with its contextual baseline. Future credit standardizes the change in contextual outcome value $V$ over $K_{\mathrm F}$ steps, with $t^+=\min(t+K_{\mathrm F},T_i)$:

```math
C^{\mathrm H}_{i,t}=Y_{i,t}-\mathcal B_{i,t}[Y],
\qquad
C^{\mathrm F}_{i,t}=z_{x_i}\bigl[V_{i,t^+}-V_{i,t}\bigr],
```

where $V_{i,s}=\mathcal B_{i,s}[Z]$ with $Z_{i,s}=\gamma^{T_i-s}R_i$ for $s<T_i$, and $V_{i,T_i}=R_i$ at the end of the trajectory.

The step advantage is

```math
A_{i,t}=\mathcal N_{x_i}\bigl[C^{\mathrm H}+C^{\mathrm F}\bigr]_{i,t}+\alpha_{\mathrm{EP}}A^{\mathrm{EP}}_{i,t},
```

which is applied to every response token of the step in the PPO objective. CCPO uses $\alpha_{\mathrm{EP}}=0$, and the episode-augmented variant adds the GRPO episode advantage with $\alpha_{\mathrm{EP}}=1$.

## Repository structure

| Path | Contents |
|---|---|
| `ccpo/core_ccpo.py` | Peer readouts, historical and future credit, and the CCPO advantage |
| `ccpo/context.py` | Context feature $\phi$ from hidden states and trajectory statistics |
| `ccpo/verl_adapter.py` | Hidden-state extraction and token advantages for [verl-agent](https://github.com/langfengQ/verl-agent) batches |
| `examples/quickstart.py` | CCPO advantages on synthetic trajectories |
| `tests/` | Unit tests |

## Installation

```bash
git clone https://github.com/phai-lab/CCPO.git
cd CCPO
pip install -e ".[torch]"
python examples/quickstart.py
python -m unittest discover -s tests
```

The estimator needs only NumPy (`pip install -e .`). PyTorch is used by the verl-agent adapter.

## Usage

CCPO takes one row per step of complete trajectories, gathered from all workers before PPO mini-batching.

```python
from ccpo import CCPOConfig, RolloutBatch, compute_ccpo_advantage

batch = RolloutBatch(
    task_ids=task_ids,              # x_i
    trajectory_ids=trajectory_ids,  # i, unique across tasks
    observations=observations,      # o_{i,t}, matched by exact string equality
    steps=steps,                    # t
    lengths=lengths,                # T_i
    hidden=hidden,                  # [N, D] frozen hidden states at the last prompt token
    returns=returns,                # Y_{i,t}, including the step's invalid-action penalty
    outcomes=outcomes,              # R_i
    episode_scores=episode_scores,  # R_i minus the step's invalid-action penalty
)
result = compute_ccpo_advantage(batch, CCPOConfig(benchmark="alfworld"))
result.advantages  # A_{i,t} for each row
```

`result` also holds $C^{\mathrm H}$, $C^{\mathrm F}$, $A^{\mathrm{EP}}$, the readouts $\mathcal B[Y]$ and $V$, and whether each step used the task-level fallback.

With verl-agent, take the hidden states from the frozen reference model and compute advantages on the gathered batch:

```python
from ccpo import CCPOConfig
from ccpo.verl_adapter import compute_verl_advantages, last_prompt_hidden

hidden = last_prompt_hidden(ref_hidden, data.batch["attention_mask"], response_length)
data.batch["ccpo_hidden"] = hidden
config = CCPOConfig(benchmark="webshop", episode_weight=1)
advantages, result = compute_verl_advantages(data, config)
data.batch["advantages"] = advantages
data.batch["returns"] = advantages
```

The adapter reads `response_mask`, `token_level_rewards`, `step_rewards` ($Y$) and `ccpo_hidden` from `data.batch`, and `uid`, `traj_uid`, `anchor_obs`, `ccpo_step_index`, `episode_lengths` and `episode_rewards` ($R$) from `data.non_tensor_batch`.

## Hyperparameters

| Parameter | Symbol | Default | Description |
|---|---|---|---|
| `future_horizon` | $K_{\mathrm F}$ | 2 | Steps spanned by future credit |
| `gamma` | $\gamma$ | 0.95 | Discount factor |
| `bandwidth_scale` | $\tau_B/m_B$ | 0.15 | Kernel bandwidth relative to the median pairwise context distance $m_B$ |
| `num_removed_pcs` | | 3 | Leading principal directions removed from hidden states |
| `episode_weight` | $\alpha_{\mathrm{EP}}$ | 0 | Weight of the episode advantage |
| `benchmark` | | `alfworld` | Normalization convention |

| `benchmark` | $\mathcal N_x$ | $A^{\mathrm{EP}}$ |
|---|---|---|
| `alfworld` | Task z-score | Task z-score of episode scores |
| `webshop` | Identity | Task-centered episode scores |

The prompt history $K_{\mathrm H}=2$ and the invalid-action penalty $\eta=0.1$ are set in the environment, since they determine the prompts and the returns passed to CCPO.

## License

This project is released under the [Apache License 2.0](LICENSE).

## Citation

```bibtex
@article{han2026ccpo,
  title  = {Contextual Credit Policy Optimization},
  author = {Han, Huiwen and Zeng, Ziyao and Zhu, Zihao and
            Bei, Sisong and Wang, Hao and Xu, Hongyu and
            Han, Yan and Zhi, Degui and Fan, Zhiwen},
  year   = {2026}
}
```
