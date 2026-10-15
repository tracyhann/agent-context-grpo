"""CCPO advantages for three synthetic ALFWorld-style trajectories of one task."""
import numpy as np

from ccpo import CCPOConfig, RolloutBatch, compute_ccpo_advantage

lengths = np.repeat([4, 3, 4], [4, 3, 4])
steps = np.concatenate([np.arange(4), np.arange(3), np.arange(4)])
outcomes = np.repeat([10., 0., 10.], [4, 3, 4])
invalid = np.zeros(len(steps), dtype=bool)
invalid[1] = True
penalty = 0.1 * invalid
gamma = 0.95

batch = RolloutBatch(
    task_ids=["task"] * len(steps),
    trajectory_ids=np.repeat(["a", "b", "c"], [4, 3, 4]),
    observations=["room", "object", "room", "goal", "room", "object", "room",
                  "room", "object", "room", "goal"],
    steps=steps,
    lengths=lengths,
    # Stand-in for the frozen reference model's last-prompt-token hidden states.
    hidden=np.sin(np.arange(len(steps) * 16).reshape(len(steps), 16) ** 2 * 0.37 + 0.5),
    returns=gamma ** (lengths - steps - 1) * outcomes - penalty,
    outcomes=outcomes,
    episode_scores=outcomes - penalty,
)

np.set_printoptions(precision=3, suppress=True, linewidth=120)
ccpo = compute_ccpo_advantage(batch, CCPOConfig(benchmark="alfworld"))
with_episode = compute_ccpo_advantage(batch, CCPOConfig(benchmark="alfworld", episode_weight=1))
print("C^H         ", ccpo.historical)
print("C^F         ", ccpo.future)
print("A_CC        ", ccpo.advantages)
print("A_CC + A_EP ", with_episode.advantages)
