"""A CPU-only example with synthetic rollouts; no model or benchmark downloads."""
import numpy as np

from ccpo import CCPOConfig, RolloutBatch, compute_advantages

lengths = np.repeat([4, 3, 4], [4, 3, 4])
turns = np.concatenate([np.arange(4), np.arange(3), np.arange(4)])
outcomes = np.repeat([10., 0., 10.], [4, 3, 4])
invalid = np.zeros(len(turns), dtype=bool)
invalid[1] = True
penalty = 0.1 * invalid

batch = RolloutBatch(
    task_ids=["task"] * len(turns),
    trajectory_ids=np.repeat(["a", "b", "c"], [4, 3, 4]),
    observations=np.array(["room", "object", "room", "goal", "room", "object", "room",
                           "room", "object", "room", "goal"]),
    turns=turns,
    lengths=lengths,
    # In training, these are the frozen reference's last-prompt-token states.
    hidden=np.random.default_rng(0).normal(size=(len(turns), 16)),
    returns=0.95 ** (lengths - turns - 1) * outcomes - penalty,
    episode_rewards=outcomes,
    episode_scores=outcomes - penalty,
)

for benchmark in ("alfworld", "webshop"):
    for episode_weight in (0, 1):
        config = CCPOConfig(benchmark=benchmark, episode_weight=episode_weight)
        result = compute_advantages(batch, config)
        print(f"{benchmark}, episode_weight={episode_weight}: "
              f"{np.round(result.advantages, 3)}")
