"""Contextual Credit Policy Optimization. Calculations use CPU NumPy."""
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Literal

import numpy as np

from .features import build_features, context_statistics

EPS = 1e-6


@dataclass(frozen=True)
class CCPOConfig:
    benchmark: Literal["alfworld", "webshop"] = "alfworld"
    episode_weight: Literal[0, 1] = 0
    horizon: int = 2
    gamma: float = 0.95
    kernel_scale: float = 0.15
    remove_top_pcs: int = 3

    def __post_init__(self):
        if self.benchmark not in ("alfworld", "webshop"):
            raise ValueError("benchmark must be 'alfworld' or 'webshop'")
        if self.episode_weight not in (0, 1):
            raise ValueError("episode_weight must be 0 or 1")
        for name, minimum in (("horizon", 1), ("remove_top_pcs", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must lie in [0,1]")
        if not np.isfinite(self.kernel_scale) or self.kernel_scale <= 0:
            raise ValueError("kernel_scale must be positive and finite")


@dataclass(frozen=True)
class RolloutBatch:
    """One row per observed action turn; complete trajectories in any row order.

    IDs and observations are strings. hidden is [N,D]; other fields are [N].
    returns: discounted environment return minus the CURRENT invalid penalty.
    episode_rewards: unpenalized binary episode outcome (10 or 0), repeated.
    episode_scores: episode outcome minus the CURRENT invalid penalty per turn.
    Identical transport copies are allowed; they never become extra peers.
    """
    task_ids: Sequence[str]
    trajectory_ids: Sequence[str]
    observations: Sequence[str]
    turns: np.ndarray
    lengths: np.ndarray
    hidden: np.ndarray
    returns: np.ndarray
    episode_rewards: np.ndarray
    episode_scores: np.ndarray


@dataclass(frozen=True)
class CCPOResult:
    """Row-aligned scalars; unsupported baselines/potentials are NaN.

    history and future are before the optional combined standardization;
    contextual is the combined channel actually used by the actor.
    episode is diagnostic even when its configured actor weight is zero.
    """
    advantages: np.ndarray
    contextual: np.ndarray
    history: np.ndarray
    future: np.ndarray
    episode: np.ndarray
    history_baseline: np.ndarray
    current_value: np.ndarray
    future_value: np.ndarray
    future_delta: np.ndarray
    supported: np.ndarray
    peer_trajectories: np.ndarray
    effective_peers: np.ndarray
    used_task_fallback: np.ndarray


def _validate(batch: RolloutBatch) -> RolloutBatch:
    n = len(batch.task_ids)
    if n == 0:
        raise ValueError("Rollout batch must not be empty")
    arrays = {}
    for name in ("task_ids", "trajectory_ids", "observations"):
        value = np.asarray(getattr(batch, name))
        if value.shape != (n,) or not all(isinstance(item, (str, bytes)) for item in value):
            raise ValueError(f"{name} must contain N strings")
        arrays[name] = value.astype(str)
    for name in ("turns", "lengths"):
        value = np.asarray(getattr(batch, name))
        if value.shape != (n,) or value.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain N integers")
        arrays[name] = value.astype(np.int64)
    if np.any(arrays["turns"] < 0) or np.any(arrays["turns"] >= arrays["lengths"]):
        raise ValueError("Each turn must lie in [0, episode length)")
    for name in ("returns", "episode_rewards", "episode_scores", "hidden"):
        value = np.asarray(getattr(batch, name), dtype=np.float64)
        if name == "hidden":
            valid_shape = value.ndim == 2 and value.shape[0] == n and value.shape[1] > 0
        else:
            valid_shape = value.shape == (n,)
        if not valid_shape or not np.isfinite(value).all():
            raise ValueError(f"{name} has an invalid shape or non-finite values")
        arrays[name] = value
    if not np.isin(arrays["episode_rewards"], [0., 10.]).all():
        raise ValueError("CCPO benchmark episode rewards must be binary 0/10")
    return RolloutBatch(**arrays)


def _canonicalize(batch: RolloutBatch):
    """Sort complete trajectories and remove exact transport copies."""
    first, owners, lengths = {}, {}, {}
    keys = []
    for row, (task, trajectory, turn, length) in enumerate(zip(
        batch.task_ids, batch.trajectory_ids, batch.turns, batch.lengths,
    )):
        if trajectory in owners and owners[trajectory] != task:
            raise ValueError("Trajectory IDs must be globally unique across tasks")
        if trajectory in lengths and lengths[trajectory] != length:
            raise ValueError("Inconsistent lengths within a trajectory")
        owners[trajectory], lengths[trajectory] = task, length
        key = (task, trajectory, turn)
        first.setdefault(key, row)
        keys.append(key)
    ordered = sorted(first)
    position = {key: row for row, key in enumerate(ordered)}
    take = np.array([first[key] for key in ordered])
    restore = np.array([position[key] for key in keys])
    canonical = RolloutBatch(**{f.name: getattr(batch, f.name)[take] for f in fields(batch)})
    for field in fields(batch):
        if not np.array_equal(getattr(batch, field.name), getattr(canonical, field.name)[restore]):
            raise ValueError(f"Inconsistent duplicate rows: {field.name}; capture each unique turn once")
    groups = defaultdict(list)
    for row, trajectory in enumerate(canonical.trajectory_ids):
        groups[trajectory].append(row)
    for trajectory, rows in groups.items():
        if canonical.turns[rows].tolist() != list(range(lengths[trajectory])):
            raise ValueError(f"Incomplete trajectory: {trajectory}")
        if not np.all(canonical.episode_rewards[rows] == canonical.episode_rewards[rows[0]]):
            raise ValueError("Episode reward must be constant within a trajectory")
    return canonical, restore, [np.array(rows) for rows in groups.values()]


def _readouts(batch: RolloutBatch, features: np.ndarray, targets: np.ndarray, scale: float):
    """Estimate history and potential together, with exact groups then task backoff."""
    n = len(features)
    baseline = np.full((n, 2), np.nan)
    support, effective = np.zeros(n, dtype=int), np.zeros(n)
    level = np.full(n, -1, dtype=int)
    for depth, keys in enumerate((zip(batch.task_ids, batch.observations), batch.task_ids)):
        buckets = defaultdict(list)
        for row, key in enumerate(keys):
            buckets[key].append(row)
        for rows in buckets.values():
            ids = np.array(rows)
            trajectories = batch.trajectory_ids[ids]
            if len(set(trajectories)) < 2 or np.all(level[ids] >= 0):
                continue
            z = features[ids]
            squared = (z * z).sum(axis=1)
            distances = np.sqrt(np.maximum(squared[:, None] + squared[None, :] - 2 * (z @ z.T), 0))
            median = np.median(distances[np.triu_indices(len(ids), 1)])
            bandwidth = scale * (median if median > 1e-9 else 1.)
            for local, row in enumerate(ids):
                if level[row] >= 0:
                    continue
                peers = np.flatnonzero(trajectories != trajectories[local])
                weights = np.exp(-distances[local, peers] / bandwidth)
                # Retain occurrence weights, while counting support by trajectory.
                sums, masses = {}, {}
                for weight, peer in zip(weights, peers):
                    trajectory = trajectories[peer]
                    masses[trajectory] = masses.get(trajectory, 0.) + weight
                    sums[trajectory] = sums.get(trajectory, np.zeros(2)) + weight * targets[ids[peer]]
                mass = np.array(list(masses.values()))
                means = np.stack([sums[t] / max(masses[t], 1e-12) for t in masses])
                baseline[row] = (mass[:, None] * means).sum(axis=0) / max(mass.sum(), 1e-12)
                support[row] = len(mass)
                effective[row] = mass.sum() ** 2 / max(float(mass @ mass), 1e-12)
                level[row] = depth
    return baseline, support, effective, level


def _normalize(values, task_ids, active, *, divide_std, keep_singleton):
    result = values.copy() if keep_singleton else np.zeros_like(values)
    for task in np.unique(task_ids):
        rows = np.flatnonzero((task_ids == task) & active)
        if len(rows) < 2:
            continue
        centered = values[rows] - values[rows].mean()
        result[rows] = centered / (values[rows].std(ddof=1) + EPS) if divide_std else centered
    return result


def compute_advantages(batch: RolloutBatch, config: CCPOConfig = CCPOConfig()) -> CCPOResult:
    """Compute CCPO credit, optionally adding episode advantage.

    Contextual baselines are applied directly. Episode credit is added after
    benchmark-specific contextual normalization. Returned arrays follow the
    original input row order.
    """
    original = _validate(batch)
    batch, restore, trajectories = _canonicalize(original)
    context = context_statistics(batch.observations, batch.trajectory_ids, batch.turns)
    features = build_features(batch.hidden, context, config.remove_top_pcs)
    labels = config.gamma ** (batch.lengths - batch.turns) * batch.episode_rewards
    baselines, support, effective, level = _readouts(
        batch, features, np.column_stack([batch.returns, labels]), config.kernel_scale,
    )
    live = level >= 0
    history = np.where(live, batch.returns - baselines[:, 0], 0.)
    current = baselines[:, 1]
    future_value = batch.episode_rewards.copy()
    for rows in trajectories:
        for turn, row in enumerate(rows):
            end = turn + config.horizon
            if end < len(rows):
                future_value[row] = current[rows[end]]
    eligible = np.isfinite(current) & np.isfinite(future_value)
    delta = np.where(eligible, future_value - current, 0.)
    future = _normalize(delta, batch.task_ids, eligible, divide_std=True, keep_singleton=False)
    # Future moments use unique experience. Final contextual/episode moments use
    # trainer rows, matching the existing turn-weighted benchmark conventions.
    live = (live | eligible)[restore]
    history, future = history[restore], future[restore]
    contextual = history + future
    if config.benchmark == "alfworld":
        contextual = _normalize(contextual, original.task_ids, live, divide_std=True, keep_singleton=True)
    episode = _normalize(original.episode_scores, original.task_ids,
                         np.ones(len(restore), dtype=bool),
                         divide_std=config.benchmark == "alfworld", keep_singleton=True)
    if config.benchmark == "alfworld":
        # The inherited episode helper uses mean=0, std=1 for singleton tasks.
        for task in np.unique(original.task_ids):
            rows = np.flatnonzero(original.task_ids == task)
            if len(rows) == 1:
                episode[rows] /= 1 + EPS
    advantages = contextual + config.episode_weight * episode
    if not np.isfinite(advantages).all():
        raise ValueError("Non-finite CCPO advantages")
    return CCPOResult(
        advantages=advantages, contextual=contextual, history=history, future=future,
        episode=episode, history_baseline=baselines[restore, 0],
        current_value=current[restore], future_value=future_value[restore],
        future_delta=delta[restore], supported=live, peer_trajectories=support[restore],
        effective_peers=effective[restore], used_task_fallback=(level[restore] == 1),
    )
