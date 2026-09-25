"""Frozen prompt representations and accumulated observation statistics."""
from collections import defaultdict
from collections.abc import Sequence

import numpy as np


def context_statistics(
    observations: Sequence[str], trajectory_ids: Sequence[str], turns: np.ndarray,
) -> np.ndarray:
    """Return [turn, unique observations, revisit indicator, unique/(turn+1)].

    Inputs contain one row per turn, with globally unique trajectory IDs.
    Statistics use each trajectory's prefix and exact observation strings.
    """
    if len(observations) != len(trajectory_ids) or len(turns) != len(observations):
        raise ValueError("Context fields must have the same number of rows")
    groups = defaultdict(list)
    for row, trajectory in enumerate(trajectory_ids):
        groups[trajectory].append(row)
    result = np.zeros((len(turns), 4), dtype=np.float64)
    for rows in groups.values():
        seen = set()
        for row in sorted(rows, key=lambda i: turns[i]):
            observation = observations[row]
            revisit = observation in seen
            seen.add(observation)
            result[row] = (turns[row], len(seen), revisit, len(seen) / (turns[row] + 1))
    return result


def _unit_rows(values: np.ndarray) -> np.ndarray:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-9)


def _thermometer(values: np.ndarray, upper: float) -> np.ndarray:
    count = (12 * np.clip(values / upper, 0, 1)).astype(int)
    return (np.arange(12)[None, :] < count[:, None]).astype(np.float64)


def build_features(hidden: np.ndarray, context: np.ndarray, remove_top_pcs: int = 3) -> np.ndarray:
    """Build unit vectors [processed frozen hidden; 37 context coordinates].

    Center hidden states batch-wide, remove leading principal directions, and
    normalize each block before concatenation. This is not learned whitening.
    Both blocks have coefficient 1. No reward labels enter this construction.
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    context = np.asarray(context, dtype=np.float64)
    if hidden.ndim != 2 or not all(hidden.shape) or context.shape != (len(hidden), 4):
        raise ValueError("Expected hidden [N,D] and context [N,4], with N,D > 0")
    if not np.isfinite(hidden).all() or not np.isfinite(context).all():
        raise ValueError("Features must be finite")
    if isinstance(remove_top_pcs, bool) or not isinstance(remove_top_pcs, (int, np.integer)) or remove_top_pcs < 0:
        raise ValueError("remove_top_pcs must be a nonnegative integer")
    encoded = hidden.copy()
    if len(encoded) >= 2:
        encoded -= encoded.mean(axis=0, keepdims=True)
        if remove_top_pcs and len(encoded) > remove_top_pcs:
            _, _, directions = np.linalg.svd(encoded, full_matrices=False)
            principal = directions[:remove_top_pcs]
            encoded -= encoded @ principal.T @ principal
        encoded = _unit_rows(encoded)
    summary = np.concatenate([
        _thermometer(context[:, 0], 30),
        _thermometer(context[:, 1], 25),
        _thermometer(context[:, 3], 1),
        (context[:, 2:3] > 0).astype(np.float64),
    ], axis=1)
    return _unit_rows(np.concatenate([encoded, _unit_rows(summary)], axis=1))
