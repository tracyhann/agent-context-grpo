"""Optional PyTorch helpers and a small adapter for verl-agent rollout batches."""
from typing import Any

import numpy as np
import torch

from .estimator import CCPOConfig, CCPOResult, RolloutBatch, compute_advantages


def last_prompt_hidden(
    hidden: torch.Tensor, attention_mask: torch.Tensor, response_length: int,
    packed_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Extract detached float32 reference features at the last prompt token.

    Padded input: hidden [N,L,D], mask [N,L], left-padded prompts followed by
    right-padded responses of width response_length. Packed input: hidden
    [1,nnz(mask),D], with packed_indices equal to flattened nonzero mask indices.
    Use the final-layer normalized hidden state of a frozen causal reference.
    """
    if attention_mask.ndim != 2 or hidden.ndim != 3:
        raise ValueError("Expected attention_mask [N,L] and hidden [N,L,D] or [1,P,D]")
    n, width = attention_mask.shape
    if isinstance(response_length, bool) or not isinstance(response_length, int) or not 0 < response_length < width:
        raise ValueError("response_length must leave at least one prompt token")
    column = width - response_length - 1
    if not bool(attention_mask[:, column].bool().all()):
        raise ValueError("Last prompt token is masked")
    if packed_indices is None:
        if hidden.shape[:2] != (n, width):
            raise ValueError("Padded hidden states do not match inputs")
        result = hidden[:, column]
    else:
        expected = attention_mask.reshape(-1).nonzero().flatten()
        if not torch.equal(packed_indices, expected) or hidden.shape[:2] != (1, expected.numel()):
            raise ValueError("Packed hidden states/indices do not match inputs")
        positions = torch.searchsorted(expected, torch.arange(n, device=expected.device) * width + column)
        result = hidden[0, positions]
    if result.shape[-1] == 0 or not bool(torch.isfinite(result).all()):
        raise ValueError("Frozen prompt features are empty or non-finite")
    return result.detach().float().cpu()


def token_advantages(result: CCPOResult, response_mask: torch.Tensor) -> torch.Tensor:
    """Broadcast one scalar to every generated response token, including thinking.

    Prompt tokens are outside this mask. Padding tokens receive zero advantage.
    Returns detached float32 on response_mask.device.
    """
    if response_mask.ndim != 2 or response_mask.shape[0] != len(result.advantages):
        raise ValueError("response_mask must have shape [number of turns, response width]")
    if not bool(((response_mask == 0) | (response_mask == 1)).all()) or not bool((response_mask.sum(-1) > 0).all()):
        raise ValueError("Each response must have a nonempty binary token mask")
    values = torch.as_tensor(result.advantages, device=response_mask.device, dtype=torch.float32)
    return (values[:, None] * response_mask.detach()).detach()


def compute_verl_advantages(
    data: Any, config: CCPOConfig = CCPOConfig(),
) -> tuple[torch.Tensor, CCPOResult]:
    """Return (token advantages, components) from a verl-agent DataProto-like batch.

    Caller supplies globally gathered, complete trajectories and frozen features
    aligned with source rows. This adapter neither launches Ray nor patches verl.
    step_rewards already include the current invalid-action penalty; episode
    rewards remain unpenalized. token_level_rewards supply episode-channel scores.
    """
    tensors, metadata = data.batch, data.non_tensor_batch
    mask = tensors["response_mask"]
    rewards = tensors["token_level_rewards"]
    if mask.ndim != 2 or rewards.shape != mask.shape:
        raise ValueError("token_level_rewards must match response_mask [N,L]")
    if bool((rewards[mask == 0] != 0).any()):
        raise ValueError("Masked response positions must not contain rewards")
    returns = tensors["step_rewards"]
    if returns.ndim == 2:
        if returns.shape != mask.shape:
            raise ValueError("Token-shaped step_rewards must match response_mask")
        returns = (returns * mask).sum(-1)

    def numpy(value):
        return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)

    def integers(name):
        # verl stores episode lengths as integral floats inside object arrays.
        raw = np.asarray(metadata[name])
        values = np.asarray(raw, dtype=np.float64)
        if (raw.ndim != 1 or any(isinstance(x, (bool, np.bool_)) for x in raw)
                or not np.isfinite(values).all() or np.any(values != np.floor(values))
                or np.any(values < 0) or np.any(values >= 2**63)):
            raise ValueError(f"{name} must contain nonnegative integral metadata")
        return values.astype(np.int64)

    batch = RolloutBatch(
        task_ids=metadata["uid"], trajectory_ids=metadata["traj_uid"],
        observations=metadata["anchor_obs"], turns=integers("ccpo_turn_index"),
        lengths=integers("episode_lengths"), hidden=numpy(tensors["ccpo_phi_feats"]),
        returns=numpy(returns), episode_rewards=numpy(metadata["episode_rewards"]),
        episode_scores=numpy(rewards.sum(-1)),
    )
    result = compute_advantages(batch, config)
    return token_advantages(result, mask), result
