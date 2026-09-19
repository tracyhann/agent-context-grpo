"""One retained reference readout per source row, with verified input identity.

Training padding is NOT additional experience. Collapse it before reference
dispatch, retain one feature/log-probability pair, and restore by source ID.
The few copies required by DP dispatch are checked and measured separately.
"""
import hashlib
import json

import numpy as np
import torch


SOURCE_ID = 'ccpo_source_row'
SOURCE_COUNT = 'ccpo_source_count'
VERIFIED = 'ccpo_verified_phi'
PACKET = 'ccpo_phi_packet'
HASH_BYTES = 32
INPUT_KEYS = ('input_ids', 'attention_mask', 'position_ids', 'responses', SOURCE_ID)


def mark_source_rows(data):
    """Assign BEFORE adjust_batch; tensor IDs follow padding and balancing."""
    if len(data) == 0 or SOURCE_ID in data.batch:
        raise ValueError('Expected a nonempty, unmarked rollout batch')
    data.batch[SOURCE_ID] = torch.arange(len(data), device=data.batch['input_ids'].device)
    data.meta_info[SOURCE_COUNT] = len(data)


def input_fingerprints(batch):
    """SHA-256 of each actual micro-batch row, including pooling geometry/ID.

    Run independently on the driver and at hidden-state capture. Never echo a
    precomputed driver hash through the worker: that would miss input mixups.
    """
    arrays, headers = [], []
    for key in INPUT_KEYS:
        value = batch[key].detach().cpu().contiguous().numpy()
        if value.dtype.kind not in 'biu':
            raise ValueError('Expected integer reference input: ' + key)
        arrays.append(value)
        headers.append(json.dumps([key, value.dtype.str, value.shape[1:]]).encode())
    n = len(arrays[0])
    if any(len(value) != n for value in arrays):
        raise ValueError('Reference input row counts differ')
    result = np.empty((n, HASH_BYTES), dtype=np.uint8)
    for row in range(n):
        digest = hashlib.sha256()
        for header, value in zip(headers, arrays):
            digest.update(header)
            digest.update(value[row].tobytes())
        result[row] = np.frombuffer(digest.digest(), dtype=np.uint8)
    return torch.from_numpy(result)


def pool_last_prompt(hidden, batch, indices=None):
    """Select the final prompt token, checking the padded/packed row mapping."""
    n, seqlen = batch['input_ids'].shape
    col = seqlen - batch['responses'].shape[-1] - 1
    if hidden is None or hidden.ndim != 3 or not 0 <= col < seqlen:
        raise ValueError('Missing or malformed frozen hidden states / prompt boundary')
    mask = batch['attention_mask']
    if mask.shape != (n, seqlen) or not bool(mask[:, col].bool().all()):
        raise ValueError('Last prompt token is masked or input geometry is inconsistent')
    if indices is None:
        if tuple(hidden.shape[:2]) != (n, seqlen):
            raise ValueError('Padded frozen hidden states do not match model inputs')
        features = hidden[:, col, :]
    else:
        expected = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]
        if not torch.equal(indices, expected):
            raise ValueError('Packed token indices do not match the attention mask')
        if tuple(hidden.shape[:2]) != (1, indices.numel()):
            raise ValueError('Packed frozen hidden states do not match model inputs')
        flat = torch.arange(n, device=indices.device) * seqlen + col
        positions = torch.searchsorted(indices, flat)
        if bool((positions >= len(indices)).any()) or not torch.equal(indices[positions], flat):
            raise ValueError('Packed last-prompt positions are missing')
        features = hidden[0, positions, :]
    return features.detach().float().cpu()


def pack_features(features, batch):
    """Bind fingerprints and features BEFORE any micro-batch inverse permutation.

    Fingerprint bytes are exactly representable in float32. A SINGLE tensor is
    reordered and gathered so identity cannot take a different permutation.
    """
    if features.ndim != 2 or len(features) != len(batch['input_ids']) or features.shape[1] == 0:
        raise ValueError('Frozen feature coverage does not match the micro-batch')
    features = features.detach().float().cpu()
    if not bool(torch.isfinite(features).all()):
        rows = (~torch.isfinite(features).all(dim=1)).nonzero().flatten().tolist()
        raise ValueError(f'Non-finite frozen features at capture; micro-batch rows {rows[:5]}')
    return torch.cat((features, input_fingerprints(batch).float()), dim=1)


def compute_verified_ref(data, worker_group):
    """Reference forward for future-progress arms, without training-pad rereads.

    No extra model pass. Ref log-probs and phi share the same canonical forward.
    Unsupported capture, nonfinite values, and identity mismatches fail before
    advantage computation / optimizer update. Finite transport-copy variation
    is recorded, never averaged into a source row's retained representation.
    """
    from verl import DataProto
    from verl.protocol import pad_dataproto_to_divisor

    if 'multi_modal_inputs' in data.non_tensor_batch:
        raise ValueError('Verified future-progress capture currently requires text-only inputs')
    if SOURCE_ID not in data.batch:
        raise ValueError('Missing source IDs: mark rows before training padding')
    source = data.batch[SOURCE_ID].detach().cpu().numpy()
    if source.shape != (len(data),) or source.dtype.kind not in 'iu' or len(source) == 0:
        raise ValueError('Malformed source row IDs')
    unique, first, inverse = np.unique(source, return_index=True, return_inverse=True)
    if data.meta_info.get(SOURCE_COUNT) != len(unique) or not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError('Lost or invalid source row IDs during padding/balancing')
    # Preserve the balanced batch's first-occurrence order, rather than sorting
    # back into trajectory order and undoing the driver's length balancing.
    order = np.argsort(first)
    take = first[order]
    restore = np.argsort(order)[inverse]
    expected = input_fingerprints(data.batch)
    if not torch.equal(expected, expected[take][restore]):
        rows = (expected != expected[take][restore]).any(dim=1).nonzero().flatten().tolist()
        raise ValueError(f'Training padding changed reference inputs; rows {rows[:5]}')

    canonical = data.select_idxs(take)
    # Pass only model inputs/IDs. Avoid old log-probs, rewards and trajectory
    # metadata, which are unrelated to the reference forward.
    canonical = canonical.select(batch_keys=list(INPUT_KEYS), non_tensor_batch_keys=[])
    canonical.meta_info = dict(data.meta_info, **{VERIFIED: True})
    request, padding = pad_dataproto_to_divisor(canonical, int(worker_group.world_size))
    request_ids = request.batch[SOURCE_ID].detach().cpu().clone()
    request_hash = input_fingerprints(request.batch)
    output = worker_group.compute_ref_log_prob(request)
    if len(output) != len(request) or PACKET not in output.batch:
        raise ValueError('Reference worker returned missing/incomplete verified feature packets')
    packet = output.batch[PACKET].detach().cpu()
    if packet.ndim != 2 or packet.shape[1] <= HASH_BYTES or packet.dtype != torch.float32:
        raise ValueError('Malformed verified feature packet (requires float32)')
    if not bool(torch.isfinite(packet).all()):
        raise ValueError('Non-finite frozen features in reference output, including transport padding')
    actual_hash = packet[:, -HASH_BYTES:]
    bad = (actual_hash != request_hash.float()).any(dim=1)
    if bool(bad.any()):
        rows = bad.nonzero().flatten()[:5].tolist()
        raise ValueError(f'Reference feature/input identity mismatch; rows {rows}, '
                         f'source IDs {request_ids[rows].tolist()}')
    features = packet[:, :-HASH_BYTES]
    log_probs = output.batch['ref_log_prob'].detach().cpu()
    if log_probs.shape != request.batch['responses'].shape or not bool(torch.isfinite(log_probs).all()):
        raise ValueError('Non-finite or malformed reference log probabilities')
    n = len(take)
    max_diff = 0.
    differing_rows = 0
    if padding:
        lookup = {int(value): row for row, value in enumerate(request_ids[:n])}
        originals = torch.tensor([lookup[int(value)] for value in request_ids[n:]])
        difference = (features[n:] - features[originals]).abs()
        max_diff = float(difference.max())
        differing_rows = int((difference != 0).any(dim=1).sum())
    result = DataProto.from_dict(tensors={
        'ref_log_prob': log_probs[:n][restore],
        'ccpo_phi_feats': features[:n][restore].clone(),
    })
    metrics = {
        'ccpo/phi_verified_capture': 1.,
        'ccpo/phi_source_rows': float(n),
        'ccpo/phi_training_padding_rows': float(len(data) - n),
        'ccpo/phi_transport_padding_rows': float(padding),
        'ccpo/phi_transport_duplicate_max_diff': max_diff,
        'ccpo/phi_transport_duplicate_differing_rows': float(differing_rows),
        'ccpo/phi_input_identity_mismatches': 0.,
        'ccpo/phi_nonfinite_rows': 0.,
    }
    return result, metrics
