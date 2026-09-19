"""CPU regression for the actual forward, micro-batch sort, DP split/collect.

Uses the live worker methods with a deterministic tiny model; no Ray processes
or GPUs. This verifies row plumbing, not H200/FlashAttention numerical accuracy.
"""
import ast
from contextlib import redirect_stdout
import io
import itertools
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Tuple
import unittest
from unittest.mock import patch

os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'verl-agent'), str(ROOT / 'tests')]
import numpy as np
import torch
from einops import rearrange
from verl import DataProto
from verl.single_controller.base.worker_group import WorkerGroup
from verl.single_controller.base.decorator import dispatch_dp_compute_data_proto, collect_dp_compute_data_proto
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from ccpo.phi_capture import (SOURCE_ID, VERIFIED, PACKET, HASH_BYTES, mark_source_rows,
                              input_fingerprints, pool_last_prompt, compute_verified_ref)


def load_methods(relative, names, namespace):
    path = ROOT / 'verl-agent' / relative
    assert path.read_bytes() == (ROOT / 'patches/verl-agent' / relative).read_bytes()
    methods = [node for node in ast.walk(ast.parse(path.read_text()))
               if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == len(names)
    for method in methods:
        method.decorator_list = []
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def unpad_input(value, mask):
    indices = torch.nonzero(mask.reshape(-1), as_tuple=True)[0]
    return value.flatten(0, 1)[indices], indices


def pad_input(hidden_states, indices, batch, seqlen):
    result = hidden_states.new_zeros((batch * seqlen, hidden_states.shape[-1]))
    result[indices] = hidden_states
    return result.reshape(batch, seqlen, -1)


def logprobs_from_logits(logits, labels, **kwargs):
    return logits.log_softmax(-1).gather(-1, labels[..., None]).squeeze(-1)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Identity()
        self.calls = 0
        self.table = torch.randn(128, 8, generator=torch.Generator().manual_seed(71))

    def forward(self, input_ids, position_ids, **kwargs):
        self.calls += 1
        hidden = self.norm(self.table[input_ids] + position_ids[..., None].float() / 100)
        logits = hidden[..., :1] * torch.linspace(-1, 1, 128)
        return SimpleNamespace(logits=logits)


actor_namespace = dict(torch=torch, Tuple=Tuple, DataProto=DataProto, itertools=itertools,
                       get_reverse_idx=get_reverse_idx, rearrange_micro_batches=rearrange_micro_batches,
                       unpad_input=unpad_input, pad_input=pad_input, rearrange=rearrange,
                       index_first_axis=lambda value, indices: value[indices],
                       logprobs_from_logits=logprobs_from_logits)
Actor = type('Actor', (), load_methods('verl/workers/actor/dp_actor.py',
             ['_forward_micro_batch', '_acg_hidden_hook', '_acg_final_norm', 'compute_log_prob'], actor_namespace))
worker_namespace = dict(torch=torch, DataProto=DataProto, os=os,
                        get_torch_device=lambda: SimpleNamespace(current_device=lambda: 'cpu'))
RefWorker = type('RefWorker', (), load_methods('verl/workers/fsdp_workers.py',
               ['compute_ref_log_prob'], worker_namespace))


def actor(packed):
    result = Actor()
    result.actor_module = TinyModel()
    result.use_remove_padding = packed
    result.use_ulysses_sp = False
    result.ulysses_sequence_parallel_size = 1
    result.use_fused_kernels = False
    result.device_name = 'cpu'
    return result


class Sharding:
    def __enter__(self): return self
    def __exit__(self, *exc): return False
    def preprocess_data(self, data): return data
    def postprocess_data(self, data): return data


class Group(WorkerGroup):
    def __init__(self, size=4, packed=True, dynamic=True, corruption=None):
        self._workers = [None] * size
        self.packed, self.dynamic, self.corruption = packed, dynamic, corruption
        self.request = None

    def compute_ref_log_prob(self, data):
        self.request = data
        args, kwargs = dispatch_dp_compute_data_proto(self, data)
        outputs = []
        for rank, shard in enumerate(args[0]):
            if self.corruption == 'inputs' and rank == 0:
                shard.batch['input_ids'][0, -5] += 1
            worker = RefWorker()
            worker._is_lora, worker._is_ref, worker.world_size = False, True, 1
            worker.ulysses_sharding_manager = Sharding()
            worker.ref_policy = actor(self.packed)
            worker.config = SimpleNamespace(
                ref=SimpleNamespace(log_prob_micro_batch_size_per_gpu=2,
                                    log_prob_max_token_len_per_gpu=16,
                                    log_prob_use_dynamic_bsz=self.dynamic),
                rollout=SimpleNamespace(temperature=1.))
            outputs.append(worker.compute_ref_log_prob(shard))
        result = collect_dp_compute_data_proto(self, outputs)
        packet = result.batch[PACKET]
        if self.corruption == 'swap':
            packet[[0, 1]] = packet[[1, 0]]
        elif self.corruption == 'drop':
            return result[:-1]
        elif self.corruption == 'missing':
            result.batch.pop(PACKET)
        elif self.corruption == 'nan':
            packet[-1, 0] = float('nan')
        elif self.corruption == 'drift':
            ids = data.batch[SOURCE_ID].tolist()
            seen = set()
            for row, source in enumerate(ids):
                if source in seen:
                    packet[row, 0] += 1.75
                seen.add(source)
        return result


def batch(n=29):
    ids = (torch.arange(n) % 100 + 1)[:, None].expand(-1, 12).clone()
    mask = torch.ones_like(ids)
    for i in range(n):
        mask[i, :i % 5] = 0
        mask[i, 12 - i % 3:] = 0
    positions = (mask.cumsum(-1) - 1).clamp(min=0) * mask
    return DataProto.from_dict(tensors=dict(input_ids=ids, attention_mask=mask,
                              position_ids=positions, responses=ids[:, -4:].clone()))


def padded_batch(n=29):
    data = batch(n)
    mark_source_rows(data)
    order = np.random.default_rng(52).permutation(np.r_[np.arange(n), np.arange(n // 2), 0, 0])
    return data.select_idxs(order)


def expected_features(data):
    return TinyModel().table[data.batch['input_ids'][:, -5]] + data.batch['position_ids'][:, -5, None].float() / 100


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, ACG_ADV_ESTIMATOR='ccpo', ACG_CCPO_PHI='hidden+ctx')
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_real_worker_split_forward_sort_collect_and_restore(self):
        for size in [1, 4, 8]:
            for packed in [False, True]:
                for dynamic in [False, True]:
                    with self.subTest(size=size, packed=packed, dynamic=dynamic):
                        data = padded_batch()
                        worker = Group(size, packed, dynamic)
                        output, metrics = compute_verified_ref(data, worker)
                        torch.testing.assert_close(output.batch['ccpo_phi_feats'], expected_features(data), rtol=0, atol=0)
                        self.assertEqual(len(worker.request), 29 + (-29 % size))
                        self.assertEqual(metrics['ccpo/phi_training_padding_rows'], len(data) - 29)
                        self.assertEqual(metrics['ccpo/phi_transport_duplicate_max_diff'], 0.)
                        self.assertFalse(output.batch['ccpo_phi_feats'].requires_grad)
                        raw = TinyModel()(data.batch['input_ids'], data.batch['position_ids']).logits
                        expected_logp = logprobs_from_logits(raw[:, -5:-1], data.batch['responses'])
                        # Packed log probs at masked response positions are not used.
                        mask = data.batch['attention_mask'][:, -4:].bool()
                        torch.testing.assert_close(output.batch['ref_log_prob'][mask], expected_logp[mask])

    def test_transport_drift_logged_but_source_features_are_retained(self):
        data = padded_batch()
        output, metrics = compute_verified_ref(data, Group(8, corruption='drift'))
        torch.testing.assert_close(output.batch['ccpo_phi_feats'], expected_features(data), rtol=0, atol=0)
        self.assertEqual(metrics['ccpo/phi_transport_duplicate_differing_rows'], 3.)
        self.assertAlmostEqual(metrics['ccpo/phi_transport_duplicate_max_diff'], 1.75, places=5)

    def test_wrong_inputs_bad_collection_missing_features_and_nonfinite_fail(self):
        for corrupt, message in [('inputs', 'identity mismatch'), ('swap', 'identity mismatch'),
                                  ('drop', 'incomplete'), ('missing', 'incomplete'), ('nan', 'Non-finite')]:
            with self.subTest(corrupt=corrupt), self.assertRaisesRegex(ValueError, message):
                compute_verified_ref(padded_batch(), Group(8, corruption=corrupt))

    def test_false_padding_and_lost_source_rows_fail_before_forward(self):
        for field in ['input_ids', 'attention_mask', 'position_ids', 'responses']:
            data = padded_batch()
            rows = (data.batch[SOURCE_ID] == 0).nonzero().flatten()
            data.batch[field][rows[-1], 0] += 1
            group = Group()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'padding changed'):
                compute_verified_ref(data, group)
            self.assertIsNone(group.request)
        data = padded_batch()
        data = data.select_idxs((data.batch[SOURCE_ID] != 3).nonzero().flatten())
        with self.assertRaisesRegex(ValueError, 'Lost or invalid'):
            compute_verified_ref(data, Group())

    def test_small_batch_and_1598_padding_counts(self):
        for n in [1, 2, 1598]:
            for size in [4, 8]:
                with self.subTest(n=n, size=size):
                    data = batch(n)
                    mark_source_rows(data)
                    output, metrics = compute_verified_ref(data, Group(size))
                    self.assertEqual(len(output), n)
                    self.assertEqual(metrics['ccpo/phi_transport_padding_rows'], -n % size)
                    torch.testing.assert_close(output.batch['ccpo_phi_feats'], expected_features(data), rtol=0, atol=0)

    def test_packing_rejects_wrong_indices_and_missing_prompt(self):
        data = batch(4).batch
        indices = torch.nonzero(data['attention_mask'].reshape(-1), as_tuple=True)[0]
        hidden = torch.zeros(1, len(indices), 8)
        with self.assertRaisesRegex(ValueError, 'indices'):
            pool_last_prompt(hidden, data, indices.flip(0))
        with self.assertRaisesRegex(ValueError, 'do not match'):
            pool_last_prompt(hidden[:, :-1], data, indices)
        data['attention_mask'][0, -5] = 0
        with self.assertRaisesRegex(ValueError, 'masked'):
            pool_last_prompt(hidden, data, indices)

    def test_capture_failure_removes_hook_and_next_call_is_clean(self):
        data = batch(4)
        mark_source_rows(data)
        data.meta_info.update({VERIFIED: True, 'micro_batch_size': 2, 'temperature': 1., 'use_dynamic_bsz': False})
        worker = actor(True)
        worker._acg_capture_hidden = True
        old = worker.actor_module.table[1, 0].clone()
        worker.actor_module.table[1, 0] = float('inf')
        with self.assertRaisesRegex(ValueError, 'Non-finite frozen'):
            worker.compute_log_prob(data)
        self.assertEqual(len(worker.actor_module.norm._forward_hooks), 0)
        self.assertEqual(worker._acg_hidden_buf, [])
        self.assertEqual(worker.actor_module.calls, 2)  # both FSDP forwards finished before the error
        worker.actor_module.table[1, 0] = old
        worker.compute_log_prob(data)
        self.assertEqual(worker._acg_hidden.shape, (4, 8 + HASH_BYTES))

    def test_legacy_capture_still_returns_unpacketized_hidden_features(self):
        for packed in [False, True]:
            data = batch(9)
            data.meta_info.update(micro_batch_size=2, temperature=1., use_dynamic_bsz=True, max_token_len=16)
            worker = actor(packed)
            worker._acg_capture_hidden = True
            worker.compute_log_prob(data)
            torch.testing.assert_close(worker._acg_hidden, expected_features(data), rtol=0, atol=0)
            self.assertEqual(len(worker.actor_module.norm._forward_hooks), 0)

    def test_verified_capture_rejects_unsupported_capture_instead_of_fallback(self):
        for fault in ['sp', 'disabled', 'missing_norm']:
            data = batch(4)
            mark_source_rows(data)
            data.meta_info.update({VERIFIED: True, 'micro_batch_size': 2, 'temperature': 1., 'use_dynamic_bsz': False})
            worker = actor(True)
            worker._acg_capture_hidden = fault != 'disabled'
            worker.use_ulysses_sp = fault == 'sp'
            if fault == 'missing_norm':
                worker._acg_final_norm = lambda: None
            with self.subTest(fault=fault), self.assertRaisesRegex(ValueError, 'Verified future-progress'):
                worker.compute_log_prob(data)

    def test_recovered_features_preserve_actual_advantage_and_ppo_gradient(self):
        from test_future_progress import rows
        from test_outlook import fixture
        from test_episode_adv_isolation import load_compute_advantage, make_data, policy_gradient
        compute = load_compute_advantage()
        source = fixture()
        data = padded_batch(len(source['index']))
        order = data.batch[SOURCE_ID].numpy()
        source['phi_feats'] = expected_features(batch(len(source['index'])))
        for size in [4, 8]:
            captured, _ = compute_verified_ref(data, Group(size, corruption='drift'))
            for mode in ['mean_norm', 'mean_std_norm']:
                for horizon in [1, 2]:
                    env = dict(ACG_CCPO_PROGRESS_HORIZON=str(horizon), ACG_CCPO_EP_W='0',
                               ACG_CCPO_PROGRESS_WEIGHT='1', ACG_CCPO_FIXED_ANCHOR='0',
                               ACG_CCPO_OUTLOOK_HORIZON='0', ACG_CCPO_OUTLOOK_BETA='0', ACG_CCPO_STEP_NORM='mode')
                    results = []
                    for features in [source['phi_feats'][order], captured.batch['ccpo_phi_feats']]:
                        kw = rows(source, order)
                        kw['phi_feats'] = features
                        training = make_data(kw)
                        with patch.dict(os.environ, env), redirect_stdout(io.StringIO()):
                            compute(training, 'CCPO', gamma=.95, gigpo_mode=mode)
                        results.append((training.batch['advantages'], *policy_gradient(training)))
                        self.assertEqual(training.meta_info['ccpo_diag']['ccpo/progress_phi_duplicate_max_diff'], 0.)
                    for left, right in zip(*results):
                        torch.testing.assert_close(left, right, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
