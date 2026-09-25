"""Release-package parity against the existing trainer, kept outside the release."""
from contextlib import redirect_stdout
import importlib.util
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_future_progress import core, np, torch, rows, load_compute_advantage, policy_gradient
from test_future_progress_active_episode import training_fixture, reward_data

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'official-ccpo/src/ccpo'
spec = importlib.util.spec_from_file_location('ccpo_release', SOURCE / '__init__.py',
                                             submodule_search_locations=[str(SOURCE)])
release = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = release
spec.loader.exec_module(release)
from ccpo_release.torch import compute_verl_advantages


class ReleaseParityTests(unittest.TestCase):
    def test_advantages_components_and_actual_ppo_gradients(self):
        compute = load_compute_advantage()
        settings = dict(_LK_FIX='1', _LAM_FIX='1', _PHI_MODE='hidden+ctx',
                        _CTX_W=1., _WHITEN_K=3, _LOO='1', _GATE='hard',
                        _WMODE='soft', _TAU_ENV='.15', _JW_C=0., _STD_MODE='task',
                        _BACKOFF_TASK=True, _SIM=0., _SIM_BACKOFF=0.,
                        _EDGE_W=0., _TARGET='return')
        cases = {}
        cases['exact-and-fallback'] = training_fixture()
        kw = training_fixture()
        kw['anchor_obs'] = np.array(['room' + str(t % 2) for t in kw['turn_index']])
        cases['repeated-observation'] = kw
        kw = training_fixture()
        kw['phi_feats'] = torch.zeros_like(kw['phi_feats'])
        cases['constant-hidden'] = kw
        kw = training_fixture()
        kw['phi_feats'] = torch.tensor(np.random.default_rng(7).normal(size=(14, 1536)), dtype=torch.float32)
        cases['backbone-width'] = kw
        kw = training_fixture()
        kw['index'][kw['traj_index'] == 'traj3'] = 'solo'
        cases['unsupported-task'] = kw
        kw = training_fixture()
        kw['index'] = np.where(np.isin(kw['traj_index'], ['traj0', 'traj3']), 'q1', 'q2')
        cases['multiple-tasks'] = kw
        for name, kw in cases.items():
            for padded in (False, True):
                indices = np.arange(len(kw['index']))
                if padded:
                    indices = np.random.default_rng(72).permutation(np.r_[indices, 0, 4, 9])
                inputs = rows(kw, indices)
                for benchmark, mode in [('alfworld', 'mean_std_norm'), ('webshop', 'mean_norm')]:
                    for weight in (0, 1):
                        with self.subTest(case=name, padded=padded, benchmark=benchmark, episode=weight):
                            env = dict(ACG_CCPO_PROGRESS_HORIZON='2', ACG_CCPO_PROGRESS_WEIGHT='1',
                                       ACG_CCPO_PROGRESS_HISTORY_WEIGHT='1', ACG_CCPO_EP_W=str(weight),
                                       ACG_CCPO_STEP_NORM='mode', ACG_CCPO_FIXED_ANCHOR='0',
                                       ACG_CCPO_OUTLOOK_HORIZON='0', ACG_CCPO_OUTLOOK_BETA='0',
                                       ACG_CCPO_DUMP='', ACG_EXP_DIR='')
                            data = reward_data(inputs)
                            adapter_data = reward_data(inputs)
                            adapter_data.non_tensor_batch = {
                                k: np.asarray(v, dtype=object) for k,v in adapter_data.non_tensor_batch.items()}
                            adapter_data.non_tensor_batch['episode_lengths'] = inputs['episode_lengths'].astype(float).astype(object)
                            with patch.dict(os.environ, env), patch.multiple(core, **settings), redirect_stdout(io.StringIO()):
                                compute(data, 'CCPO', gamma=.95, gigpo_mode=mode, ccpo_step_tag=1)
                            actual, result = compute_verl_advantages(
                                adapter_data, release.CCPOConfig(benchmark=benchmark, episode_weight=weight))
                            torch.testing.assert_close(actual, data.batch['advantages'], atol=3e-5, rtol=3e-5)
                            standalone = SimpleNamespace(batch={'advantages': actual, 'response_mask': inputs['response_mask']})
                            for expected, got in zip(policy_gradient(data), policy_gradient(standalone)):
                                torch.testing.assert_close(got, expected, atol=3e-6, rtol=3e-5)
                            # Compare raw baselines, potential and credit before benchmark fusion.
                            from ccpo.future_progress import ccpo_future_progress_advantage
                            direct = {k:v for k,v in inputs.items() if k != 'immediate_rewards'}
                            with patch.multiple(core, **settings), redirect_stdout(io.StringIO()):
                                _, diagnostics = ccpo_future_progress_advantage(**direct, horizon=2)
                            payload = diagnostics['progress_payload']
                            for released, original in [('history_baseline', 'history_baseline'),
                                                       ('current_value', 'current_value'),
                                                       ('future_value', 'future_value'),
                                                       ('history', 'history_adv'), ('future', 'progress_normalized')]:
                                np.testing.assert_allclose(getattr(result, released),
                                    payload['arrays'][original][payload['restore']], atol=3e-5, rtol=3e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
