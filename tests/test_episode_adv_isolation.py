"""CPU regression: unweighted episode diagnostics cannot change no-EP policy gradients.

Extract the live trainer's function to avoid starting Ray; use the actual CCPO
estimators and PPO loss. Perturb only the episode advantage, since episode
rewards legitimately remain inputs to the step-credit estimator.
"""
import ast
from collections import defaultdict
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Reuse the deterministic multi-trajectory fixture and its CPU-only setup.
from test_outlook import core, fixture, np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'verl-agent'))
from gigpo import core_gigpo
from verl.trainer.ppo.core_algos import compute_policy_loss


def load_compute_advantage():
    path = ROOT / 'verl-agent/verl/trainer/ppo/ray_trainer.py'
    overlay = ROOT / 'patches/verl-agent/verl/trainer/ppo/ray_trainer.py'
    if path.read_bytes() != overlay.read_bytes():
        raise AssertionError('Live trainer differs from the checked-in overlay')
    function = next(node for node in ast.parse(path.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'compute_advantage')
    names = ('GAE', 'GRPO', 'GRPO_PASSK', 'REINFORCE_PLUS_PLUS_BASELINE',
             'REINFORCE_PLUS_PLUS', 'REMAX', 'RLOO', 'CCPO', 'GiGPO')
    namespace = dict(DataProto=object, torch=torch, np=np, os=os,
                     defaultdict=defaultdict, core_gigpo=core_gigpo,
                     AdvantageEstimator=SimpleNamespace(**{name: name for name in names}))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['compute_advantage']


def make_data(kw):
    return SimpleNamespace(
        batch={'step_rewards': kw['step_rewards'].clone(),
               'response_mask': kw['response_mask'].clone(),
               'ccpo_phi_feats': kw['phi_feats'].clone(),
               'token_level_rewards': torch.zeros_like(kw['response_mask'])},
        non_tensor_batch={'anchor_obs': kw['anchor_obs'], 'uid': kw['index'],
                          'traj_uid': kw['traj_index'], 'episode_rewards': kw['episode_rewards'],
                          'episode_lengths': kw['episode_lengths'],
                          'is_action_valid': kw['is_action_valid'],
                          'ccpo_turn_index': kw['turn_index'], 'rewards': kw['immediate_rewards']},
        meta_info={})


def policy_gradient(data):
    mask = data.batch['response_mask']
    logits = torch.linspace(-1, 1, mask.numel() * 2).reshape(*mask.shape, 2).requires_grad_()
    log_prob = logits.log_softmax(dim=-1)[..., 0]
    loss, *_ = compute_policy_loss(
        old_log_prob=log_prob.detach(), log_prob=log_prob,
        advantages=data.batch['advantages'], response_mask=mask,
        cliprange=.2, loss_agg_mode='token-mean')
    return loss.detach(), torch.autograd.grad(loss, logits)[0]


class EpisodeAdvIsolationTests(unittest.TestCase):
    def test_zero_episode_weight_preserves_loss_and_gradient(self):
        compute_advantage = load_compute_advantage()
        kw = fixture()
        # Include masked response tokens as in real variable-length rollouts.
        kw['response_mask'][::2, -1] = 0
        raw = torch.linspace(-2, 2, len(kw['step_rewards']))[:, None] * kw['response_mask']
        perturbed = (raw * 31 + 100) * kw['response_mask']
        for mode in ('mean_std_norm', 'mean_norm'):
            for variant, edge, horizon, beta in (('original', 1, 0, 0),
                                                ('noedge', 0, 0, 0),
                                                ('outlook', 0, 2, .25)):
                with self.subTest(mode=mode, variant=variant):
                    outputs = {}
                    for ep_weight in (0, 1):
                        results = []
                        for episode_adv in (raw, perturbed):
                            data = make_data(kw)
                            settings = {'ACG_CCPO_EP_W': str(ep_weight),
                                        'ACG_CCPO_STEP_NORM': 'mode',
                                        'ACG_CCPO_OUTLOOK_HORIZON': str(horizon),
                                        'ACG_CCPO_OUTLOOK_BETA': str(beta)}
                            with patch.dict(os.environ, settings), patch.object(core, '_EDGE_W', edge), \
                                    patch.object(core_gigpo, 'episode_norm_reward', return_value=episode_adv), \
                                    redirect_stdout(io.StringIO()):
                                compute_advantage(data, 'CCPO', gamma=.95, gigpo_mode=mode)
                            results.append((data, *policy_gradient(data)))
                        outputs[ep_weight] = results
                    (first, loss_a, grad_a), (second, loss_b, grad_b) = outputs[0]
                    self.assertGreater(torch.count_nonzero(grad_a).item(), 0)
                    self.assertTrue(torch.isfinite(grad_a).all())
                    for key in ('advantages', 'returns'):
                        torch.testing.assert_close(first.batch[key], second.batch[key], rtol=0, atol=0)
                    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
                    torch.testing.assert_close(grad_a, grad_b, rtol=0, atol=0)
                    for key in ('ccpo/adv_ep_absmean', 'ccpo/adv_ep_over_cc'):
                        self.assertNotEqual(first.meta_info['ccpo_diag'][key],
                                            second.meta_info['ccpo_diag'][key])
                    # Positive control: the same perturbation MUST affect gradients
                    # when the episode channel is enabled, even though it is detached.
                    self.assertFalse(torch.allclose(outputs[1][0][2], outputs[1][1][2]))

    def test_raw_episode_normalization_is_detached(self):
        kw = fixture()
        rewards = torch.zeros_like(kw['response_mask'])
        rewards[:, -1] = torch.as_tensor(kw['episode_rewards'], dtype=rewards.dtype)
        rewards.requires_grad_()
        for remove_std in (False, True):
            with self.subTest(remove_std=remove_std):
                advantage = core_gigpo.episode_norm_reward(
                    rewards, kw['response_mask'], kw['index'], kw['traj_index'],
                    remove_std=remove_std)
                self.assertFalse(advantage.requires_grad)
                self.assertIsNone(advantage.grad_fn)
                self.assertGreater(advantage.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
