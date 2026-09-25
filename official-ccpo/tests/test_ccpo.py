"""Independent checks of the method, its support rules, and temporal alignment."""
from dataclasses import fields, replace
import unittest

import numpy as np

from ccpo import CCPOConfig, RolloutBatch, build_features, compute_advantages, context_statistics


def fixture():
    lengths = np.repeat([4, 4, 3, 3], [4, 4, 3, 3])
    turns = np.concatenate([np.arange(n) for n in [4, 4, 3, 3]])
    outcomes = np.repeat([10., 0., 10., 0.], [4, 4, 3, 3])
    penalties = np.zeros(len(turns)); penalties[1] = .1
    return RolloutBatch(
        task_ids=np.array(['task'] * len(turns)),
        trajectory_ids=np.repeat(['a', 'b', 'c', 'd'], [4, 4, 3, 3]),
        observations=np.array(['room-' + str(t % 2) for t in turns]),
        turns=turns, lengths=lengths,
        hidden=np.random.default_rng(5).normal(size=(len(turns), 8)),
        returns=.95 ** (lengths - turns - 1) * outcomes - penalties,
        episode_rewards=outcomes, episode_scores=outcomes - penalties,
    )


def take(batch, rows):
    return RolloutBatch(**{field.name: np.asarray(getattr(batch, field.name))[rows] for field in fields(batch)})


class FeatureTests(unittest.TestCase):
    def test_statistics_use_the_prefix_exact_text_and_original_row_order(self):
        obs = np.array(['A', 'B', 'A', 'C', 'a'])
        turns = np.arange(5); order = np.array([3, 0, 4, 2, 1])
        actual = context_statistics(obs[order], ['traj'] * 5, turns[order])
        expected = np.array([[0, 1, 0, 1], [1, 2, 0, 1], [2, 2, 1, 2/3],
                             [3, 3, 0, 3/4], [4, 4, 0, 4/5]])
        np.testing.assert_allclose(actual, expected[order])

    def test_summary_encoding_dimensions_saturation_and_block_norms(self):
        hidden = np.random.default_rng(9).normal(size=(8, 16))
        context = np.tile([15, 12, 1, .5], (8, 1))
        features = build_features(hidden, context)
        self.assertEqual(features.shape, (8, 53))
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1)
        np.testing.assert_allclose(np.linalg.norm(features[:, :16], axis=1), 1/np.sqrt(2))
        np.testing.assert_allclose(np.linalg.norm(features[:, 16:], axis=1), 1/np.sqrt(2))
        expected = np.r_[np.ones(6), np.zeros(6), np.ones(5), np.zeros(7),
                         np.ones(6), np.zeros(6), 1]
        np.testing.assert_allclose(features[0, 16:], expected / np.linalg.norm(expected) / np.sqrt(2))
        a = build_features(hidden, np.tile([30, 25, 1, 1], (8, 1)))
        b = build_features(hidden, np.tile([80, 70, 1, 1], (8, 1)))
        np.testing.assert_array_equal(a, b)

    def test_centering_and_pc_removal_keep_dimensions_and_do_not_mutate_inputs(self):
        batch = fixture(); hidden = batch.hidden.copy()
        context = context_statistics(batch.observations, batch.trajectory_ids, batch.turns)
        a = build_features(hidden, context)
        b = build_features(hidden + np.arange(8), context)
        np.testing.assert_allclose(a, b, atol=1e-13)
        np.testing.assert_array_equal(hidden, batch.hidden)
        with self.assertRaises(ValueError):
            build_features(hidden * np.nan, context)


class CreditTests(unittest.TestCase):
    def test_manual_two_trajectory_baselines_and_terminal_clipping(self):
        batch = RolloutBatch(
            task_ids=['q'] * 6, trajectory_ids=['a'] * 3 + ['b'] * 3,
            observations=['x', 'y', 'z'] * 2, turns=np.tile(np.arange(3), 2),
            lengths=np.full(6, 3), hidden=np.zeros((6, 4)),
            returns=np.array([9.025, 9.5, 10., 0., 0., 0.]),
            episode_rewards=np.array([10.] * 3 + [0.] * 3),
            episode_scores=np.array([10.] * 3 + [0.] * 3),
        )
        result = compute_advantages(batch, CCPOConfig(benchmark='webshop'))
        np.testing.assert_allclose(result.history_baseline, [0, 0, 0, 9.025, 9.5, 10])
        np.testing.assert_allclose(result.current_value, [0, 0, 0, 10*.95**3, 10*.95**2, 9.5])
        np.testing.assert_allclose(result.future_value, [0, 10, 10, 9.5, 0, 0])
        np.testing.assert_allclose(result.future_delta, result.future_value - result.current_value)
        np.testing.assert_array_equal(result.peer_trajectories, 1)
        self.assertFalse(result.used_task_fallback.any())

    def test_same_trajectory_targets_cannot_enter_query_baselines(self):
        batch = fixture(); own = batch.trajectory_ids == 'a'
        altered_returns = batch.returns.copy(); altered_returns[own] += 6
        altered_rewards = batch.episode_rewards.copy(); altered_rewards[own] = 0
        before = compute_advantages(batch)
        after = compute_advantages(replace(batch, returns=altered_returns, episode_rewards=altered_rewards))
        np.testing.assert_array_equal(before.history_baseline[own], after.history_baseline[own])
        np.testing.assert_array_equal(before.current_value[own], after.current_value[own])
        self.assertGreater(np.max(abs(before.current_value[~own] - after.current_value[~own])), .1)

    def test_task_backoff_and_no_cross_task_leakage(self):
        batch = fixture()
        unique = replace(batch, observations=np.array(['unique-' + str(i) for i in range(len(batch.turns))]))
        result = compute_advantages(unique)
        self.assertTrue(result.used_task_fallback.all())
        self.assertTrue(result.supported.all())
        # Isolating each trajectory into its own task leaves no legal peers.
        isolated = replace(unique, task_ids=unique.trajectory_ids.copy())
        result = compute_advantages(isolated)
        np.testing.assert_array_equal(result.advantages, 0)
        self.assertFalse(result.supported.any())
        self.assertTrue(np.isnan(result.current_value).all())

    def test_horizon_difference_telescopes_before_normalization(self):
        batch = fixture()
        one = compute_advantages(batch, CCPOConfig(horizon=1))
        two = compute_advantages(batch, CCPOConfig(horizon=2))
        expected = one.future_delta.copy()
        for i, turn in enumerate(batch.turns):
            if turn + 1 < batch.lengths[i]:
                expected[i] += one.future_delta[i+1]
        np.testing.assert_allclose(two.future_delta, expected, atol=1e-14)
        np.testing.assert_allclose(two.future.mean(), 0, atol=1e-14)

    def test_episode_modes_and_benchmark_normalization(self):
        batch = fixture()
        for benchmark in ['alfworld', 'webshop']:
            off = compute_advantages(batch, CCPOConfig(benchmark=benchmark))
            on = compute_advantages(batch, CCPOConfig(benchmark=benchmark, episode_weight=1))
            episode = batch.episode_scores - batch.episode_scores.mean()
            contextual = off.history + off.future
            if benchmark == 'alfworld':
                episode /= batch.episode_scores.std(ddof=1) + 1e-6
                contextual = (contextual - contextual.mean()) / (contextual.std(ddof=1) + 1e-6)
            np.testing.assert_allclose(on.episode, episode, atol=1e-14)
            np.testing.assert_allclose(off.advantages, contextual, atol=1e-14)
            np.testing.assert_allclose(on.advantages, contextual + episode, atol=1e-14)
            np.testing.assert_array_equal(on.contextual, off.contextual)

    def test_disabled_episode_scores_do_not_affect_contextual_credit(self):
        batch = fixture()
        alternate = replace(batch, episode_scores=np.arange(len(batch.turns)) ** 2)
        for benchmark in ['alfworld', 'webshop']:
            cfg = CCPOConfig(benchmark=benchmark)
            a, b = compute_advantages(batch, cfg), compute_advantages(alternate, cfg)
            np.testing.assert_array_equal(a.advantages, b.advantages)
            self.assertFalse(np.allclose(a.episode, b.episode))

    def test_shuffle_and_padding_do_not_add_peers_or_change_future_moments(self):
        batch = fixture(); n = len(batch.turns)
        order = np.random.default_rng(3).permutation(np.r_[np.arange(n), 0, 0, 5])
        for benchmark in ['alfworld', 'webshop']:
            cfg = CCPOConfig(benchmark=benchmark)
            base = compute_advantages(batch, cfg)
            padded = compute_advantages(take(batch, order), cfg)
            for name in ['history', 'future', 'current_value', 'future_value', 'peer_trajectories']:
                np.testing.assert_allclose(getattr(padded, name), getattr(base, name)[order], atol=1e-13)
            expected = (base.history + base.future)[order]
            if benchmark == 'alfworld':
                expected = (expected - expected.mean()) / (expected.std(ddof=1) + 1e-6)
            np.testing.assert_allclose(padded.advantages, expected, atol=1e-13)

    def test_singleton_task_preserves_episode_helper_convention(self):
        batch = RolloutBatch(['q'], ['a'], ['obs'], np.array([0]), np.array([1]),
                             np.ones((1, 4)), np.array([10.]), np.array([10.]), np.array([10.]))
        for benchmark, expected in [('alfworld', 10/(1+1e-6)), ('webshop', 10.)]:
            result = compute_advantages(batch, CCPOConfig(benchmark=benchmark, episode_weight=1))
            self.assertAlmostEqual(result.advantages[0], expected)
            self.assertEqual(result.contextual[0], 0.)

    def test_invalid_metadata_duplicate_features_and_nonfinite_rows_fail(self):
        batch = fixture()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            compute_advantages(take(batch, np.arange(1, len(batch.turns))))
        padded = take(batch, np.r_[np.arange(len(batch.turns)), 0])
        bad = padded.hidden.copy(); bad[-1, 0] += 1
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            compute_advantages(replace(padded, hidden=bad))
        bad[-1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            compute_advantages(replace(padded, hidden=bad))
        bad_tasks = batch.task_ids.copy(); bad_tasks[0] = 'else'
        with self.assertRaisesRegex(ValueError, 'globally unique'):
            compute_advantages(replace(batch, task_ids=bad_tasks))
        with self.assertRaises(ValueError):
            compute_advantages(replace(batch, turns=batch.turns.astype(float)))
        with self.assertRaises(ValueError):
            compute_advantages(replace(batch, episode_rewards=np.ones(len(batch.turns))))

    def test_invalid_configuration_is_rejected(self):
        for change in [dict(benchmark='other'), dict(episode_weight=.25), dict(horizon=0),
                       dict(horizon=True), dict(horizon=1.5), dict(gamma=np.nan),
                       dict(gamma=1.1), dict(kernel_scale=0), dict(remove_top_pcs=-1)]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                CCPOConfig(**change)


if __name__ == '__main__':
    unittest.main()
