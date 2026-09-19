"""Check the statistical units and covariance of the offline fusion audit."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analyse_adaptive_progress import TaskReadout, paired_progress, snr_gate


def fixture(labels=None, observation=None):
    return TaskReadout(np.zeros((6, 2)), np.array(labels if labels is not None else [10, 12, 20, 22, 30, 32.]),
                       np.array(observation if observation is not None else ['A', 'B'] * 3),
                       np.array(['own', 'own', 'peer1', 'peer1', 'peer2', 'peer2']))


class AdaptiveProgressAnalysisTests(unittest.TestCase):
    def test_point_readout_matches_hand_calculated_kernel_and_prior(self):
        values, kernels, support, lam = fixture().read(np.array([0, 1]), np.ones((2, 1)))
        np.testing.assert_allclose(kernels[:, 0], [25, 27])
        np.testing.assert_allclose(values[:, 0], [25.5, 26.5])
        np.testing.assert_array_equal(support[:, 0], [2, 2])
        np.testing.assert_allclose(lam[:, 0], [.5, .5])

    def test_joint_resampling_cancels_perfectly_correlated_endpoint_noise(self):
        counts = np.array([[2, 1, 0], [0, 1, 2]])
        values = fixture().read(np.array([0, 1]), counts)[0]
        progress = paired_progress(values, np.array([1, -1]), np.array([10, 10]))
        self.assertGreater(values[0].var(), 10)
        self.assertGreater(values[1].var(), 10)
        np.testing.assert_allclose(progress[0], 1)
        self.assertAlmostEqual(progress[0].var(), 0)
        np.testing.assert_allclose(progress[1], 10 - values[1])

    def test_whole_query_trajectory_excluded_from_both_endpoints(self):
        counts = np.array([[2, 1, 0], [0, 1, 2]])
        baseline = fixture().read(np.array([0, 1]), counts)[0]
        changed = fixture(labels=[-100, 200, 20, 22, 30, 32]).read(np.array([0, 1]), counts)[0]
        np.testing.assert_array_equal(baseline, changed)

    def test_missing_observation_peers_uses_whole_task_kernel(self):
        result = fixture(observation=['unique', 'B', 'A', 'B', 'A', 'B']).read(np.array([0, 1]), np.ones((2, 1)))
        self.assertEqual(result[0][0, 0], 26)
        self.assertEqual(result[2][0, 0], 0)
        self.assertEqual(result[3][0, 0], 1)

    def test_gate_zero_signal_bounds_and_sign_symmetry(self):
        result = snr_gate(np.array([0., 0., 2., -2., 1.]), np.array([0., 1., 1., 1., 2.]))
        np.testing.assert_allclose(result, [0, 0, .75, .75, 0], atol=1e-12)
        self.assertTrue(np.all((result >= 0) & (result <= 1)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
