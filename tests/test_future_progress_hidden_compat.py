"""Hidden-only future progress: registry composition, runtime and launcher."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_future_progress import (arguments, core, np, torch,
    ccpo_future_progress_advantage, load_compute_advantage, make_data, fixture)

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


registry = load('hidden_compat_registry', ROOT/'ablations/ablations.py')
launcher = load('hidden_compat_launcher', ROOT/'scripts/exp_run.py')


class HiddenFutureProgressTests(unittest.TestCase):
    def test_generic_noctx_delta_composes_with_h1_h2_and_matches_zero_context_weight(self):
        for horizon, method in [(1, 'attncred-context-future-progress'),
                                (2, 'attncred-context-future-progress-h2')]:
            with self.subTest(horizon=horizon):
                _, cfg = registry.arms.build(method, 'alfworld', '1.5b',
                    extra=registry.ABLATIONS['no-context-vector']['delta'])
                self.assertEqual(cfg['ccpo_phi'], 'hidden')
                self.assertEqual(cfg['ccpo_progress_horizon'], horizon)
                launcher.validate_future_progress_config(cfg)
                kw = arguments()
                kw['phi_feats'] = torch.tensor(np.random.default_rng(72).normal(
                    size=(len(kw['index']), 1536)), dtype=torch.float32)
                with patch.object(core, '_PHI_MODE', 'hidden+ctx'), patch.object(core, '_CTX_W', 0.):
                    expected, expected_diag = ccpo_future_progress_advantage(**kw, horizon=horizon)
                with patch.object(core, '_PHI_MODE', cfg['ccpo_phi']), patch.object(core, '_CTX_W', 1.):
                    actual, actual_diag = ccpo_future_progress_advantage(**kw, horizon=horizon)
                    changed = dict(kw, ctx_override=[dict(t=30, n_unique=25, progress=1., revisit=1.)
                                                    for _ in kw['index']])
                    perturbed, _ = ccpo_future_progress_advantage(**changed, horizon=horizon)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(perturbed, expected, rtol=0, atol=0)
                self.assertFalse(actual.requires_grad)
                a = actual_diag['progress_payload']['arrays']
                b = expected_diag['progress_payload']['arrays']
                self.assertEqual(a['current_phi'].shape[1], 1536)
                for key in ('current_phi', 'future_phi', 'history_baseline', 'history_adv',
                            'current_value', 'future_value', 'raw_progress', 'progress_normalized',
                            'current_lambda_k', 'future_lambda_k', 'endpoint_index'):
                    np.testing.assert_array_equal(a[key], b[key])

    def test_actual_trainer_advantage_path_accepts_both_hidden_spellings(self):
        compute = load_compute_advantage()
        for mode in ('mean_norm', 'mean_std_norm'):
            for horizon in (1, 2):
                result = []
                for phi, weight in [('hidden', 1.), ('hidden+ctx', 0.)]:
                    with self.subTest(mode=mode, horizon=horizon, phi=phi), tempfile.TemporaryDirectory() as d:
                        env = dict(ACG_CCPO_PROGRESS_HORIZON=str(horizon), ACG_CCPO_PROGRESS_WEIGHT='1',
                                   ACG_CCPO_EP_W='0', ACG_EXP_DIR=d, ACG_CCPO_PROGRESS_SNAPSHOT_EVERY='1')
                        with patch.dict(os.environ, env), patch.object(core, '_PHI_MODE', phi), \
                             patch.object(core, '_CTX_W', weight), contextlib.redirect_stdout(io.StringIO()):
                            data = compute(make_data(fixture()), 'CCPO', gamma=.95,
                                           gigpo_mode=mode, ccpo_step_tag=1)
                        result.append(data.batch['advantages'].clone())
                        self.assertTrue(torch.isfinite(result[-1]).all())
                        self.assertFalse(result[-1].requires_grad)
                torch.testing.assert_close(result[0], result[1], rtol=0, atol=0)

    def test_hidden_mode_still_rejects_missing_and_nonfinite_features(self):
        with patch.object(core, '_PHI_MODE', 'hidden'):
            kw = arguments(); kw['phi_feats'] = None
            with self.assertRaisesRegex(ValueError, 'phi_feats supplied'):
                ccpo_future_progress_advantage(**kw, horizon=2)
            for value in (float('nan'), float('inf'), -float('inf')):
                kw = arguments(); kw['phi_feats'][0, 0] = value
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Non-finite frozen features'):
                    ccpo_future_progress_advantage(**kw, horizon=2)
        for phi in ('bow', 'hidden-typo'):
            with patch.object(core, '_PHI_MODE', phi), self.assertRaisesRegex(ValueError, 'frozen hidden features'):
                ccpo_future_progress_advantage(**arguments(), horizon=2)

    def test_launcher_rejects_invalid_phi_before_files_or_subprocesses(self):
        for phi in ('bow', 'hidden-typo'):
            with tempfile.TemporaryDirectory() as root, patch.object(launcher, 'ROOT', root), \
                 patch.object(launcher.subprocess, 'run', side_effect=AssertionError('subprocess started')), \
                 patch.object(launcher.subprocess, 'Popen', side_effect=AssertionError('trainer started')), \
                 patch('sys.argv', ['exp_run.py', '--name', 'invalid-hidden', '--arm', 'ccpo',
                                    '--set', 'ccpo_progress_horizon=2', '--set', 'ccpo_phi='+phi]), \
                 contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    launcher.main()
                self.assertEqual(error.exception.code, 2)
                self.assertIn('ccpo_phi=hidden or hidden+ctx', stderr.getvalue())
                self.assertFalse((Path(root)/'experiments').exists())
        # Programmatic preparation also fails before command construction.
        with self.assertRaisesRegex(ValueError, r'ccpo_phi=hidden or hidden\+ctx'):
            launcher.build_command(dict(arm='ccpo', ccpo_progress_horizon=2, ccpo_phi='bow'), '/unused')


if __name__ == '__main__':
    unittest.main()
