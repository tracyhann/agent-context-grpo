#!/usr/bin/env python3
"""Check audit provenance, scalar replay and Monte Carlo sensitivity on CPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from scripts.analyse_adaptive_progress import write_json


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    errors = {}
    rows = 0
    snapshots = 0
    lone_cases = {}
    hashes = {}

    def compare(label, actual, expected, atol=2e-7):
        actual, expected = np.asarray(actual), np.asarray(expected)
        errors[label] = max(errors.get(label, 0.), float(np.max(abs(actual - expected))))
        np.testing.assert_allclose(actual, expected, atol=atol, rtol=1e-8, err_msg=label)

    for run in plan['runs']:
        config = Path(run['config_path'])
        sha = hashlib.sha256(config.read_bytes()).hexdigest()
        assert sha == run['config_sha256']
        hashes[str(config.relative_to(ROOT))] = sha
        lone_cases[run['tag']] = dict(nonterminal_rows=0, raw_zero_rows=0, variance_zero_rows=0,
                                      original_negative_future_rows=0, original_negative_combined_rows=0)
        for source in run['snapshots']:
            path = Path(source['path'])
            stat = path.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (source['size'], source['mtime_ns'])
            keys = ('uid traj_uid current_value future_value endpoint_index episode_rewards raw_progress '
                    'progress_norm_mean progress_norm_std progress_normalized history_adv combined_pre '
                    'combined_applied eligible live terminal').split()
            with np.load(path, allow_pickle=False) as z:
                a = {k: z[k] for k in keys}
            assert a['eligible'].all() and a['live'].all()
            endpoint = a['endpoint_index']
            nonterminal = endpoint >= 0
            np.testing.assert_array_equal(nonterminal, ~a['terminal'])
            np.testing.assert_array_equal(a['traj_uid'][endpoint[nonterminal]], a['traj_uid'][nonterminal])
            future = a['episode_rewards'].astype(float).copy()
            future[nonterminal] = a['current_value'][endpoint[nonterminal]]
            compare('future_value', future, a['future_value'])
            compare('raw_progress', future - a['current_value'], a['raw_progress'])
            for task in np.unique(a['uid']):
                ix = a['uid'] == task
                progress = a['raw_progress'][ix]
                assert len(progress) >= 2
                compare('task_progress_mean', progress.mean(), a['progress_norm_mean'][ix])
                compare('task_progress_std', progress.std(ddof=1), a['progress_norm_std'][ix])
                compare('task_progress_normalization',
                        (progress - progress.mean()) / (progress.std(ddof=1) + 1e-6),
                        a['progress_normalized'][ix])
            compare('combined_pre', a['history_adv'] + a['progress_normalized'], a['combined_pre'])
            if run['tag'] == 'M11':
                compare('webshop_actor_scalar', a['combined_pre'], a['combined_applied'], atol=2e-6)
            stem = f'{run["tag"]}-{source["step"]:04d}'
            result = json.loads((HERE / 'audit/per_step' / f'{stem}.json').read_text())
            variants = result['variants']
            compare('matched_future_mass', variants['row_snr']['future_mass_retention'],
                    variants['matched_task_scale']['future_mass_retention'], atol=1e-12)
            for label in ['fixed_1', 'fixed_025', 'row_snr_centered', 'matched_task_scale', 'task_nonterminal_snr']:
                compare(label + '_task_mean', variants[label]['task_mean_abs'], 0., atol=1e-12)
            with np.load(HERE / 'audit/row_diagnostics' / f'{stem}.npz', allow_pickle=False) as z:
                mask = z['lone_success'] & ~z['terminal']
                out = lone_cases[run['tag']]
                out['nonterminal_rows'] += int(mask.sum())
                out['raw_zero_rows'] += int(np.sum(mask & (abs(z['raw_progress']) <= 1e-8)))
                out['variance_zero_rows'] += int(np.sum(mask & (z['variance_gate'] <= 1e-12)))
                out['original_negative_future_rows'] += int(np.sum(mask & (z['future'] < -1e-8)))
                out['original_negative_combined_rows'] += int(np.sum(mask & (z['history'] + z['future'] < -1e-8)))
            rows += len(a['uid'])
            snapshots += 1

    sensitivity = []
    all_gate_differences = []
    for p in sorted((HERE / 'sensitivity-b1024/per_step').glob('*.json')):
        main_result = json.loads((HERE / 'audit/per_step' / p.name).read_text())
        other_result = json.loads(p.read_text())
        arrays = []
        for output in ['audit', 'sensitivity-b1024']:
            with np.load(HERE / output / 'row_diagnostics' / (p.stem + '.npz'), allow_pickle=False) as z:
                arrays.append({k: z[k] for k in ['uid', 'traj_uid', 'raw_progress', 'gate']})
        for key in ['uid', 'traj_uid', 'raw_progress']:
            np.testing.assert_array_equal(arrays[0][key], arrays[1][key])
        diff = abs(arrays[0]['gate'] - arrays[1]['gate'])
        all_gate_differences.append(diff)
        record = dict(run=other_result['tag'], step=other_result['step'], rows=len(diff),
                      gate_row_mean_abs_difference=float(diff.mean()),
                      gate_row_p95_abs_difference=float(np.quantile(diff, .95)),
                      gate_row_max_abs_difference=float(diff.max()))
        for field in ['weight_mean', 'future_mass_retention', 'terminal_mass_share', 'task_mean_abs', 'frozen_norm_noise_fraction']:
            record[field + '_abs_difference'] = abs(main_result['variants']['row_snr'][field] - other_result['variants']['row_snr'][field])
        sensitivity.append(record)
    assert len(sensitivity) == 11
    differences = np.concatenate(all_gate_differences)
    sensitivity_summary = dict(
        comparison='256 versus 1024 bootstrap draws per independent split; same empirical data, same seed prefix; not independent datasets.',
        batches=len(sensitivity), rows=len(differences),
        pooled_row_gate_mean_abs_difference=float(differences.mean()),
        pooled_row_gate_p95_abs_difference=float(np.quantile(differences, .95)),
        max_batch_weight_mean_difference=max(r['weight_mean_abs_difference'] for r in sensitivity),
        max_batch_future_mass_retention_difference=max(r['future_mass_retention_abs_difference'] for r in sensitivity),
        max_batch_terminal_share_difference=max(r['terminal_mass_share_abs_difference'] for r in sensitivity),
        per_batch=sensitivity)
    write_json(HERE / 'sensitivity-summary.json', sensitivity_summary)
    unit = subprocess.run([sys.executable, 'tests/test_adaptive_progress_analysis.py'], cwd=ROOT,
                          capture_output=True, text=True, check=True)
    for path in ['scripts/analyse_adaptive_progress.py', 'tests/test_adaptive_progress_analysis.py',
                 'ccpo/core_ccpo.py', 'ccpo/future_progress.py', str((HERE / 'plan.json').relative_to(ROOT)),
                 str((HERE / 'validate.py').relative_to(ROOT)), str((HERE / 'render.py').relative_to(ROOT))]:
        hashes[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    point_errors = [max(json.loads(p.read_text())['parity'].values()) for p in (HERE / 'audit/per_step').glob('*.json')]
    result = dict(status='passed', snapshots=snapshots, rows=rows,
                  source_metadata_unchanged=True, config_sha256_matches_manifest=True,
                  all_rows_eligible_and_live=True, max_point_estimator_error=max(point_errors),
                  scalar_max_errors=errors, lone_success_diagnostic=lone_cases,
                  sensitivity={k: v for k, v in sensitivity_summary.items() if k != 'per_batch'},
                  unit_tests=unit.stdout + unit.stderr, sha256=hashes)
    write_json(HERE / 'VALIDATION.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['sha256', 'unit_tests']}, indent=2))


if __name__ == '__main__':
    main()
