#!/usr/bin/env python3
"""CPU checks for the pooled validation rates and the Search-R1 report.

The trainer's val/<key> metrics average the validation batches' means. On Search-R1
the batches are unequal and a dataset spans batch boundaries, so that is not the
rate over questions. These tests run the overlay's own code on stand-in data:

1  pooled_success: the rate over episodes, against hand-computed values
2  the rollout loop attaches the per-episode values on the validation path only
3  _validate, fed by that rollout loop: val/<key> keeps its definition, and
   val/<key>/pooled and val/<key>/episodes equal the rates and counts over episodes
4  scripts/report_search_r1.py: unweighted mean over datasets, overall rate, and
   refusal of runs validated on different questions

    python3 tests/test_pooled_validation.py

Section 3 needs torch and is skipped without it.
"""
import ast
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from typing import Dict, List
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts')]
import report_search_r1 as report

try:
    import torch
except ImportError:
    torch = None

OVERLAY = ROOT/'patches/verl-agent'
TRAINER = 'verl/trainer/ppo/ray_trainer.py'
ROLLOUT = 'agent_system/multi_turn_rollout/rollout_loop.py'


def overlay_functions(relative, names, namespace):
    """Compile named functions or methods of an overlay file without importing its runtime."""
    source = OVERLAY/relative
    found = {}
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            assert node.name not in found, f'{node.name} is defined twice in {relative}'
            found[node.name] = node
    assert set(found) == set(names), f'{sorted(set(names) - set(found))} not found in {relative}'
    exec(compile(ast.Module(body=[found[n] for n in names], type_ignores=[]), str(source), 'exec'), namespace)
    return [namespace[n] for n in names]


def pooled_success():
    return overlay_functions(TRAINER, ['pooled_success'], dict(np=np))[0]


# ---------------------------------------------------------------------------
# 1. the pooled rate
# ---------------------------------------------------------------------------
class PooledRateTests(unittest.TestCase):
    def test_rate_and_count_over_episodes_of_unequal_batches(self):
        pooled = pooled_success()([
            {'success_rate': np.array([1., 1., 0., 0.]), 'a_success_rate': [1., 1., 0.], 'b_success_rate': [0.]},
            {'success_rate': np.array([1.]), 'b_success_rate': np.array([1.])},
        ])
        self.assertEqual(pooled, {'success_rate': (3 / 5, 5), 'a_success_rate': (2 / 3, 3),
                                  'b_success_rate': (1 / 2, 2)})
        # The mean of the two batch means would be (0.5 + 1) / 2 for the overall rate.
        self.assertNotAlmostEqual(pooled['success_rate'][0], .75)

    def test_equal_batches_agree_with_the_mean_of_batch_means(self):
        batches = [{'success_rate': [1., 0., 0., 1.]}, {'success_rate': [1., 1., 1., 0.]}]
        rate, episodes = pooled_success()(batches)['success_rate']
        self.assertAlmostEqual(rate, np.mean([np.mean(b['success_rate']) for b in batches]))
        self.assertEqual(episodes, 8)

    def test_types_are_plain_and_empty_input_gives_nothing(self):
        rate, episodes = pooled_success()([{'k': np.array([1, 0, 1], dtype=np.float32)}])['k']
        self.assertIs(type(rate), float); self.assertIs(type(episodes), int)
        self.assertEqual(pooled_success()([]), {})
        self.assertEqual(pooled_success()([{'k': np.array([])}, {}]), {})


# ---------------------------------------------------------------------------
# stand-ins for the runtime the overlay code touches
# ---------------------------------------------------------------------------
class Proto:
    """The parts of verl's DataProto that validation and the rollout loop use."""
    def __init__(self, batch, non_tensor_batch):
        self.batch, self.non_tensor_batch, self.meta_info = batch, non_tensor_batch, {}

    @classmethod
    def from_single_dict(cls, data, meta_info=None):
        tensors = {k: v for k, v in data.items() if torch is not None and isinstance(v, torch.Tensor)}
        return cls(tensors, {k: v for k, v in data.items() if k not in tensors})

    def repeat(self, repeat_times, interleave):
        assert repeat_times == 1
        return self

    def pop(self, batch_keys, non_tensor_batch_keys):
        return Proto({k: self.batch.pop(k) for k in batch_keys},
                     {k: self.non_tensor_batch.pop(k) for k in non_tensor_batch_keys})


def collate(rows):
    out = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        out[key] = torch.stack(values) if isinstance(values[0], torch.Tensor) else np.array(values)
    return out


# Nine questions in file order, validated four at a time: source `a` spans two
# batches and the last batch holds one question, as on the Search-R1 test file.
SOURCES = ['a'] * 5 + ['b'] * 3 + ['c']
WON = [1, 1, 1, 0, 0, 1, 0, 1, 1]
TURNS = [1, 2, 1, 3, 2, 1, 1, 2, 1]
VAL_BATCH = 4


def collector(attach=True):
    """The overlay's multi_turn_loop and gather_rollout_data around a scripted rollout."""
    namespace = dict(np=np, torch=torch, List=List, Dict=Dict, DataProto=Proto, collate_fn=collate,
                     EnvironmentManagerBase=object)
    multi_turn_loop, gather = overlay_functions(ROLLOUT, ['multi_turn_loop', 'gather_rollout_data'], namespace)

    class Collector:
        config = SimpleNamespace(algorithm=SimpleNamespace(filter_groups=SimpleNamespace(enable=False)),
                                 env=SimpleNamespace(rollout=SimpleNamespace(n=1)))
        gather_rollout_data = gather

        def vanilla_multi_turn_loop(self, gen_batch, actor_rollout_wg, envs):
            questions = [int(k['question']) for k in gen_batch.non_tensor_batch['env_kwargs']]
            rows, success = [], {'success_rate': []}
            for q in questions:
                # One padded turn after the episode ended, as the real loop leaves behind.
                rows.append([dict(traj_uid=f't{q}', active_masks=turn < TURNS[q], data_source=SOURCES[q],
                                  responses=torch.zeros(3, dtype=torch.long)) for turn in range(TURNS[q] + 1)])
                success['success_rate'].append(float(WON[q]))
                success.setdefault(f'{SOURCES[q]}_success_rate', []).append(float(WON[q]))
            return (rows, np.array([10. * WON[q] for q in questions]), np.array([float(TURNS[q]) for q in questions]),
                    {k: np.array(v) for k, v in success.items()}, np.array([f't{q}' for q in questions]),
                    np.array([TURNS[q] - 1 for q in questions]))

        def multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=True):
            out = multi_turn_loop(self, gen_batch, actor_rollout_wg, envs, is_train=is_train)
            if not attach:      # a rollout loop that predates the pooled rates
                out.meta_info.pop('success_episodes', None)
            return out

    return Collector()


def question_batch(questions):
    n = len(questions)
    return {'input_ids': torch.zeros(n, 2, dtype=torch.long), 'attention_mask': torch.ones(n, 2, dtype=torch.long),
            'position_ids': torch.zeros(n, 2, dtype=torch.long), 'raw_prompt_ids': np.zeros(n),
            'data_source': np.array([SOURCES[q] for q in questions]),
            'env_kwargs': np.array([{'question': q} for q in questions], dtype=object)}


def validate(attach=True):
    """Run the overlay's _validate over the nine questions."""
    validate_fn, _ = overlay_functions(TRAINER, ['_validate', 'pooled_success'],
                                       dict(np=np, torch=torch, DataProto=Proto, print=lambda *a, **k: None))
    order = list(range(len(SOURCES)))
    trainer = SimpleNamespace(
        val_dataloader=[question_batch(order[i:i + VAL_BATCH]) for i in range(0, len(order), VAL_BATCH)],
        config=SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(rollout=SimpleNamespace(val_kwargs=SimpleNamespace(n=1, do_sample=False))),
            reward_model=SimpleNamespace(enable=False)),
        tokenizer=SimpleNamespace(decode=lambda ids, skip_special_tokens: 'text', eos_token_id=0, pad_token_id=0),
        traj_collector=collector(attach), actor_rollout_wg=None, val_envs=None,
        val_reward_fn=lambda batch, return_dict: {'reward_tensor': torch.tensor(
            [[float(r), 0.] for r in batch.non_tensor_batch['episode_rewards']])},
        _maybe_log_val_generations=lambda **kwargs: None)
    return validate_fn(trainer)


def batch_mean_of_means(source=None):
    """val/<key> as the trainer defines it: each batch's mean, averaged over batches."""
    means = []
    for start in range(0, len(SOURCES), VAL_BATCH):
        won = [WON[q] for q in range(start, min(start + VAL_BATCH, len(SOURCES)))
               if source is None or SOURCES[q] == source]
        if won:
            means.append(np.mean(won))
    return float(np.mean(means))


def rate_over_questions(source=None):
    won = [w for w, s in zip(WON, SOURCES) if source is None or s == source]
    return sum(won) / len(won), len(won)


# ---------------------------------------------------------------------------
# 2. the rollout loop
# ---------------------------------------------------------------------------
@unittest.skipIf(torch is None, 'needs torch')
class RolloutLoopTests(unittest.TestCase):
    def run_loop(self, is_train):
        gen = Proto.from_single_dict(question_batch([3, 4, 5, 8]))
        return collector().multi_turn_loop(gen_batch=gen, actor_rollout_wg=None, envs=None, is_train=is_train)

    def test_validation_carries_the_per_episode_values(self):
        out = self.run_loop(is_train=False)
        episodes = out.meta_info['success_episodes']
        self.assertEqual(set(episodes), {'success_rate', 'a_success_rate', 'b_success_rate', 'c_success_rate'})
        np.testing.assert_array_equal(episodes['success_rate'], [0., 0., 1., 1.])
        np.testing.assert_array_equal(episodes['a_success_rate'], [0., 0.])
        np.testing.assert_array_equal(episodes['c_success_rate'], [1.])
        self.assertEqual(episodes['success_rate'].dtype, np.float64)
        # The rows still carry the batch means, and only the turns that were played.
        self.assertEqual(len(out.non_tensor_batch['traj_uid']), TURNS[3] + TURNS[4] + TURNS[5] + TURNS[8])
        np.testing.assert_array_equal(out.non_tensor_batch['success_rate'], .5)
        np.testing.assert_array_equal(out.non_tensor_batch['a_success_rate'], 0.)

    def test_training_rollouts_are_left_as_they_were(self):
        self.assertEqual(self.run_loop(is_train=True).meta_info, {})

    def test_existing_and_absent_meta_info_are_both_handled(self):
        gather = Proto.from_single_dict
        for before, kept in (({'eos_token_id': 2}, {'eos_token_id': 2}), (None, {})):
            def from_single_dict(data, meta_info=None, before=before):
                out = gather(data)
                out.meta_info = before
                return out
            with patch.object(Proto, 'from_single_dict', staticmethod(from_single_dict)):
                out = collector().multi_turn_loop(gen_batch=gather(question_batch([0, 8])), actor_rollout_wg=None,
                                                  envs=None, is_train=False)
            self.assertEqual({k: v for k, v in out.meta_info.items() if k != 'success_episodes'}, kept)
            np.testing.assert_array_equal(out.meta_info['success_episodes']['success_rate'], [1., 1.])
            if before is not None:                      # the caller's dict is not written to
                self.assertNotIn('success_episodes', before)


# ---------------------------------------------------------------------------
# 3. validation
# ---------------------------------------------------------------------------
@unittest.skipIf(torch is None, 'needs torch')
class ValidationTests(unittest.TestCase):
    def test_pooled_rates_are_over_questions_and_published_keys_are_unchanged(self):
        metrics = validate()
        for source, key in ((None, 'success_rate'), ('a', 'a_success_rate'), ('b', 'b_success_rate'),
                            ('c', 'c_success_rate')):
            rate, questions = rate_over_questions(source)
            self.assertAlmostEqual(metrics[f'val/{key}/pooled'], rate, msg=key)
            self.assertEqual(metrics[f'val/{key}/episodes'], questions, msg=key)
            self.assertAlmostEqual(metrics[f'val/{key}'], batch_mean_of_means(source), msg=key)
        # The scenario separates the two definitions: 6 of 9 questions against a mean of
        # batch means of 0.75, and 3 of 5 for `a` against (3/4 + 0/1) / 2.
        self.assertAlmostEqual(metrics['val/success_rate/pooled'], 6 / 9)
        self.assertAlmostEqual(metrics['val/success_rate'], .75)
        self.assertAlmostEqual(metrics['val/a_success_rate/pooled'], .6)
        self.assertAlmostEqual(metrics['val/a_success_rate'], .375)

    def test_best_checkpoint_score_is_still_the_published_overall_rate(self):
        metrics = validate()
        # fit() takes the first val key ending in success_rate as the score.
        score = next(v for k, v in metrics.items() if k.startswith('val') and k.endswith('success_rate'))
        self.assertEqual(score, metrics['val/success_rate'])
        added = [k for k in metrics if k.endswith(('/pooled', '/episodes'))]
        self.assertEqual(len(added), 8)
        self.assertFalse([k for k in added if k.endswith('success_rate')])

    def test_other_validation_metrics_are_unchanged(self):
        with_pooled, without = validate(), validate(attach=False)
        self.assertEqual({k: v for k, v in with_pooled.items() if not k.endswith(('/pooled', '/episodes'))}, without)
        self.assertFalse([k for k in without if k.endswith(('/pooled', '/episodes'))])
        self.assertAlmostEqual(without['val/length/mean'], np.mean(TURNS))


# ---------------------------------------------------------------------------
# 4. the report
# ---------------------------------------------------------------------------
def metrics_row(step, rates, questions):
    """A validation row as the trainer's JSONL logger writes it."""
    row = {'step': step, 'wall_time': 0., 'val/success_rate': .1}
    total = sum(questions.values())
    for name, rate in rates.items():
        row[f'val/{name}_success_rate'] = .9                       # the batch-mean key, never read
        row[f'val/{name}_success_rate/pooled'] = rate
        row[f'val/{name}_success_rate/episodes'] = float(questions[name])
    row['val/success_rate/pooled'] = sum(rates[n] * questions[n] for n in rates) / total
    row['val/success_rate/episodes'] = float(total)
    return row


def write_run(directory, name, rows):
    run = Path(directory)/name
    (run/'outputs').mkdir(parents=True)
    (run/'outputs/metrics.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return run


QUESTIONS = {'bamboogle': 125, 'nq': 3610, 'hotpotqa': 7405, 'extra': 10}
RATES = {'bamboogle': .4, 'nq': .5, 'hotpotqa': .3, 'extra': 1.}


class ReportTests(unittest.TestCase):
    def test_average_is_unweighted_and_overall_is_over_questions(self):
        with tempfile.TemporaryDirectory() as d:
            run = write_run(d, 'search-a', [{'step': 1, 'episode/success_rate': .2},
                                            metrics_row(50, {k: 0. for k in RATES}, QUESTIONS),
                                            metrics_row(200, RATES, QUESTIONS)])
            (name, step, summary), = report.collect([run])
        self.assertEqual((name, step), ('search-a', 200))
        # Published column order for the known datasets, then any other data source.
        self.assertEqual(summary['order'], ['nq', 'hotpotqa', 'bamboogle', 'extra'])
        self.assertAlmostEqual(summary['avg'], (.4 + .5 + .3 + 1.) / 4)
        self.assertAlmostEqual(summary['all'], (.4 * 125 + .5 * 3610 + .3 * 7405 + 10) / 11150)
        self.assertEqual(summary['all_questions'], 11150)
        self.assertEqual(summary['questions'], QUESTIONS)

    def test_table_for_several_runs_at_one_step(self):
        with tempfile.TemporaryDirectory() as d:
            a = write_run(d, 'search-a', [metrics_row(200, RATES, QUESTIONS)])
            b = write_run(d, 'search-b', [metrics_row(200, {**RATES, 'nq': .25}, QUESTIONS),
                                          metrics_row(250, RATES, QUESTIONS)])
            out = io.StringIO()
            with redirect_stdout(out):
                report.main([str(a), str(b), '--step', '200', '--markdown'])
        lines = out.getvalue().splitlines()
        self.assertEqual(lines[0], '| run | NQ | HotpotQA | Bamboogle | extra | Avg | All |')
        # All: (0.4 * 125 + 0.5 * 3610 + 0.3 * 7405 + 10) / 11150 = 0.36650
        self.assertEqual(lines[2], '| search-a @200 | 50.0 | 30.0 | 40.0 | 100.0 | 55.0 | 36.7 |')
        self.assertTrue(lines[3].startswith('| search-b @200 | 25.0 | 30.0 | 40.0 | 100.0 | 48.8 |'))
        self.assertEqual(lines[4], '| questions | 3610 | 7405 | 125 | 10 |  | 11150 |')
        self.assertEqual(len(lines), 5)

    def test_runs_validated_on_different_questions_are_refused(self):
        with tempfile.TemporaryDirectory() as d:
            a = write_run(d, 'search-a', [metrics_row(200, RATES, QUESTIONS)])
            b = write_run(d, 'search-b', [metrics_row(200, RATES, {**QUESTIONS, 'nq': 500})])
            with self.assertRaisesRegex(ValueError, 'different questions'):
                report.collect([a, b])
            with self.assertRaises(SystemExit) as stopped:
                report.main([str(a), str(b)])
            self.assertIn('different questions', str(stopped.exception.code))

    def test_missing_pooled_rates_and_missing_steps_are_reported(self):
        with tempfile.TemporaryDirectory() as d:
            old = write_run(d, 'search-old', [{'step': 50, 'val/success_rate': .4, 'val/nq_success_rate': .4}])
            new = write_run(d, 'search-new', [metrics_row(50, RATES, QUESTIONS)])
            with self.assertRaisesRegex(ValueError, 'no pooled validation rates'):
                report.collect([old])
            with self.assertRaisesRegex(ValueError, r'no evaluation at step 200; it has \[50\]'):
                report.collect([new], step=200)
            with self.assertRaises(FileNotFoundError):
                report.collect([Path(d)/'absent'])


if __name__ == '__main__':
    unittest.main()
