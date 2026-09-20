"""Regression checks for sampling, score accounting, HTTP errors and inference bounds."""
import io
import json
import unittest
from unittest.mock import Mock, patch
import urllib.error

import numpy as np
from qwen_baseline import common, evaluate, serve


class ProtocolTests(unittest.TestCase):
    def test_webshop_sampling_matches_reference_rng_batches(self):
        rng = np.random.RandomState(1000)
        expected = [int(x) for _ in range(2) for x in rng.choice(range(500), 128, replace=False)]
        plan = common.episode_plan('webshop')
        self.assertEqual([x['goal_index'] for x in plan], expected)
        self.assertEqual(plan[128]['worker_seed'], 1000)
        self.assertEqual(len(set(expected[:128])), 128)
        later = common.episode_plan('webshop', eval_round=1)
        expected_later = [int(x) for _ in range(2) for x in rng.choice(range(500), 128, replace=False)]
        self.assertEqual([x['goal_index'] for x in later], expected_later)

    def test_smoke_subset_preserves_full_eval_identities(self):
        for benchmark in common.PROFILES:
            self.assertEqual(common.episode_plan(benchmark, 3), common.episode_plan(benchmark)[:3])
        plan = common.episode_plan('alfworld', eval_round=2)
        self.assertEqual([plan[i]['reset_index'] for i in [0, 32, 64, 96]], [8, 9, 10, 11])
        self.assertEqual([plan[i]['worker_seed'] for i in [0, 32, 64, 96]], [1000]*4)

    def test_invalid_episode_count_is_rejected(self):
        for count in [0, -1, 129]:
            with self.assertRaises(ValueError):
                common.episode_plan('alfworld', count)

    def test_errors_never_produce_an_optimistic_final_score(self):
        win = dict(status='completed', success=True, score=1., length=1, turns=[])
        error = dict(status='error', error='HTTP failure', turns=[])
        partial = common.aggregate([win, error], 2)
        self.assertEqual(partial['status'], 'incomplete')
        self.assertIsNone(partial['success_rate'])
        self.assertEqual(partial['observed_success_rate'], 1.)
        loss = dict(status='completed', success=False, score=.4, length=2, turns=[])
        full = common.aggregate([win, loss], 2)
        self.assertEqual(full['success_rate'], .5)
        self.assertAlmostEqual(full['score'], .7)

    def test_completion_sends_native_token_ids_and_explicit_sampling(self):
        settings = dict(base_url='http://localhost:8018/v1', max_tokens=512, temperature=.4, timeout=30)
        response = dict(choices=[dict(text='<action>look</action>', finish_reason='stop')], usage={})
        with patch.object(common, 'post_json', return_value=response) as post:
            result = common.complete([1, 2, 3], settings, 27)
        payload = post.call_args.args[2]
        self.assertEqual(payload['prompt'], [1, 2, 3])
        self.assertEqual(payload['model'], common.SERVED_MODEL)
        self.assertEqual(payload['seed'], 27)
        self.assertEqual((payload['max_tokens'], payload['temperature']), (512, .4))
        self.assertEqual(result['text'], response['choices'][0]['text'])

    def test_bad_request_is_not_retried(self):
        error = urllib.error.HTTPError('http://localhost', 400, 'bad request', {}, io.BytesIO(b'context overflow'))
        with patch.object(common.urllib.request, 'urlopen', side_effect=error) as urlopen:
            with self.assertRaisesRegex(RuntimeError, 'context overflow'):
                common.post_json('http://localhost/v1', 'completions', {})
            self.assertEqual(urlopen.call_count, 1)

    def test_transient_server_error_retries_the_identical_request(self):
        error = urllib.error.HTTPError('http://localhost', 503, 'unavailable', {}, io.BytesIO(b'busy'))
        response = io.BytesIO(json.dumps({'choices': []}).encode())
        with patch.object(common.urllib.request, 'urlopen', side_effect=[error, response]) as urlopen, patch.object(common.time, 'sleep'):
            self.assertEqual(common.post_json('http://localhost/v1', 'completions', {'seed': 4}), {'choices': []})
            self.assertEqual(urlopen.call_args_list[0].args[0].data, urlopen.call_args_list[1].args[0].data)

    def test_malformed_completion_fails(self):
        with patch.object(common, 'post_json', return_value={'choices': []}):
            with self.assertRaises(ValueError):
                common.complete([1], dict(base_url='x', max_tokens=1, temperature=0, timeout=10), 0)

    def test_prompt_overflow_is_an_error_before_inference(self):
        manager, raw, tok = Mock(), Mock(), Mock()
        manager.reset.return_value = ({'text': ['prompt'], 'anchor': ['obs']}, [{}])
        tok.apply_chat_template.return_value = list(range(11))
        settings = dict(benchmark='alfworld', split='eval_in_distribution', history_length=2,
                        max_steps=1, max_prompt_tokens=10, thinking=False, reasoning_effort='low', generation_seed=0)
        with patch.object(evaluate, 'runtime_setup'), patch.object(evaluate, 'build_environment', return_value=(manager, raw)), patch.object(evaluate, 'tokenizer', return_value=tok), patch.object(evaluate, 'complete') as complete:
            record = evaluate.run_episode(dict(episode_id=0), settings)
        self.assertEqual(record['status'], 'error')
        self.assertIn('Prompt has 11 tokens', record['error'])
        complete.assert_not_called()
        raw.close.assert_called_once()
        self.assertFalse(tok.apply_chat_template.call_args.kwargs['enable_thinking'])

    def test_busy_gpu_blocks_launcher_before_model_loading(self):
        with patch.object(serve.subprocess, 'check_output', return_value='0, 81920, 29000, 80\n1, 81920, 0, 0\n'):
            with self.assertRaisesRegex(RuntimeError, 'GPU 0 is busy'):
                serve.gpu_preflight([0, 1])
            self.assertEqual(serve.gpu_preflight([1]), {'1': [81920, 0, 0]})


if __name__ == '__main__':
    unittest.main()
