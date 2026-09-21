"""Protect accelerated census ownership and the original controller's join path."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import qwen_eval_coordination as c


class CoordinationTests(unittest.TestCase):
    def fixture(self, root):
        args=argparse.Namespace(output=root,suite='webshop',seed=101,max_tokens=65536,
              max_model_len=131072,no_thinking=False,webshop_goal_manifest=None,
              base_url='http://127.0.0.1:8019/v1',workers=16)
        marker=dict(protocol=c.protocol(args),output=str(root),base_url=args.base_url,
                    workers=16,expected_count=2,status='prepared')
        c.atomic_json(root/c.MARKER,marker)
        return args

    def completed(self,args):
        episodes=[dict(episode_id=0,goal_index=0),dict(episode_id=1,goal_index=1)]
        c.atomic_json(args.output/'config.json',dict(model=c.MODEL,episodes=episodes,
            settings={k:c.protocol(args)[k] for k in ['seed','max_tokens','max_model_len','thinking']}))
        for e in episodes:c.atomic_json(args.output/'episodes'/f"{e['episode_id']:04d}.json",dict(e,status='completed'))
        metrics=dict(status='completed',requested=2,completed=2)
        c.atomic_json(args.output/'metrics.json',metrics);c.finish_owner(args,metrics)

    def test_regular_output_keeps_default_behavior(self):
        with tempfile.TemporaryDirectory() as t:
            a=argparse.Namespace(output=Path(t))
            self.assertIsNone(c.join_if_managed(a))

    def test_exclusive_owner_and_finished_join_with_original_endpoint(self):
        with tempfile.TemporaryDirectory() as t:
            a=self.fixture(Path(t));lock=c.claim_owner(a)
            try:
                with self.assertRaises(BlockingIOError):c.claim_owner(a)
                self.completed(a)
            finally:lock.close()
            a.base_url='http://127.0.0.1:8018/v1';a.workers=8
            self.assertEqual(c.join_if_managed(a),0)
            a.base_url='http://127.0.0.1:8019/v1';a.workers=16
            with self.assertRaisesRegex(ValueError,'overwrite'):c.claim_owner(a)

    def test_join_rejects_protocol_drift(self):
        with tempfile.TemporaryDirectory() as t:
            a=self.fixture(Path(t));a.seed=102
            with self.assertRaisesRegex(ValueError,'protocol differs'):c.join_if_managed(a)

    def test_join_rejects_dead_or_missing_owner(self):
        with tempfile.TemporaryDirectory() as t:
            a=self.fixture(Path(t))
            with self.assertRaisesRegex(RuntimeError,'never started'):c.join_if_managed(a)
            lock=c.claim_owner(a);lock.close()
            with self.assertRaisesRegex(RuntimeError,'exited before completing'):c.join_if_managed(a)

    def test_join_rejects_missing_or_wrong_task(self):
        with tempfile.TemporaryDirectory() as t:
            a=self.fixture(Path(t));lock=c.claim_owner(a);self.completed(a);lock.close()
            file=a.output/'episodes/0001.json';file.unlink()
            with self.assertRaisesRegex(ValueError,'missing episodes'):c.join_if_managed(a)
            c.atomic_json(file,dict(episode_id=1,goal_index=0,status='completed'))
            with self.assertRaisesRegex(ValueError,'identity differs'):c.join_if_managed(a)

    def test_incomplete_census_never_claims_success(self):
        with tempfile.TemporaryDirectory() as t:
            a=self.fixture(Path(t));lock=c.claim_owner(a)
            metrics=dict(status='incomplete',requested=2,completed=1)
            c.atomic_json(a.output/'metrics.json',metrics);c.finish_owner(a,metrics);lock.close()
            self.assertEqual(c.join_if_managed(a),1)


    def test_acceleration_command_preserves_protocol(self):
        from run_qwen_parallel_webshop import accelerated_command
        original=['python','qwen_census_eval.py','--suite','webshop','--seed','101',
                  '--workers','8','--max-tokens','65536','--max-model-len','131072',
                  '--output','/temporary/webshop','--webshop-goal-manifest','/temporary/goals.json']
        actual=accelerated_command(original,'http://127.0.0.1:8019/v1',16)
        expected=original.copy();expected[expected.index('--workers')+1]='16'
        self.assertEqual(actual,expected+['--base-url','http://127.0.0.1:8019/v1','--managed-owner'])
        self.assertEqual(original[original.index('--workers')+1],'8')

    def test_cleanup_never_signals_reused_or_unknown_process(self):
        import run_qwen_parallel_webshop as supervisor
        with patch.object(supervisor,'identity',return_value='new'),patch.object(supervisor.os,'killpg') as kill:
            supervisor.stop_owned(1234,'old');kill.assert_not_called()
            supervisor.stop_owned(1234,None);kill.assert_not_called()


if __name__=='__main__':unittest.main()
