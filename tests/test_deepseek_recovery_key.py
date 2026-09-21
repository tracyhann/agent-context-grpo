"""Credential rotation must preserve cached responses and the original run config."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import recover_deepseek_eval as recover


class RecoveryKeyTests(unittest.TestCase):
    def fixture(self,root):
        (root/'episodes').mkdir();(root/'api_calls').mkdir()
        turns=[];cached=[]
        for i in range(2):
            prompt=f'prompt-{i}'
            turn=dict(prompt=prompt,observation=f'obs-{i}',next_observation=f'obs-{i+1}',
                      action=f'action-{i}',reward=0.,score=0.,done=False,request_id=f'saved-{i}')
            turns.append(turn);cached.append(dict(turn,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
        old=dict(episode_id=100,status='error',benchmark='alfworld',turns=turns,error='RuntimeError: API HTTP 402; response body omitted')
        episode=root/'episodes/0100.json';episode.write_text(json.dumps(old))
        (root/'api_calls/0100.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in cached))
        (root/'config.json').write_text(json.dumps(dict(settings={'key_file':str(root/'old-key.txt')},episodes=[{'episode_id':100}])))
        return episode,old

    def test_new_key_applies_only_to_new_requests_without_mutating_config(self):
        for override in [False,True]:
            with self.subTest(override=override),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);episode,old=self.fixture(root);before=(root/'config.json').read_bytes();paid=[]
                def complete(prompt,settings,path):
                    paid.append((prompt,settings['key_file']))
                    return {'request_id':'new'}
                def run_episode(item,settings):
                    got=[]
                    for i in range(3):got.append(recover.evaluator.complete(f'prompt-{i}',settings,root/'api_calls/0100.jsonl'))
                    self.assertEqual([r['request_id'] for r in got],['saved-0','saved-1','new'])
                    return dict(old,status='completed',turns=old['turns']+[{'request_id':'new'}])
                key=root/'new-key.txt' if override else None
                with patch.object(recover.evaluator,'complete',side_effect=complete),patch.object(recover.evaluator,'run_episode',side_effect=run_episode):
                    result=recover.recover_one(episode,key_file=key)
                expected=str(key.resolve()) if key else str(root/'old-key.txt')
                self.assertEqual(paid,[('prompt-2',expected)])
                self.assertEqual(result['credential_source_path'],expected)
                self.assertEqual(result['replayed_requests'],2);self.assertEqual(result['new_requests'],1)
                self.assertEqual((root/'config.json').read_bytes(),before)
                self.assertEqual(len(list((root/'recovery').glob('*/0100.json'))),1)

    def test_mismatched_replay_fails_before_new_paid_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);episode,_=self.fixture(root)
            def run_episode(item,settings):
                return recover.evaluator.complete('different prompt',settings,root/'api_calls/0100.jsonl')
            with patch.object(recover.evaluator,'complete') as paid,patch.object(recover.evaluator,'run_episode',side_effect=run_episode):
                with self.assertRaisesRegex(AssertionError,'Replayed prompt differs'):recover.recover_one(episode,root/'new-key.txt')
                paid.assert_not_called()

    def test_completed_episode_is_never_replayed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);episode,old=self.fixture(root);old['status']='completed';episode.write_text(json.dumps(old))
            with patch.object(recover.evaluator,'run_episode') as run:
                result=recover.recover_one(episode,root/'new-key.txt')
                self.assertEqual(result['status'],'skipped');run.assert_not_called()


if __name__=='__main__':unittest.main()
