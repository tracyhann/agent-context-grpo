"""Census integrity and native-reasoning action isolation for full benchmark evals."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import benchmark_census as census
from qwen_census_eval import split_native_thinking
from report_census_eval import report

class CensusTests(unittest.TestCase):
    def test_complete_disjoint_alfworld_splits(self):
        seen,_=census.plan('alfworld',101,'seen');unseen,_=census.plan('alfworld',101,'unseen')
        a={x['gamefile'] for x in seen};b={x['gamefile'] for x in unseen}
        self.assertEqual((len(a),len(b)),(140,134));self.assertFalse(a&b)
        self.assertEqual(seen,census.plan('alfworld',101,'seen')[0])
        self.assertEqual({x['gamefile'] for x in census.plan('alfworld',102,'seen')[0]},a)
        webshop,_=census.plan('webshop',101)
        self.assertEqual({x['goal_index'] for x in webshop},set(range(500)))
    def test_only_final_answer_can_supply_action(self):
        action='<action>buy now</action>'
        self.assertEqual(split_native_thinking('Maybe '+action,True),('', 'Maybe '+action))
        content,reasoning=split_native_thinking('Reject '+action+'</think>\n<action>look</action>',True)
        self.assertEqual(content,'<action>look</action>');self.assertIn(action,reasoning)
        self.assertEqual(split_native_thinking(action,False),(action,''))
    def test_report_cannot_hide_error_or_duplicate_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'outputs/webshop';(out/'episodes').mkdir(parents=True)
            (root/'suites.json').write_text(json.dumps(dict(provider='local-qwen',title='test',suites=[dict(name='webshop',benchmark='webshop',count=2)])))
            plan=[dict(episode_id=0,goal_index=0),dict(episode_id=1,goal_index=1)]
            (out/'config.json').write_text(json.dumps(dict(episodes=plan,settings={})))
            win=dict(**plan[0],status='completed',success=True,score=1.,length=0,turns=[])
            error=dict(**plan[1],status='error',turns=[])
            (out/'episodes/0000.json').write_text(json.dumps(win));(out/'episodes/0001.json').write_text(json.dumps(error))
            r=report(root);self.assertFalse(r['suites']['webshop']['final']);self.assertEqual(r['suites']['webshop']['errors'],1)
            self.assertIn('pending',(root/'RESULTS.md').read_text())
            loss=dict(**plan[1],status='completed',success=False,score=.4,length=0,turns=[])
            (out/'episodes/0001.json').write_text(json.dumps(loss));r=report(root)
            self.assertEqual(r['status'],'completed');self.assertEqual(r['suites']['webshop']['success_pct'],50.)
            self.assertEqual(r['suites']['webshop']['score_pct'],70.)
            loss['goal_index']=0;(out/'episodes/0001.json').write_text(json.dumps(loss))
            with self.assertRaisesRegex(ValueError,'task identity'):report(root)

if __name__=='__main__':unittest.main()
