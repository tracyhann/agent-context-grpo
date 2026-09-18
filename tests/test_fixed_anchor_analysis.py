"""CPU checks for the offline fixed-anchor measurement, not experiment results."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyse_fixed_anchor import evaluate, summarize, core
import numpy as np
import unittest


def fixture():
    n=24
    task=np.array(['task']*n)
    traj=np.repeat([f'traj{i}' for i in range(6)],4)
    obs=np.tile(['shared','room','shared','last'],6)
    turn=np.tile(np.arange(4),6)
    reward=np.zeros(n);reward[[3,11,19]]=10
    outcome=np.repeat([10,0,10,0,10,0],4)
    returns=np.zeros(n)
    for j in range(6):
        v=0
        for i in range(j*4+3,j*4-1,-1):
            v=reward[i]+.95*v;returns[i]=v
    valid=np.ones(n,bool);valid[9]=False;returns[9]-=.1
    history_hidden=np.random.default_rng(123).normal(size=(n,8)).astype(np.float32)
    future_hidden=np.random.default_rng(456).normal(size=(n,8)).astype(np.float32)
    ctx=core.derive_context(obs,task,traj)
    future_ctx=np.array([[ctx[min((i//4)*4+3,i+2)][k] for k in ('t','n_unique','revisit','progress')] for i in range(n)])
    return dict(uid=task,traj_uid=traj,anchor_obs=obs,turn_index=turn,episode_lengths=np.full(n,4),
        immediate_rewards=reward,episode_rewards=outcome,returns=returns,is_action_valid=valid,
        history_hidden=history_hidden,future_hidden=future_hidden,future_context=future_ctx)


class FixedAnchorTests(unittest.TestCase):
    def test_identical_context_is_zero_signal(self):
        b=fixture();b['future_hidden']=b['history_hidden'].copy()
        ctx=core.derive_context(b['anchor_obs'],b['uid'],b['traj_uid'])
        b['future_context']=np.array([[r[k] for k in ('t','n_unique','revisit','progress')] for r in ctx])
        a=evaluate(b)
        np.testing.assert_allclose(a['fixed_future'],0,atol=1e-10)

    def test_future_features_change_weights_without_changing_anchor_support(self):
        b=fixture();a=evaluate(b)
        self.assertTrue(a['eligible'].all())
        self.assertTrue((a['support']==5).all())
        np.testing.assert_allclose(a['shrinkage'],5/7)
        self.assertGreater(np.max(abs(a['fixed_future'])),.01)
        np.testing.assert_allclose(a['history'],a['fixed_future']+a['future_residual'])

    def test_own_trajectory_return_excluded_from_both_readouts(self):
        b=fixture();first=evaluate(b)
        own=b['traj_uid']=='traj0'
        for k in ('returns','immediate_rewards','episode_rewards'):
            b[k]=b[k].copy();b[k][own]=0
        other=evaluate(b)
        for k in ('baseline_history','baseline_future','fixed_future'):
            np.testing.assert_allclose(first[k][own],other[k][own],atol=1e-10)

    def test_shuffle_and_duplicate_padding(self):
        b=fixture();first=evaluate(b)
        ix=np.random.default_rng(1).permutation(np.r_[np.arange(24),[0,5,14]])
        other=evaluate({k:v[ix] for k,v in b.items()})
        for k in ('fixed_future','history','edge','old_outlook'):
            np.testing.assert_allclose(first[k],other[k],atol=1e-7)

    def test_no_exact_peer_does_not_regroup_future(self):
        b=fixture();b['anchor_obs'][0]='lonely'
        a=evaluate(b)
        self.assertFalse(a['eligible'][0])
        self.assertTrue(np.isnan(a['fixed_future'][0]))
        self.assertEqual(a['added_signal'][0],0)
        self.assertTrue(np.isfinite(a['old_outlook'][0]))

    def test_missing_features_rejected(self):
        b=fixture();del b['future_hidden']
        with self.assertRaisesRegex(ValueError,'future_hidden'):
            evaluate(b)

    def test_penalty_must_be_applied_once(self):
        b=fixture();b['returns'][0]-=.1
        with self.assertRaisesRegex(ValueError,'exactly once'):
            evaluate(b)

    def test_json_summary(self):
        import json
        result=summarize(evaluate(fixture()))
        json.dumps(result,allow_nan=False)


if __name__=='__main__': unittest.main(verbosity=2)
