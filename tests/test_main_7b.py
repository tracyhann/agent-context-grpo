"""Prevent 7B preparations from launching 1.5B weights or changing the paired method."""
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'ablations'))
sys.path.insert(0,str(ROOT/'scripts'))
import ablations as registry
import prepare_main_7b as prep
spec=importlib.util.spec_from_file_location('seven_b_ablation_cli',ROOT/'ablations/run.py')
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


class MainSevenBTests(unittest.TestCase):
    def test_backbone_changes_only_model_and_preserves_legacy_override_api(self):
        for benchmark in ['alfworld','webshop']:
            for key in [prep.MAIN,prep.ACTIVE]:
                old_name,old=registry.build(key,benchmark)
                name,new=registry.build(key,benchmark,backbone='7b')
                self.assertTrue(old_name.endswith('-1.5b'));self.assertEqual(name,old_name[:-4]+'7b')
                self.assertEqual({k:v for k,v in new.items() if old[k]!=v},{'model':'Qwen/Qwen2.5-7B-Instruct'})
                self.assertTrue(registry.control_name(benchmark,key,backbone='7b').endswith('-7b'))
                self.assertEqual(registry.build(key,benchmark,{'seed':17})[1]['seed'],17)
        with self.assertRaises(KeyError):registry.build(prep.MAIN,'alfworld',backbone='unknown')

    def test_cli_passes_backbone_to_builder_gpu_check_and_launch(self):
        for benchmark in ['alfworld','webshop']:
            for key in [prep.MAIN,prep.ACTIVE]:
                args=['run.py','--ablation',key,'--benchmark',benchmark,'--backbone','7b','--gpus','0,1,2,3,4,5,6,7','--dry-run']
                with patch.object(sys,'argv',args),patch.object(cli.arms,'launch',return_value=0) as launch,patch.object(cli.arms,'check_gpus') as gpu,redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(),0)
                gpu.assert_called_once_with('0,1,2,3,4,5,6,7','7b')
                name,cfg,bench,gpus,extra=launch.call_args.args
                self.assertTrue(name.endswith('-7b'));self.assertEqual(cfg['model'],'Qwen/Qwen2.5-7B-Instruct')
                self.assertEqual(cfg['ccpo_lk_fix'],1.);self.assertEqual(cfg['ccpo_ep_w'],float(key==prep.ACTIVE))
                self.assertEqual(extra,['--dry-run'])
        with patch.object(sys,'argv',['run.py','--list','--backbone','7b']),redirect_stdout(io.StringIO()) as out:
            self.assertEqual(cli.main(),0)
        self.assertIn('ccpo-attncred-abl-fph2-noshrink-ws-7b',out.getvalue())

    def test_prepared_configs_preserve_science_and_isolate_episode_coefficient(self):
        for benchmark,episode in prep.CONTROLS:
            folder=ROOT/'experiments'/prep.run_name(benchmark,episode,'20260920')
            record=json.loads((folder/'config.json').read_text());cfg=record['config']
            old=json.loads((ROOT/'experiments'/prep.CONTROLS[benchmark,episode]/'config.json').read_text())['config']
            self.assertEqual(set(cfg),set(old))
            self.assertEqual({k for k in cfg if cfg[k]!=old[k]},prep.SCALE_KEYS)
            self.assertEqual(cfg['model'],'Qwen/Qwen2.5-7B-Instruct');self.assertIn('Qwen2.5-7B',cfg['model_path'])
            paired=json.loads((ROOT/'experiments'/prep.run_name(benchmark,1-episode,'20260920')/'config.json').read_text())['config']
            self.assertEqual({k for k in cfg if cfg[k]!=paired[k]},{'exp_id','ccpo_ep_w'})
            for k,v in dict(ccpo_ep_w=float(episode),ccpo_lk_fix=1.,ccpo_lam_fix=1.,ccpo_edge_w=0.,ccpo_progress_horizon=2,
                ccpo_ctx_w=1.,ccpo_progress_weight=1.,ccpo_target='return',ccpo_wmode='soft',history_length=2,
                total_epochs=150,seed=0,group_size=8,train_batch_size=16,tp_size=2,resume_from='').items():self.assertEqual(cfg[k],v,k)
            self.assertEqual(cfg['max_steps'],50 if benchmark=='alfworld' else 15)
            self.assertEqual(cfg['adv_mode'],'mean_std_norm' if benchmark=='alfworld' else 'mean_norm')

    def test_rendered_commands_match_configs_and_only_prepare(self):
        ray_paths=[]
        for benchmark,episode in prep.CONTROLS:
            folder=ROOT/'experiments'/prep.run_name(benchmark,episode,'20260920')
            record=json.loads((folder/'config.json').read_text());cfg=record['config']
            self.assertEqual(record['hydra_overrides'],prep.er.build_command(cfg,str(folder))[3:])
            self.assertIn('trainer.n_gpus_per_node=8',record['hydra_overrides'])
            self.assertIn('actor_rollout_ref.rollout.tensor_model_parallel_size=2',record['hydra_overrides'])
            self.assertEqual(record['env']['CUDA_VISIBLE_DEVICES'],'0,1,2,3,4,5,6,7')
            for key,env in prep.er.ENV_KEYS.items():
                if key in ('ccpo_progress_history_weight', 'ccpo_loo') and key not in cfg:
                    self.assertEqual(prep.er.DEFAULTS[key], 1.0); self.assertNotIn(env, record['env'])
                else:
                    self.assertEqual(record['env'][env],str(cfg[key]),env)
            ray_paths.append(record['env']['RAY_TMPDIR'])
            status=json.loads((folder/'PREPARED.json').read_text());self.assertFalse(status['launched']);self.assertFalse(status['queued'])
            command=json.loads((folder/'prepare-command.json').read_text());self.assertIn('--dry-run',command['argv'])
            self.assertEqual(command['gpu_visibility_for_preparation'],'')
            self.assertFalse((folder/'outputs/train.pid').exists());self.assertFalse((folder/'outputs/metrics.jsonl').exists())
        self.assertEqual(len(set(ray_paths)),4)

    def test_preparation_never_overwrites_an_existing_run(self):
        with patch.object(prep.subprocess,'run') as run:
            with self.assertRaises(FileExistsError):prep.prepare('alfworld',0,'20260920','0,1,2,3,4,5,6,7')
        run.assert_not_called()


if __name__=='__main__':unittest.main()
