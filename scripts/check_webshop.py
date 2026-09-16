#!/usr/bin/env python3
"""CPU checks and generation of missing WebShop index/parquet assets.

Normally invoked by setup_webshop.sh so Java, Python paths and thread limits are set.
"""
import argparse
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

from setup_webshop import ROOT, WEBSHOP, atomic_json, sha256, verify_source


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def prepare_inputs(assets):
    import pyarrow as pa
    import pyarrow.parquet as pq
    destination = ROOT / 'envdata/webshop_data/text'
    destination.mkdir(parents=True, exist_ok=True)
    for split, rows in [('train', assets['protocol']['train_rows']), ('test', assets['protocol']['test_rows'])]:
        path = destination / (split + '.parquet')
        if path.exists():
            continue
        # These are modality/length placeholders, not external question-answer data.
        table = pa.Table.from_pylist([{
            'answer': '', 'data_source': 'text', 'prompt': [{'role': 'user', 'content': ''}],
            'ability': 'agent', 'extra_info': {'split': split, 'index': i},
        } for i in range(rows)])
        temporary = path.with_suffix('.parquet.tmp')
        pq.write_table(table, temporary)
        temporary.replace(path)
    engine = ROOT / 'envdata/webshop_data/search_engine'
    index = engine / 'indexes'
    if index.exists():
        return
    engine.mkdir(parents=True, exist_ok=True)
    from web_agent_site.engine.engine import load_products
    data = ROOT / 'envdata/webshop_data/data'
    random.seed(0)
    products, *_ = load_products(filepath=str(data / 'items_shuffle_1000.json'),
                                attrpath=str(data / 'items_ins_v2_1000.json'))
    require(len(products) == assets['protocol']['products'], 'Wrong catalogue size during index generation')
    with tempfile.TemporaryDirectory(prefix='.build-', dir=engine) as stage:
        stage = Path(stage)
        documents = stage / 'resources_1k'
        documents.mkdir()
        with (documents / 'documents.jsonl').open('w') as out:
            for product in products:
                options = ', and '.join(f"{name}: {', '.join(contents)}"
                                        for name, contents in product.get('options', {}).items())
                contents = ' '.join([product['Title'], product['Description'],
                                     product['BulletPoints'][0], options]).lower()
                out.write(json.dumps({'id': product['asin'], 'contents': contents,
                                      'product': product}) + '\n')
        subprocess.run([sys.executable, '-m', 'pyserini.index.lucene', '--collection', 'JsonCollection',
                        '--input', str(documents), '--index', str(stage / 'indexes'), '--generator',
                        'DefaultLuceneDocumentGenerator', '--threads', '1', '--storePositions',
                        '--storeDocvectors', '--storeRaw'], check=True)
        # Publish only a successfully built index. Existing indexes are never rebuilt.
        from pyserini.search.lucene import LuceneSearcher
        searcher = LuceneSearcher(str(stage / 'indexes'))
        require(searcher.num_docs == len(products), 'New Lucene index has missing documents')
        searcher.close()
        (stage / 'indexes').rename(index)
        if not (engine / 'resources_1k').exists():
            documents.rename(engine / 'resources_1k')


def validate(assets, skip_model):
    import torch
    import vllm
    import transformers
    import ray
    import flash_attn
    import flash_attn_2_cuda
    import spacy
    import verl
    from verl import DataProto
    import verl.trainer.ppo.ray_trainer
    import verl.workers.actor.dp_actor
    import pyarrow.parquet as pq
    from pyserini.search.lucene import LuceneSearcher
    from agent_system.environments.env_package.webshop.envs import WebshopWorker

    verify_source()
    require(sys.version_info[:2] == (3, 10), 'Expected Python 3.10')
    versions = {name: metadata.version(name) for name in
                ('torch', 'vllm', 'transformers', 'ray', 'tensordict', 'numpy', 'pyserini',
                 'pyjnius', 'spacy', 'en-core-web-sm', 'en-core-web-lg', 'flash-attn')}
    expected = {'torch': '2.6.0', 'vllm': '0.8.4', 'transformers': '4.51.1', 'ray': '2.46.0',
                'tensordict': '0.7.2', 'numpy': '1.26.4', 'pyserini': '0.17.0', 'pyjnius': '1.7.0',
                'en-core-web-sm': '3.7.1', 'en-core-web-lg': '3.7.1', 'flash-attn': '2.7.4.post1'}
    for name, version in expected.items():
        require(versions[name].split('+')[0] == version, f'{name}: expected {version}, got {versions[name]}')
    require(versions['spacy'] in ('3.7.2', '3.7.5'), 'Unexpected spaCy version')
    require(torch.version.cuda == '12.4', f'Expected CUDA 12.4 PyTorch build, got {torch.version.cuda}')
    require(str(flash_attn_2_cuda.__file__).endswith('.so'), 'FlashAttention must be the compiled extension')
    require(Path(verl.__file__).resolve().is_relative_to(ROOT / 'verl-agent'), 'verl import points outside this project')
    for name in ('en_core_web_sm', 'en_core_web_lg'):
        nlp = spacy.load(name)
        require(len(nlp('a blue cotton shirt')) > 0, f'Failed to load {name}')
    # Exercise the torch/tensordict interface used by verl without allocating CUDA.
    proto = DataProto.from_dict(tensors={'input_ids': torch.zeros((2, 4), dtype=torch.long)})
    require(len(proto) == 2, 'DataProto / tensordict incompatibility')

    java = subprocess.run([str(Path(os.environ['JAVA_HOME']) / 'bin/java'), '-version'],
                          check=True, capture_output=True, text=True).stderr
    require('11.0.32.1' in java, f'Unexpected Java runtime: {java}')
    data = ROOT / 'envdata/webshop_data/data'
    for name, asset in assets['dataset']['files'].items():
        require(sha256(data / name) == asset['sha256'], f'Dataset checksum mismatch: {name}')
    webshop = ROOT / 'verl-agent' / WEBSHOP
    index = ROOT / 'envdata/webshop_data/search_engine/indexes'
    require((webshop / 'data').resolve() == data.resolve(), 'WebShop catalogue link points to another dataset')
    require((webshop / 'search_engine/indexes').resolve() == index.resolve(), 'WebShop index link points elsewhere')
    products = json.loads((data / 'items_shuffle_1000.json').read_text())
    asins = {p['asin'] for p in products}
    searcher = LuceneSearcher(str(index))
    require(searcher.num_docs == len(asins) == assets['protocol']['products'], 'Lucene/catalogue size mismatch')
    require(all(searcher.doc(asin) is not None for asin in asins), 'Lucene index and catalogue ASINs differ')
    require(bool(searcher.search('shirt', k=3)), 'Lucene keyword search returned no results')
    searcher.close()
    parquets = {}
    for split, rows in [('train', assets['protocol']['train_rows']), ('test', assets['protocol']['test_rows'])]:
        table = pq.read_table(ROOT / f'envdata/webshop_data/text/{split}.parquet')
        require(len(table) == rows, f'Unexpected {split} parquet row count')
        for i, row in enumerate(table.to_pylist()):
            require(row['data_source'] == 'text' and row['ability'] == 'agent'
                    and row['prompt'] == [{'role': 'user', 'content': ''}]
                    and row['extra_info'] == {'split': split, 'index': i}, f'Invalid {split} row {i}')
        parquets[split] = rows
    episodes = []
    worker = WebshopWorker(0, {'observation_mode': 'text', 'num_products': None, 'human_goals': False,
                              'file_path': str(data / 'items_shuffle_1000.json'),
                              'attr_path': str(data / 'items_ins_v2_1000.json')})
    try:
        goals = len(worker.get_goals())
        require(goals == assets['protocol']['goals'], f'Expected 6910 synthetic goals; got {goals}')
        for case in assets['smoke_episodes']:
            obs, info = worker.reset(case['goal_index'])
            require(bool(obs) and 'available_actions' in info, 'Reset failed')
            for step, action in enumerate(case['actions']):
                obs, reward, done, info = worker.step(action)
                if step < len(case['actions']) - 1:
                    require(not done and reward == 0, 'Unexpected intermediate reward/termination')
            require(done, f'Purchase did not finish for goal {case["goal_index"]}')
            require(abs(info['task_score'] - case['task_score']) < 1e-8, f'Wrong graded score: {info}')
            require(reward == case['reward'] and info['won'] == (reward == 10), 'Binary reward mapping changed')
            episodes.append({'goal_index': case['goal_index'], 'task_score': info['task_score'], 'reward': reward})
    finally:
        worker.close()
    model_result = {'checked': False, 'reason': '--skip-model'}
    if not skip_model:
        from safetensors import safe_open
        from transformers import AutoConfig, AutoTokenizer
        model = assets['model']
        snapshot = ROOT / 'hf/hub' / ('models--' + model['repo'].replace('/', '--')) / 'snapshots' / model['revision']
        require(snapshot.is_dir(), f'Missing model snapshot: {snapshot}')
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        require(config.model_type == 'qwen2' and bool(tokenizer.encode('Ready.')), 'Model config/tokenizer failed')
        weight_index = snapshot / 'model.safetensors.index.json'
        shards = set(json.loads(weight_index.read_text())['weight_map'].values()) if weight_index.exists() else {'model.safetensors'}
        for shard in shards:
            require((snapshot / shard).is_file(), f'Missing model shard: {shard}')
            with safe_open(str(snapshot / shard), framework='pt', device='cpu') as weights:
                require(bool(weights.keys()), f'Empty model shard: {shard}')
        model_result = {'checked': True, 'repo': model['repo'], 'revision': model['revision'], 'shards': len(shards)}
    pip = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    marker = Path(sys.prefix) / '.webshop-setup.json'
    managed = marker.exists()
    if managed:
        require(json.loads(marker.read_text())['lock_sha256'] == sha256(ROOT / 'requirements/webshop-lock.txt'),
                'Managed venv was built with a different dependency lock')
    require(not managed or pip.returncode == 0, f'Managed environment has dependency conflicts:\n{pip.stdout}{pip.stderr}')
    if pip.returncode:
        print('Pre-existing venv metadata conflicts (packages preserved):\n' + pip.stdout, flush=True)
    return {'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
            'python': sys.executable, 'managed_venv': managed, 'versions': versions,
            'java_version': java.strip(), 'products': len(asins), 'index_docs': len(asins), 'goals': goals,
            'parquets': parquets, 'scripted_episodes': episodes, 'model': model_result,
            'pip_check': {'ok': pip.returncode == 0, 'output': (pip.stdout + pip.stderr).strip()},
            'gpu_kernel_checked': False, 'note': 'CPU provisioning checks; scripted rewards are not model evaluation scores.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', help='Generate only missing index/parquet assets')
    parser.add_argument('--skip-model', action='store_true')
    parser.add_argument('--report', type=Path, default=ROOT / '.cache/webshop-setup/report.json')
    args = parser.parse_args()
    assets = json.loads((ROOT / 'requirements/webshop-assets.json').read_text())
    try:
        if args.prepare:
            prepare_inputs(assets)
        result = validate(assets, args.skip_model)
    except Exception as error:
        atomic_json(args.report, {'status': 'failed', 'error': str(error), 'python': sys.executable})
        raise
    atomic_json(args.report, result)
    print(json.dumps({key: result[key] for key in ('status', 'products', 'goals', 'parquets', 'scripted_episodes')}, indent=2))


if __name__ == '__main__':
    main()
