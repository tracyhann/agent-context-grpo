#!/usr/bin/env python3
"""Serve the pinned Qwen3.8-27B checkpoint from its isolated vLLM environment."""
import argparse
import csv
import datetime
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qwen_baseline.common import MODEL, SERVED_MODEL, atomic_json, snapshot_path


def gpu_preflight(indices):
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    devices = {int(row[0]): [int(x.strip()) for x in row[1:]] for row in csv.reader(output.splitlines())}
    for index in indices:
        if index not in devices:
            raise RuntimeError(f'GPU {index} does not exist')
        total, used, utilization = devices[index]
        if used > 2048 or utilization > 10:
            raise RuntimeError(f'GPU {index} is busy ({used} MiB used, {utilization}% utilization). '
                               'Select idle GPUs after existing jobs finish.')
    return {str(index): devices[index] for index in indices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', required=True, help='Physical idle GPU indices, e.g. 2,3; TP equals their count')
    parser.add_argument('--port', type=int, default=8018)
    parser.add_argument('--max-model-len', type=int, default=8192)
    parser.add_argument('--max-num-seqs', type=int, default=8)
    parser.add_argument('--gpu-memory-utilization', type=float, default=.85)
    parser.add_argument('--enforce-eager', action='store_true', help='Disable CUDA graphs for initial compatibility diagnosis')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        indices = [int(x) for x in args.gpus.split(',')]
        if not indices or len(indices) != len(set(indices)) or min(indices) < 0:
            raise ValueError()
    except ValueError:
        parser.error('gpus must be distinct nonnegative integer indices')
    if not 0 < args.gpu_memory_utilization < 1 or min(args.max_model_len, args.max_num_seqs) < 1 or not 1 <= args.port <= 65535:
        parser.error('Invalid serving limits')
    python = ROOT/'.venv-qwen/bin/python'
    if not python.is_file():
        parser.error('Create .venv-qwen and install requirements-server.txt first')
    snapshot = snapshot_path()
    command = [str(python), '-m', 'vllm.entrypoints.openai.api_server',
               '--model', str(snapshot), '--served-model-name', SERVED_MODEL,
               '--host', '127.0.0.1', '--port', str(args.port), '--dtype', MODEL['dtype'],
               '--tensor-parallel-size', str(len(indices)), '--max-model-len', str(args.max_model_len),
               '--max-num-seqs', str(args.max_num_seqs), '--gpu-memory-utilization', str(args.gpu_memory_utilization),
               '--generation-config', 'vllm', '--language-model-only', '--seed', '0']
    if args.enforce_eager:
        command.append('--enforce-eager')
    print('CUDA_VISIBLE_DEVICES=' + ','.join(map(str, indices)) + ' ' + shlex.join(command), flush=True)
    if args.dry_run:
        return
    gpus = gpu_preflight(indices)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=','.join(map(str, indices)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
               HF_HOME=str(ROOT/'hf'), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               VLLM_WORKER_MULTIPROC_METHOD='spawn', OMP_NUM_THREADS='1')
    # Training environments sometimes export settings for their older vLLM stack.
    for key in ['VLLM_ATTENTION_BACKEND', 'VLLM_USE_V1', 'PYTHONPATH']:
        env.pop(key, None)
    atomic_json(ROOT/f'.local/qwen38-server-{args.port}.json', dict(model=MODEL, command=command,
                gpu_indices=indices, gpu_preflight=gpus, pid=os.getpid(),
                started=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    os.chdir(ROOT)
    os.execve(python, command, env)


if __name__ == '__main__':
    main()
