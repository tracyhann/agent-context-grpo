#!/usr/bin/env python3
"""Select the existing CPU environment for an ALFWorld or WebShop evaluation."""
import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('benchmark', choices=['alfworld', 'webshop'])
    args, remaining = parser.parse_known_args()
    venv = '.venv' if args.benchmark == 'alfworld' else '.venv-webshop'
    python = ROOT / venv / 'bin/python'
    if not python.is_file():
        raise SystemExit(f'Missing benchmark environment: {python}')
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        env[key] = '1'
    os.chdir(ROOT)
    os.execve(python, [str(python), str(ROOT/'qwen_baseline/evaluate.py'),
                      '--benchmark', args.benchmark, *remaining], env)


if __name__ == '__main__':
    main()
