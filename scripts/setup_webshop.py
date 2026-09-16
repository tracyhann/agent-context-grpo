#!/usr/bin/env python3
"""Provision the pinned 1,000-product WebShop runtime without replacing existing assets."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WEBSHOP = Path('agent_system/environments/env_package/webshop/webshop')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def run(args, *, env=None, capture=False):
    return subprocess.run([str(a) for a in args], cwd=ROOT, env=env, check=True,
                          text=True, stdout=subprocess.PIPE if capture else None)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def download(asset, destination):
    """Never replace a pre-existing file with different contents."""
    destination = Path(destination)
    if destination.is_symlink() and not destination.exists():
        raise RuntimeError(f'Broken asset link left untouched: {destination}')
    if destination.exists():
        if sha256(destination) != asset['sha256']:
            raise RuntimeError(f'Checksum mismatch; existing file left untouched: {destination}')
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.part')
    for attempt in range(3):
        try:
            print(f'Downloading {destination.name}', flush=True)
            with urllib.request.urlopen(asset['url'], timeout=120) as response, partial.open('wb') as out:
                shutil.copyfileobj(response, out, 4 * 1024 * 1024)
            if sha256(partial) != asset['sha256']:
                raise RuntimeError(f'Download checksum mismatch: {destination.name}')
            partial.replace(destination)
            return destination
        except (OSError, RuntimeError):
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2)


def extract(archive, destination):
    """Extract trusted, hash-verified archives with traversal/link checks as well."""
    destination = Path(destination).resolve()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(f'Unsafe archive member: {member.name}')
            if member.isdev() or member.isfifo():
                raise RuntimeError(f'Unsupported archive member: {member.name}')
            if member.issym() or member.islnk():
                base = target.parent if member.issym() else destination
                if not (base / member.linkname).resolve().is_relative_to(destination):
                    raise RuntimeError(f'Unsafe archive link: {member.name}')
        tar.extractall(destination)


def patch_files():
    return sorted(p for p in (ROOT / 'patches/verl-agent').rglob('*')
                  if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')


def verify_source():
    source = ROOT / 'verl-agent'
    for relative in ('verl/__init__.py', str(WEBSHOP / 'web_agent_site/engine/engine.py')):
        if not (source / relative).is_file():
            raise RuntimeError(f'Missing runtime source: {source / relative}')
    for patch in patch_files():
        target = source / patch.relative_to(ROOT / 'patches/verl-agent')
        if not target.is_file() or sha256(patch) != sha256(target):
            raise RuntimeError(f'Runtime overlay differs: {target}. Resolve it before setup; '
                               'setup does not overwrite source that may be in use by training.')


def provision_source(assets, cache):
    source = ROOT / 'verl-agent'
    if source.exists():
        verify_source()
        print('Reusing existing patched verl-agent source', flush=True)
        return
    archive = download(assets['source'], cache / assets['source']['name'])
    with tempfile.TemporaryDirectory(prefix='.webshop-source-', dir=ROOT) as staging:
        extract(archive, staging)
        entries = list(Path(staging).iterdir())
        if len(entries) != 1 or not entries[0].is_dir():
            raise RuntimeError('Unexpected verl-agent archive layout')
        for patch in patch_files():
            target = entries[0] / patch.relative_to(ROOT / 'patches/verl-agent')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(patch, target)
        entries[0].rename(source)
    verify_source()


def provision_java(assets, cache):
    java = assets['java']
    home = ROOT / 'jdk' / java['directory']
    if (home / 'bin/java').is_file():
        return home
    if home.exists() or home.is_symlink():
        raise RuntimeError(f'Incomplete JDK left untouched: {home}')
    home.parent.mkdir(parents=True, exist_ok=True)
    archive = download(java, cache / java['name'])
    with tempfile.TemporaryDirectory(prefix='.webshop-java-', dir=home.parent) as staging:
        extract(archive, staging)
        extracted = Path(staging) / java['directory']
        if not (extracted / 'bin/java').is_file():
            raise RuntimeError('Unexpected JDK archive layout')
        extracted.rename(home)
    return home


def link_directory(link, target):
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f'Conflicting link left untouched: {link}')
        return
    if link.exists():
        # Upstream archives can contain an empty placeholder directory.
        if not link.is_dir() or any(link.iterdir()):
            raise RuntimeError(f'Existing runtime data left untouched: {link}; expected link to {target}')
        link.rmdir()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent), target_is_directory=True)


def runtime_env(venv, java_home):
    env = dict(os.environ)
    env.update(JAVA_HOME=str(java_home), HF_HOME=str(ROOT / 'hf'),
               PYTHONPATH=os.pathsep.join(map(str, [ROOT, ROOT / 'verl-agent',
                   ROOT / 'verl-agent' / WEBSHOP])),
               PATH=str(venv / 'bin') + os.pathsep + str(java_home / 'bin') + os.pathsep + env['PATH'],
               PIP_CACHE_DIR=str(ROOT / '.cache/pip'), XDG_CACHE_HOME=str(ROOT / '.cache'),
               CUDA_VISIBLE_DEVICES='', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
               JAVA_TOOL_OPTIONS='-XX:ActiveProcessorCount=1 -XX:+UseSerialGC -Xss512k -Xms32m -Xmx512m')
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'RAYON_NUM_THREADS'):
        env[name] = '1'
    return env


def provision_venv(args, assets, cache, env):
    venv = args.venv
    python = venv / 'bin/python'
    marker = venv / '.webshop-setup.json'
    installing = venv / '.webshop-setup-installing'
    lock_hash = sha256(ROOT / 'requirements/webshop-lock.txt')
    if not python.exists():
        if venv.exists() and any(venv.iterdir()):
            raise RuntimeError(f'Incomplete/unrecognized venv left untouched: {venv}')
        version = run([args.python, '-c', 'import sys; print("%s.%s" % sys.version_info[:2])'],
                      capture=True).stdout.strip()
        if version != '3.10':
            raise RuntimeError(f'Need CPython 3.10 (found {version}); pass --python /path/to/python3.10')
        venv.parent.mkdir(parents=True, exist_ok=True)
        run([args.python, '-m', 'venv', venv])
        installing.touch()
    if installing.exists():
        run([python, '-m', 'pip', 'install', 'pip==25.3', 'setuptools==75.8.0', 'wheel==0.45.1'], env=env)
        run([python, '-m', 'pip', 'install', '-r', ROOT / 'requirements/webshop-lock.txt'], env=env)
        wheels = [download(asset, cache / asset['name'])
                  for asset in [assets['flash_attention'], *assets['spacy_models']]]
        run([python, '-m', 'pip', 'install', '--no-deps', *wheels], env=env)
        # Upstream setup.py disagrees with its training requirements about tensordict.
        # Import the pinned, patched source using .pth; do not invoke that resolver.
        site = Path(run([python, '-c', 'import sysconfig; print(sysconfig.get_path("purelib"))'],
                        capture=True).stdout.strip())
        (site / 'agent_context_grpo.pth').write_text(str(ROOT) + '\n' + str(ROOT / 'verl-agent') + '\n')
        run([python, '-m', 'pip', 'check'], env=env)
        atomic_json(marker, {'lock_sha256': lock_hash, 'source_revision': assets['source']['revision']})
        installing.unlink()
    elif marker.exists():
        if json.loads(marker.read_text())['lock_sha256'] != lock_hash:
            raise RuntimeError('Dependency lock changed. Use --venv with a new directory; '
                               'setup never upgrades an existing runtime in place.')
        print(f'Reusing managed venv: {venv}', flush=True)
    else:
        print(f'Reusing pre-existing venv without installing/upgrading packages: {venv}', flush=True)
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Validate only; no downloads, installs or rebuilds')
    parser.add_argument('--skip-model', action='store_true', help='Skip the Qwen2.5-1.5B download and weight check')
    parser.add_argument('--python', default='python3.10', help='CPython 3.10 executable for a new venv')
    parser.add_argument('--venv', type=Path, default=ROOT / '.venv-webshop', help='Alternate venv; never upgrades existing ones')
    parser.add_argument('--download-cache', type=Path, default=ROOT / '.cache/webshop-setup/downloads')
    args = parser.parse_args()
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        parser.error('The pinned CUDA/FlashAttention wheels require Linux x86_64.')
    args.venv = args.venv.expanduser().absolute()
    cache = args.download_cache.expanduser().absolute()
    assets = json.loads((ROOT / 'requirements/webshop-assets.json').read_text())
    state = ROOT / '.cache/webshop-setup'
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'setup.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        atomic_json(state / 'report.json', {'status': 'checking', 'check_only': args.check})
        java_home = ROOT / 'jdk' / assets['java']['directory']
        env = runtime_env(args.venv, java_home)
        python = args.venv / 'bin/python'
        if not args.check:
            provision_source(assets, cache)
            provision_java(assets, cache)
            python = provision_venv(args, assets, cache, env)
            data = ROOT / 'envdata/webshop_data/data'
            dataset = assets['dataset']
            for name, metadata in dataset['files'].items():
                download({**metadata, 'url': f'https://huggingface.co/datasets/{dataset["repo"]}/resolve/'
                                            f'{dataset["revision"]}/{name}'}, data / name)
            webshop = ROOT / 'verl-agent' / WEBSHOP
            link_directory(webshop / 'data', data)
            link_directory(webshop / 'search_engine/indexes', ROOT / 'envdata/webshop_data/search_engine/indexes')
            if not args.skip_model:
                model = assets['model']
                code = ('from huggingface_hub import snapshot_download; '
                        'snapshot_download(repo_id=' + repr(model['repo']) + ', revision=' + repr(model['revision']) +
                        ', allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"], max_workers=2)')
                run([python, '-c', code], env=env)
        verify_source()
        command = [python, ROOT / 'scripts/check_webshop.py', '--report', state / 'report.json']
        if not args.check:
            command.append('--prepare')
        if args.skip_model:
            command.append('--skip-model')
        run(command, env=env)
        print(f'WebShop CPU validation passed. Report: {state / "report.json"}', flush=True)
        print('Training launchers separately test FlashAttention on the selected GPU before starting.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        report = ROOT / '.cache/webshop-setup/report.json'
        # Preserve the checker's more specific error if it already wrote one.
        previous = json.loads(report.read_text()) if report.exists() else {}
        if previous.get('status') != 'failed':
            atomic_json(report, {'status': 'failed', 'error': str(error)})
        sys.exit(f'WebShop setup failed: {error}')
