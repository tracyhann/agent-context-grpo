"""Download the pinned public checkpoint and verify its complete shard set."""
import argparse
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = json.loads((Path(__file__).parent / 'model.json').read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=ROOT / 'hf/hub')
    parser.add_argument('--manifest', type=Path, default=ROOT / '.local/qwen38-model.json')
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi(token=False).model_info(MODEL['repo_id'], revision=MODEL['revision'], files_metadata=True)
    assert info.sha == MODEL['revision']
    path = Path(snapshot_download(MODEL['repo_id'], revision=MODEL['revision'],
                                 cache_dir=str(args.cache_dir), token=False, max_workers=4))
    sizes = {f.rfilename: f.size for f in info.siblings}
    for name, size in sizes.items():
        target = path / name
        if not target.is_file() or (size is not None and target.stat().st_size != size):
            raise RuntimeError(f'Missing or incomplete model file: {name}')
    index = json.loads((path / 'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    if not shards or any(not (path / shard).is_file() for shard in shards):
        raise RuntimeError('Incomplete safetensors shard set')
    result = dict(MODEL, snapshot=str(path), status='downloaded_verified_sizes_and_index',
                  verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  bytes=sum(v or 0 for v in sizes.values()), shards=shards, files=sizes)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temp = args.manifest.with_suffix('.tmp')
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(args.manifest)
    print(json.dumps({k: result[k] for k in ['repo_id', 'revision', 'snapshot', 'status', 'bytes']}), flush=True)


if __name__ == '__main__':
    main()
