"""Join an explicitly managed census instead of starting a duplicate evaluation."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from qwen_baseline.common import MODEL, atomic_json

MARKER = 'ACCELERATED_EXECUTION.json'


def stamp():
    return datetime.now(timezone.utc).isoformat()


def identity(pid):
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] not in ('Z', 'X') else None
    except (OSError, ValueError, TypeError):
        return None


def protocol(args):
    return dict(suite=args.suite, seed=args.seed, max_tokens=args.max_tokens,
                max_model_len=args.max_model_len, thinking=not args.no_thinking,
                model=MODEL, goal_manifest_sha256=hashlib.sha256(args.webshop_goal_manifest.read_bytes()).hexdigest()
                if args.webshop_goal_manifest else None)


def checked_marker(args):
    output = args.output.resolve()
    marker = json.loads((output/MARKER).read_text())
    if marker['protocol'] != protocol(args):
        raise ValueError('Managed evaluation protocol differs; refusing to join/overwrite')
    if Path(marker['output']).resolve() != output:
        raise ValueError('Managed output directory differs')
    return marker


def claim_owner(args):
    """Only the recorded owner command may create this suite; retain returned lock."""
    marker = checked_marker(args)
    if args.base_url != marker['base_url'] or args.workers != marker['workers']:
        raise ValueError('Managed owner endpoint/worker count differs from plan')
    lock = (args.output/'.managed-evaluation.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output/'config.json').exists():
            raise ValueError('Managed owner refuses to overwrite an existing census')
        marker.update(status='running', owner_pid=os.getpid(), owner_start_ticks=identity(os.getpid()), started=stamp())
        atomic_json(args.output/MARKER, marker)
    except BaseException:
        lock.close()
        raise
    return lock


def finish_owner(args, metrics):
    marker = checked_marker(args)
    marker.update(status=metrics['status'], ended=stamp(), metrics=metrics)
    atomic_json(args.output/MARKER, marker)


def validate_finished(output, marker):
    """Require every planned episode exactly once before returning success to chain."""
    metrics = json.loads((output/'metrics.json').read_text())
    if metrics.get('status') != 'completed':
        return 1
    config = json.loads((output/'config.json').read_text())
    count = marker['expected_count']
    if metrics['requested'] != count or metrics['completed'] != count or len(config['episodes']) != count:
        raise ValueError('Managed census count mismatch')
    planned = {item['episode_id']: item for item in config['episodes']}
    if len(planned) != count:
        raise ValueError('Managed census has duplicate planned IDs')
    settings = config['settings']; expected = marker['protocol']
    for key in ['seed', 'max_tokens', 'max_model_len', 'thinking']:
        if settings[key] != expected[key]:
            raise ValueError(f'Managed census config differs: {key}')
    if config['model'] != expected['model']:
        raise ValueError('Managed census model differs')
    found = set()
    for file in (output/'episodes').glob('*.json'):
        episode = json.loads(file.read_text()); eid = episode['episode_id']
        if eid in found or eid not in planned or episode['status'] != 'completed':
            raise ValueError('Managed census has incomplete/duplicate/unplanned episodes')
        if any(episode.get(k) != v for k, v in planned[eid].items()):
            raise ValueError('Managed census task identity differs')
        found.add(eid)
    if found != set(planned):
        raise ValueError('Managed census is missing episodes')
    return 0


def join_if_managed(args):
    """Original sequential controller waits for the independently scheduled owner."""
    output = args.output.resolve()
    if not (output/MARKER).exists():
        return None  # Preserve normal CLI's refusal to overwrite existing output.
    marker = checked_marker(args)
    print(f'Joining managed evaluation at {output}; no tasks will be repeated', flush=True)
    with (output/'.managed-evaluation.lock').open('a') as lock:
        while True:
            marker = checked_marker(args)
            if not marker.get('owner_pid'):
                coordinator = marker.get('coordinator_pid')
                if not coordinator or identity(coordinator) != marker.get('coordinator_start_ticks'):
                    raise RuntimeError('Managed evaluation was never started')
                time.sleep(1)
                continue
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(20)
                continue
            if marker['status'] not in ('completed', 'incomplete'):
                raise RuntimeError('Managed owner exited before completing its census')
            return validate_finished(output, marker)
