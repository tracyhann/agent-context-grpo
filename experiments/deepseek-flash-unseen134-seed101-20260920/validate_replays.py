"""Replay one completed unseen success and one failure, without API calls."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import benchmark_census as census
import replay_deepseek_episode as replay
replay.evaluator.environment=census.environment
exp=Path(__file__).resolve().parent
chosen={}
for p in sorted((exp/'outputs/alfworld-unseen/episodes').glob('*.json')):
 e=json.loads(p.read_text())
 if e['status']=='completed':chosen.setdefault(bool(e['success']),p)
assert set(chosen)=={False,True}
results=[]
for success,p in chosen.items():
 output=io.StringIO()
 with contextlib.redirect_stdout(output):
  sys.argv=['replay_deepseek_episode.py',str(p)];replay.main()
 (exp/f'replay-{p.stem}.log').write_text(output.getvalue())
 results.append(json.loads(output.getvalue().strip().splitlines()[-1]))
record={'status':'passed','api_calls':0,'replays':results,'validator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(exp/'REPLAY_VALIDATION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
