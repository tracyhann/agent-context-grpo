"""Real environments + synthetic HTTP actions. This does NOT measure model quality.

Run from the project root: .venv/bin/python qwen_baseline/tests/integration_smoke.py
Artifacts go under .local/qwen38-integration-smoke; no GPU/model inference occurs.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from qwen_baseline.common import SERVED_MODEL


class Handler(BaseHTTPRequestHandler):
    action = 'look'
    requests = 0

    def log_message(self, *args):
        pass

    def reply(self, status, value):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, dict(data=[dict(id=SERVED_MODEL)]))

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        assert request['model'] == SERVED_MODEL
        assert isinstance(request['prompt'], list) and isinstance(request['prompt'][0], int)
        type(self).requests += 1
        self.reply(200, dict(choices=[dict(text=f'<think>synthetic test</think><action>{self.action}</action>',
                                          finish_reason='stop')], usage=dict(completion_tokens=20)))


def main():
    root = ROOT/'.local/qwen38-integration-smoke'
    root.mkdir(parents=True, exist_ok=True)
    output_root = Path(tempfile.mkdtemp(prefix='smoke-', dir=root))
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    summary = dict(kind='synthetic HTTP actions with real benchmark environments; NOT Qwen results', checks={})
    try:
        for benchmark, action in [('alfworld', 'look'), ('webshop', 'search[mug]')]:
            Handler.action = action
            output = output_root/benchmark
            cmd = [sys.executable, str(ROOT/'qwen_baseline/run.py'), benchmark,
                   '--episodes', '2', '--workers', '2', '--max-steps', '2',
                   '--output', str(output), '--base-url', f'http://127.0.0.1:{server.server_port}/v1']
            with (output_root/f'{benchmark}.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=300)
                metrics = json.loads((output/'metrics.json').read_text())
                assert metrics['status'] == 'completed' and metrics['completed_episodes'] == 2
                records = [json.loads(x) for x in (output/'episodes.jsonl').read_text().splitlines()]
                assert all(len(r['turns']) == 2 for r in records)
                assert all(r['turns'][0]['action'] == action for r in records)
                if benchmark == 'webshop':
                    assert all('goal' in r['initial_session'] and 'options' in r['final_session'] for r in records)
                requests = Handler.requests
                subprocess.run(cmd + ['--resume'], cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=60)
                assert Handler.requests == requests, 'Completed episodes were rerun on resume'
                rejected = subprocess.run(cmd + ['--resume', '--temperature', '0.5'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=60)
                assert rejected.returncode != 0, 'Changed sampling configuration was accepted on resume'
            summary['checks'][benchmark] = dict(episodes=2, turns=4, resumed_without_requests=True, changed_config_rejected=True)
            print(benchmark, 'integration smoke passed', flush=True)
        (output_root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        print('Synthetic smoke artifacts:', output_root)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
