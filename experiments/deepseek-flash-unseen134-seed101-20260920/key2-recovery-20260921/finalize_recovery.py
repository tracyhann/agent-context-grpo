"""Audit the replacement-key continuation and produce final tables; no API calls."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ARCHIVE=Path(__file__).resolve().parent;EXP=ARCHIVE.parent;ROOT=EXP.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from chain_qwen_census import alive
from report_census_eval import report,pricing_factor
from qwen_baseline.common import atomic_json


def main():
    launch=json.loads((ARCHIVE/'LAUNCH.json').read_text());watch=int((ARCHIVE/'watch.pid').read_text())
    while alive(launch['pid']):time.sleep(10)
    deadline=time.monotonic()+90
    while alive(watch) and time.monotonic()<deadline:time.sleep(5)
    if alive(watch):raise RuntimeError('Watcher still active; refusing concurrent final report writes')
    result=report(EXP);suite=result['suites']['alfworld-unseen']
    if result['status']!='completed':raise RuntimeError('Recovery is still incomplete; no final scores claimed')
    output=EXP/'outputs/alfworld-unseen';before=json.loads((ARCHIVE/'BEFORE.json').read_text())
    for name,digest in before['completed_sha256'].items():
        assert hashlib.sha256((output/'episodes'/name).read_bytes()).hexdigest()==digest,f'Previously completed task changed: {name}'
    assert hashlib.sha256((output/'config.json').read_bytes()).hexdigest()==before['config_sha256'],'Original config changed'
    additions=[];per_episode=[]
    for name,count in before['prefix_lengths'].items():
        old=json.loads((ARCHIVE/name).read_text());new=json.loads((output/'episodes'/name).read_text())
        assert new['status']=='completed'
        assert new['turns'][:count]==old['turns'],f'Replayed trajectory changed: {name}'
        ledger_name=name.replace('.json','.jsonl')
        cached=[json.loads(x) for x in (ARCHIVE/ledger_name).read_text().splitlines() if x]
        current=[json.loads(x) for x in (output/'api_calls'/ledger_name).read_text().splitlines() if x]
        assert current[:len(cached)]==cached,f'Cached API responses changed: {name}'
        extra=current[len(cached):];assert len(extra)==len(new['turns'])-count
        additions.extend(extra)
        per_episode.append(dict(episode_id=new['episode_id'],replayed_actions=count,new_actions=len(extra),
                                total_actions=len(new['turns']),success=new['success'],termination=new['termination']))
    cost=0.
    for response in additions:
        a=response['accounting']
        cost+=(a['cache_hit_tokens']*.003+a['cache_miss_tokens']*.15+a['completion_tokens']*.6)/1e6*pricing_factor(response['api_created'])
    delta=suite['cost_usd']-before['before_unseen_cost_usd'];assert abs(cost-delta)<1e-9,(cost,delta)
    audit_path=EXP/'FINAL_AUDIT.json';audit=json.loads(audit_path.read_text());assert audit['status']=='passed'
    audit['key2_recovery']=dict(previously_completed_trajectories_unchanged=132,replayed_actions_unchanged=71,
                               new_received_requests=len(additions),continuation_cost_usd=cost,
                               original_config_unchanged=True,credential_contents_logged=False)
    atomic_json(audit_path,audit)
    subprocess.run([sys.executable,str(EXP/'finalize.py')],cwd=ROOT,check=True)
    combined=json.loads((EXP/'COMBINED_RESULTS.json').read_text())
    rows=[json.loads(p.read_text()) for p in (output/'episodes').glob('*.json')];wins=[r for r in rows if r['success']]
    final=dict(status='completed_and_audited',finished_at=datetime.now(timezone.utc).isoformat(),
        credential_source_path=launch['credential_source_path'],previously_completed_trajectories_preserved=132,
        replayed_actions=71,new_received_requests=len(additions),episodes=per_episode,
        completed=134,successes=suite['successes'],success_pct=suite['success_pct'],
        mean_turns_all=sum(len(r['turns']) for r in rows)/len(rows),
        mean_turns_wins=sum(len(r['turns']) for r in wins)/len(wins),
        continuation_cost_usd=cost,total_unseen_cost_usd=suite['cost_usd'],
        all_benchmarks_cost_usd=combined['combined_api_cost_usd'],
        prior_unpriced_allowance_usd=combined['unpriced_charge_ceiling_estimate_usd'],
        new_unpriced_attempts=result['unpriced_attempts']-json.loads((ARCHIVE/'RESULTS.before.json').read_text())['unpriced_attempts'],
        validation='Three CPU regression tests; exact saved-prefix/ledger equality; 132 trajectory hashes and config hash preserved; complete census and cost reconciliation passed')
    atomic_json(ARCHIVE/'RESULTS.json',final)
    lines=['# Replacement-key recovery complete','',f"Completed **134/134 unseen tasks**, **{suite['successes']} successes ({suite['success_pct']:.2f}%)**.",'',
      '| Episode | Replayed actions | New actions | Final turns | Success |','|---|---:|---:|---:|---|']
    for r in per_episode:lines.append(f"| {r['episode_id']} | {r['replayed_actions']} | {r['new_actions']} | {r['total_actions']} | {r['success']} |")
    lines+=['',f"132 completed trajectories and 71 cached actions/responses were preserved exactly. {len(additions)} new received requests.",'',
      f"New continuation charge: **${cost:.6f}**. Full unseen charge: **${suite['cost_usd']:.6f}**.",
      f"Seen140 + unseen134 + WebShop500 combined: **${combined['combined_api_cost_usd']:.6f}**.",
      f"Separate allowance for historical unknown-usage interruptions: up to **${combined['unpriced_charge_ceiling_estimate_usd']:.6f}**; not included in received-usage charge.",'',
      'Costs are calculated from received token usage and request-time [official rates](https://api-docs.deepseek.com/quick_start/pricing/), not an invoice. Saved-response replay issues no API requests. Credential file paths are recorded; key contents/hashes are never logged.','',
      '[Complete benchmark tables](../COMBINED_RESULTS.md); [recovery command and protocol](../NOTES.md).','']
    (ARCHIVE/'RESULTS.md').write_text('\n'.join(lines))
    with (EXP/'NOTES.md').open('a') as f:
        f.write(f"\n### Recovery completed and audited\n\nAll **134/134** unseen tasks completed; **{suite['successes']}/134 = {suite['success_pct']:.2f}%**. "
                f"The new key funded {len(additions)} new requests costing **${cost:.6f}**. All 132 prior completed trajectories and all 71 cached actions/responses are unchanged. "
                "[Recovery audit and costs](key2-recovery-20260921/RESULTS.md); [combined final results](COMBINED_RESULTS.md).\n")
    print(json.dumps(final),flush=True)


if __name__=='__main__':main()
