"""Combine the completed seen/WebShop results and this unseen extension, without new API calls."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

exp=Path(__file__).resolve().parent;root=exp.parents[1]
old=exp.parent/'deepseek-flash-default-full-seed101-20260920'
a=json.loads((old/'RESULTS.json').read_text());b=json.loads((exp/'RESULTS.json').read_text())
assert a['status']==b['status']=='completed','Both runs must complete before final reporting'
assert json.loads((old/'FINAL_AUDIT.json').read_text())['status']=='passed'
assert json.loads((exp/'REPLAY_VALIDATION.json').read_text())['status']=='passed'
inputs=json.loads((exp/'input-sha256.json').read_text())
for name,digest in inputs.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
seen=a['benchmarks']['alfworld'];unseen=b['suites']['alfworld-unseen'];ws=a['benchmarks']['webshop']
assert (seen['completed'],unseen['completed'],ws['completed'])==(140,134,500)
allowance=a['unpriced_charge_ceiling_estimate_usd']+b['unpriced_ceiling_estimate_usd']
result=dict(completed=datetime.now(timezone.utc).isoformat(),model='deepseek-flash',seed=101,
    alfworld_seen=seen,alfworld_unseen=unseen,webshop=ws,
    incremental_unseen_cost_usd=b['cost_usd'],combined_api_cost_usd=a['total_cost_usd']+b['cost_usd'],
    unpriced_attempts=a['unpriced_api_attempts']+b['unpriced_attempts'],unpriced_charge_ceiling_estimate_usd=allowance,
    source_results_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [old/'RESULTS.json',exp/'RESULTS.json']})
(exp/'COMBINED_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
labels=['Pick','Look','Clean','Heat','Cool','Pick2']
lines=['# DeepSeek Flash — complete seed101 evaluation','',
    'One round: ALFWorld140 seen +134 unseen, WebShop500. Thinking high, output65536, history2, ALF50 / WebShop15 actions. Percent units.','',
    '| ALFWorld split | Pick | Look | Clean | Heat | Cool | Pick2 | All |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name,s,total in [('Seen (140)',seen,seen['observed_success_pct']),('Unseen (134)',unseen,unseen['success_pct'])]:
    lines.append('| '+name+' | '+' | '.join(f'{v:.2f}' for v in [s['task_types'][k]['success_pct'] for k in labels]+[total])+' |')
lines+=['','| WebShop | Score | Succ. |','|---|---:|---:|',f"| 500 test goals | {ws['observed_score_pct']:.2f} | {ws['observed_success_pct']:.2f} |",'',
    f"ALFWorld successes: seen {seen['observed_successes']}/140; unseen {unseen['successes']}/134. WebShop successes: {ws['observed_successes']}/500.",'',
    'All uses the micro-average. All horizon failures count. WebShop uses the original 1K-product-catalog scorer, without post-hoc item-option correction.','',
    '| Evaluation | Received-usage cost USD |','|---|---:|',
    f"| ALFWorld seen140 | ${seen['cost_usd']:.6f} |",f"| ALFWorld unseen134 (new) | ${unseen['cost_usd']:.6f} |",f"| WebShop500 | ${ws['cost_usd']:.6f} |",f"| Total | **${result['combined_api_cost_usd']:.6f}** |",'',
    f"{result['unpriced_attempts']} interrupted attempts lacked returned usage. Their charges are excluded above; conservative additional allowance **up to ${allowance:.6f}**. This is an allowance assuming full output caps, not measured usage or an invoice.",'',
    '[Unseen protocol and reproduction](NOTES.md); [seen/WebShop protocol and reproduction](../deepseek-flash-default-full-seed101-20260920/NOTES.md). Exact tasks, data/package/source snapshots and raw prompt/action/token ledgers are retained. API generation is not seedable; seed101 fixes the environment and task order. Model aliases can change over time.','']
(exp/'COMBINED_RESULTS.md').write_text('\n'.join(lines))
audit=json.loads((exp/'FINAL_AUDIT.json').read_text());audit.update(source_and_data_hashes_verified=len(inputs),local_replays='REPLAY_VALIDATION.json',combined_results='COMBINED_RESULTS.json',finalized=result['completed'])
(exp/'FINAL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['incremental_unseen_cost_usd','combined_api_cost_usd','unpriced_attempts','unpriced_charge_ceiling_estimate_usd']}))
