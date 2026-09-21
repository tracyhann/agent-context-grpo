from pathlib import Path
from datetime import datetime,timezone
import concurrent.futures,json,time,urllib.request,statistics,sys
root=Path('/workspace/agent-context-grpo');sys.path.insert(0,str(root));from qwen_baseline.common import post_json,SERVED_MODEL
exp=root/'experiments/qwen38-27b-full-heldout-seed101-20260920';out=exp/'acceleration-20260921'
episode=json.loads((exp/'outputs/alfworld-seen/episodes/0001.json').read_text());prompt=episode['turns'][0]['prompt']
ids=post_json('http://127.0.0.1:8019','tokenize',dict(model=SERVED_MODEL,messages=[dict(role='user',content=prompt)],add_generation_prompt=True,chat_template_kwargs=dict(enable_thinking=True,preserve_thinking=True,reasoning_effort='xhigh')))['tokens']
def call(i,n):
 start=time.monotonic();d=post_json('http://127.0.0.1:8019/v1','completions',dict(model=SERVED_MODEL,prompt=ids,max_tokens=n,min_tokens=n,ignore_eos=True,temperature=1.,top_p=.95,top_k=20,min_p=0.,presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,seed=90000000+i,stream=False),timeout=180)
 return dict(seconds=time.monotonic()-start,tokens=d['usage']['completion_tokens'],finish_reason=d['choices'][0]['finish_reason'])
call(0,32);runs=[]
for batch in [1,8,16]:
 start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=batch) as pool:r=list(pool.map(lambda i:call(i,256),range(batch)))
 duration=time.monotonic()-start
 runs.append(dict(concurrency=batch,tokens=sum(x['tokens'] for x in r),elapsed_s=duration,tokens_per_s=sum(x['tokens'] for x in r)/duration,median_request_s=statistics.median(x['seconds'] for x in r),requests=r))
 print(json.dumps(runs[-1]),flush=True)
record=dict(created=datetime.now(timezone.utc).isoformat(),purpose='Serving throughput diagnostic only; synthetic fixed-length requests excluded from benchmark scores',model=SERVED_MODEL,port=8019,compiled=True,graphs=True,gpus=[2,3],runs=runs,original_observed_tokens_per_s={'1':12.1,'8':91.6},comparison_caveat='Original rates are production observations, not a controlled same-batch benchmark')
(out/'THROUGHPUT.json').write_text(json.dumps(record,indent=2)+'\n')
