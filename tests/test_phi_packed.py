#!/usr/bin/env python3
"""The packed (use_remove_padding) phi capture must equal the padded one.

With flash-attn available verl packs the batch instead of padding every sequence
to max_prompt_length + max_response_length, which is a >4x saving on this
workload. The estimator's affinity features are pooled at the last PROMPT token,
and in the packed layout that token is not at a fixed offset -- it has to be
located through the `indices` unpad_input returns. An off-by-one there would give
every sample another sample's features with no visible error, so this test
compares the two paths directly on the real model.

Needs one GPU and flash-attn; skips otherwise.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0]=[ROOT, os.path.join(ROOT,"verl-agent")]
# The Rust tokenizer builds a rayon pool sized from nproc; on a box whose
# cgroup pid budget is already spent by a training run that fails outright.
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

try:
    from flash_attn.bert_padding import index_first_axis, unpad_input
except Exception as _e:                                       # noqa: BLE001
    print(f"SKIP: no usable flash-attn ({_e})")
    sys.exit(0)
if not torch.cuda.is_available():
    print("SKIP: needs a GPU")
    sys.exit(0)
from transformers import AutoModelForCausalLM, AutoTokenizer

snap=os.path.join(ROOT,"hf","hub","models--Qwen--Qwen2.5-1.5B-Instruct","snapshots")
CKPT=os.path.join(snap, sorted(os.listdir(snap))[0])

class Stub:
    from verl.workers.actor.dp_actor import DataParallelPPOActor as _A
    _acg_final_norm=_A._acg_final_norm; _acg_hidden_hook=_A._acg_hidden_hook

tok=AutoTokenizer.from_pretrained(CKPT)
model=AutoModelForCausalLM.from_pretrained(CKPT, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2").cuda().eval()
stub=Stub(); stub.actor_module=model
mod=stub._acg_final_norm(); assert mod is not None

prompts=["You are in the kitchen. You see a fridge 1 and a cabinet 2.",
         "You are in the middle of a room holding an apple and a mug.",
         "You open cabinet 3. You see a plate 1."]
resps=["<action>go to fridge 1</action>","<action>open cabinet 2</action>",
       "<action>take plate 1</action>"]
penc=tok(prompts, return_tensors="pt", padding=True, padding_side="left")
renc=tok(resps, return_tensors="pt", padding=True, padding_side="right")
input_ids=torch.cat([penc["input_ids"], renc["input_ids"]],1).cuda()
attn=torch.cat([penc["attention_mask"], renc["attention_mask"]],1).cuda()
B,S=input_ids.shape; R=renc["input_ids"].shape[1]
col=S-R-1
print(f"batch {B} seqlen {S} response {R} -> last-prompt column {col}")

# ---- padded path
h=mod.register_forward_hook(stub._acg_hidden_hook)
with torch.no_grad():
    model(input_ids=input_ids, attention_mask=attn, use_cache=False)
padded=stub._acg_hook_out[:, col, :].detach().float().cpu()
h.remove()

# ---- packed path (what verl does with use_remove_padding=True)
ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attn)
ids_rmpad = ids_rmpad.transpose(0,1)
pos_ids = torch.clamp(attn.cumsum(-1)-1, min=0)
from einops import rearrange
pos_rmpad = index_first_axis(rearrange(pos_ids.unsqueeze(-1),"b s ... -> (b s) ..."), indices).transpose(0,1)
h=mod.register_forward_hook(stub._acg_hidden_hook)
with torch.no_grad():
    model(input_ids=ids_rmpad, attention_mask=None, position_ids=pos_rmpad, use_cache=False)
hp=stub._acg_hook_out
h.remove()

flat=(torch.arange(B, device=indices.device, dtype=indices.dtype)*S+col)
p=torch.searchsorted(indices, flat).clamp(max=indices.numel()-1)
assert bool((indices[p]==flat).all()), "index mapping wrong"
packed=hp[0, p, :].detach().float().cpu()

d=(padded-packed).abs().max().item()
rel=d/max(padded.abs().max().item(),1e-9)
cos=torch.nn.functional.cosine_similarity(padded, packed, dim=-1).min().item()
print(f"max|padded-packed| = {d:.4f}  relative {rel:.2e}  min cosine {cos:.6f}")
ok = cos > 0.999
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
