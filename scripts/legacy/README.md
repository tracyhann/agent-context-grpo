# Legacy scripts

These belong to the first hardware path: an sm_120 box where flash-attn had no
build and the then-current vLLM produced incoherent generations, so generation ran
in **separate vLLM containers** talking HTTP to the trainer while the trainer used
the HF rollout. They also hardcode that machine's paths (`/DATA/tracy/...`).

They are kept because the comments in them record measurements that are still the
justification for choices elsewhere in the repo — the 6.3x sidecar speedup, the
`position_ids` rotary-drift fix (parser-valid responses ~13% -> 100%), the reverted
length penalty, the arm guard added after a GRPO run silently trained as CCPO.

**They are not the way to run anything now.** Current runs go through
`scripts/exp_run.py`, which uses in-process vLLM with `VLLM_ATTENTION_BACKEND=TRITON_ATTN`
and writes a self-contained experiment directory. `scripts/setup_env.sh` builds the
environment from scratch.
