# M5 noctx ablation — WebShop, Qwen2.5-1.5B

User-authorized on 2026-09-16. Sequential order: noedge, then noctx.
Control: `m5-ccpo-attncred-ctxadv-ret-ws-1.5b-2gpu-20260916`.

Trajectory-context vector removed; hidden-state features retained and M5 edge weight restored to 1.

Both runs start independently from the base Qwen2.5-1.5B-Instruct weights, seed 0.
Binary reward target `ccpo_target=return`; episode advantage `ccpo_ep_w=0`.
Credit shrinkage is unchanged: kappa=2, support-derived lambda_k=J/(J+2),
`ccpo_lk_fix` empty, `ccpo_lam_fix=1`, `ccpo_backoff_task=1`.
The fixed phi-readout weight does not disable the separate credibility shrinkage.
The no-context run is a single ablation of M5, not a cumulative noedge+noctx run.

150 optimizer steps; evaluate/save every 5; retain best, step 100, and final.
Two A100 GPUs 2,3. All learning, training/evaluation batch sizes, horizon,
reward penalties, and sampling settings match M5.

Runtime-only adjustment: Ray scheduling capacity 24 instead of 64 CPUs, sufficient
for 128 training + 128 validation actors at 0.05 CPU each. Current launcher disables
idle worker prestarts and bounds GCS thread pools so the run can coexist with M3.
Private Ray and compilation-cache paths are generated per experiment.

The durable controller is `.local/m5-ablation-chain-20260916/runner.py`.
It waits for GPUs 2,3 to be idle; the second run starts only after successful
150-step completion, final validation, and retained checkpoint checks for the first.
A failed run stops the chain. See controller `state.json` and `chain.log`.

Configuration deltas are recorded in `config-diff-from-m5.json`.
