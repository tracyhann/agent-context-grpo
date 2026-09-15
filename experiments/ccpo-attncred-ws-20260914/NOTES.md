# ccpo-attncred-ws-20260914

**Question.** Does the attncred estimator (phi-attention readout + credibility prior +
task fallback) do anything on WebShop, where the published protocol differs from ALFWorld
and the state space is pages rather than rooms? On ALFWorld it was a null (H-AL).

**Arm.** `--set env_name=Webshop` plus the attncred flags: `ccpo_lam_fix=1.0`,
`ccpo_prior_kappa=2.0`, `ccpo_backoff_task=1`, `ccpo_jweight_c=0.0`,
`ccpo_phi=hidden+ctx`, `ccpo_gate=hard`, `ccpo_target=return`, `ccpo_rho=0.0`,
`ccpo_edge_w=1.0`, `ccpo_tau=0.15`, `ccpo_wmode=soft`, FLASH_ATTN, seed 0, **150 steps**,
`keep_ckpts=1`, early stopping OFF (`early_stop_patience=0`).

150 steps is GiGPO's published WebShop horizon (`run_webshop.sh: total_epochs=150`);
G2PO's script stops at 100, so step 100 is also worth reading as a comparison point.
Early stopping is off so the run reaches 150 whatever the middle looks like -- a
benchmark run, not a hypothesis test. No `pin_steps`: the volume had 99 GB free at
launch with another arm checkpointing beside this one, and a pinned copy would have put
the worst case (rolling + in-flight save + best + pin) over the edge. The 0913 run's
disk-full crash came from exactly that arithmetic.

Everything else comes from `WEBSHOP_PROTOCOL` in `scripts/exp_run.py`, which is
G2PO's `run_webshop.sh` verbatim: prompt 4096, val batch 128, 15 turns, `mean_norm`,
mini-batch 64, 0.05 CPU per env worker, group 8, train batch 16, lr 1e-6, KL 0.01
low-var, gamma 0.95, invalid-action penalty 0.1, val temperature 0.4 with sampling,
1,000-product catalogue. `tests/test_webshop_protocol.py` asserts all 37 of those
overrides against the reference script.

**Resource findings from launching this (three failed attempts, all distinct).**
1. `.venv-webshop` had no `datasets` -- the venv had run the env but never the trainer.
   `datasets 2.14.4` (what pip picks if fsspec is held) is broken against pyarrow 25
   (`pa.PyExtensionType` is gone); `datasets 5.0.1` works and moves only fsspec
   2026.7->2026.6, requests 2.27->2.34, tqdm 4.64->4.70. numpy/torch/vllm/transformers
   pins are untouched. A pre-install freeze is in the session scratchpad.
2. **The binding constraint here is pids.max = 20,000, not RAM.** RAM peaked at 155 GiB
   of 256. WebShop runs one JVM per env worker and the JVM sizes GC/JIT pools from the
   96 visible cores: 88.3 pids per worker, 26,152 projected for 256 workers. Fixed in
   `exp_run.py` with `JAVA_TOOL_OPTIONS=-XX:ActiveProcessorCount=1 -XX:+UseSerialGC`,
   measured at 18.6 per worker on a 16-worker probe.
3. That was not enough: with the JVM fixed, each **Ray actor** still carries ~65 threads
   (16 `event_engine` + 16 `nexting_thread` + gRPC pollers, all sized from the core
   count), so 256 actors alone cost 16,640 and the run reached 19,820 before step 1.
   Hence `val_batch_size=64` and `ray_num_cpus=48`: 192 actors, projected ~16,000.
4. `.venv-webshop` had **no flash-attn**, so `exp_run.py` fell back to
   `docker/fa_stub` (built for Blackwell, where no FA2 exists) and the first
   `compute_log_prob` raised `flash_attn stub: FA kernels are unavailable`. The stub is
   appended to PYTHONPATH only when the venv lacks a real build, and PYTHONPATH precedes
   site-packages -- so the fix is to install the real thing, not to edit the path.
   `flash_attn 2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310` from the upstream release
   page matches torch 2.6.0+cu124 / cxx11abi False / py3.10, and its varlen kernel and
   `unpad_input` were verified on GPU 4 (sm_80) before relaunching. `_have_flash_attn`
   now returns True for this venv and the stub is no longer on the path, so
   `use_remove_padding=True` (the reference setting) and dynamic-bsz packing both hold --
   no sdpa fallback, no deviation.

Deviations, all hardware and all recorded in the config's `reference_protocol.deltas`:
tp=1 not 2, `gpu_memory_utilization` 0.25 not 0.6 (they hold 8 cards, we train and
generate on 2), `use_dynamic_bsz=True` with this box's token budgets in place of their
fixed micro-batches, checkpointing on (`save_freq=5`) where they use -1, and
`val_batch_size=64` not 128 -- **the same 256-row validation set, evaluated in 4 chunks
instead of 2**, forced by the pid ceiling in point 3 above. Evaluation episodes, split,
temperature and sampling are unchanged, so held-out numbers stay comparable; only eval
wall-clock rises.

**Baselines at this protocol (Qwen2.5-1.5B, 1k catalogue).** GiGPO 67.4% at 150 steps. G2PO reports
WebShop in the same table; take its number from the paper before reading this arm.
No CCPO arm has ever run on WebShop, so the FIRST run is a baseline for us, not a test
of attncred -- the paired control (`ret-hard` on WebShop) does not exist yet.

**Watch.** `plots/progress.png`, held-out score every 5 steps.
- `ccpo/live_frac` and `ccpo/bucket_singleton_frac` in the first 5 steps: WebShop pages
  are long strings and exact-state grouping may be far sparser than ALFWorld's. If
  `live_frac` sits near zero the gate is degenerate here and the arm needs
  `ccpo_sim > 0` (as the search benchmark does), not a longer run.
- `ccpo/effect_rel` should be non-zero, as it was on ALFWorld (0.24-0.35). Zero means
  the readout is inert.
- `ccpo/phi_rel_corr`: on ALFWorld it rose 0.08 -> 0.21. The context half of
  `phi=hidden+ctx` (`t`, `n_unique`, `revisit`, `progress`) was selected on ALFWorld
  prose; it is computable on any env but its usefulness here is untested. If the arm
  disappoints, `ccpo_phi=hidden` is the cleaner second run.

**Result.** _(fill in with the step each number came from)_

**Reading.** _(including the negative case)_
