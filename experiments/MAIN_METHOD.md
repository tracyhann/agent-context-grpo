# Current main method: H2, no credit shrinkage, no episode advantage

**Designated by the user on 2026-09-20.** The primary method for ALFWorld (M10)
and WebShop (M11) is the existing `future-progress-h2-no-credit-shrinkage` arm.
Its registry key, experiment IDs and training configurations are retained.
The `ablations/` location records its implementation history; this arm is now
the main method for reporting and designing paired comparisons.

| Setting | Main method |
|---|---|
| Prompt-history turns / future-progress horizon | 2 / 2 |
| Credit shrinkage | off: usable `ccpo_lk_fix=1` |
| Kernel-versus-uniform mixture | `ccpo_lam_fix=1` |
| Episode advantage coefficient | `ccpo_ep_w=0` |
| Original-edge coefficient | `ccpo_edge_w=0` |
| History / future coefficients | 1 / 1 |
| Context representation | processed frozen hidden state + accumulated context statistics |
| Context-statistics weight / PCA directions removed | 1 / 3 |
| Similarity weighting | soft exponential kernel, `ccpo_wmode=soft`, tau scale 0.15 |
| Grouping | exact task/observation groups; whole query trajectory excluded |
| Targets | penalized discounted history return; unpenalized potential return |
| Current experiment protocol | Qwen2.5-1.5B-Instruct, 150 training steps, seed 0, 8 rollouts/task |
| Prepared resources | two GPUs per experiment |
| Train/eval turn ceilings | ALFWorld 50; WebShop 15 |

## Method

Let C_s[q] be the existing kernel-weighted, whole-trajectory-excluded contextual
readout of target q, using the frozen history-plus-statistics representation.
All usable history/current/future baselines use B_s[q]=C_s[q] at lambda_k=1,
including one-peer groups. The recorded kappa=2 has no shrinkage effect under
this fixed weight. Existing task-bucket fallback and unsupported-row handling
remain in place. Terminal potential is fixed at 10 for success and 0 otherwise.

\[
H_{i,t}=Y_{i,t}-C_{i,t}[Y],\qquad
Z_{i,s}=\gamma^{T_i-s}R_i,\quad V_{i,s}=C_{i,s}[Z],\quad\gamma=0.95,
\]
\[
F^{H2}_{i,t}=z_{\mathrm{task}}(V_{i,\min(t+2,T_i)}-V_{i,t}),\qquad
A_{i,t,\ell}=M_{i,t,\ell}N_{\mathrm{CC}}(H+F^{H2})_{i,t}.
\]
Y includes the existing invalid-action penalty; Z is the unpenalized potential
label. Current/future values use their respective observation groups. N_CC is
task standardization on ALFWorld and identity on WebShop; F is standardized per
task in both. Features and advantage estimates stay detached. KL remains a
separate loss.

**Episode reward still defines the return and potential targets.** Only the
separate episode-advantage channel is disabled. Raw episode advantage and the
original edge remain available as diagnostics with zero applied actor weight.

## Existing implementations

Registry: [ablations.py](../ablations/ablations.py), key
`future-progress-h2-no-credit-shrinkage`.
Variant: `CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK`.

| Benchmark | Canonical ID | Existing experiment |
|---|---|---|
| M10 ALFWorld | `ccpo-attncred-abl-fph2-noshrink-alfworld-1.5b` | [Method notes](m10-h2-no-credit-shrinkage-alfworld-1.5b-2gpu-20260919/NOTES.md) |
| M11 WebShop | `ccpo-attncred-abl-fph2-noshrink-ws-1.5b` | [Method notes](m11-h2-no-credit-shrinkage-webshop-1.5b-2gpu-20260919/NOTES.md) |

This designation records the chosen method; it is not a run launch or a claim
of completed results. Existing preparation/validation records remain the source
for the configurations' execution provenance.

## Comparisons relative to this main method

| Variant | Change from the main method |
|---|---|
| H2 with credit shrinkage | restore derived lambda_k=J/(J+2) |
| H2 no shrinkage + episode advantage | episode weight 0 -> 1 |
| H2 no shrinkage + cosine similarity | soft exponential -> clipped cosine weights |
| H2 no shrinkage + cosine + episode | change both similarity and episode weight |
| History 1 / future 1, no shrinkage, no episode | history/future windows 2/2 -> 1/1 |
| H2 no shrinkage without context statistics | context-stat weight 1 -> 0 |
| WebShop 30-turn version | train/eval ceiling 15 -> 30 |

M3/M5, original H1 M10/M11 and earlier shrinkage arms remain historical
comparators. Historical result labels retain their original settings; they
must not be relabelled as this main method. Existing registry parents/deltas
still describe how each configuration was built; the table above describes
scientific comparisons against the selected main method.

## 7B scale-up preparation

The main H2/no-shrink/no-episode method and its active-episode comparison are
now prepared for **Qwen2.5-7B-Instruct, eight GPUs, 150 steps, seed 0**, on both
benchmarks. See [the four-run plan](MAIN_METHOD_7B.md) for exact configs, resource
changes, reproduction commands and validation. The original 1.5B experiments
above retain their recorded protocol. No 7B run is launched or queued.

## History/future component comparisons

[Four 1.5B component ablations](HISTORY_FUTURE_ABLATIONS.md) retain H2, no shrinkage
and context statistics, disabling either the historical-residual channel or
future-progress channel. ALFWorld uses EP0; WebShop uses its EP1 comparison.
These are prepared comparisons; the main method's 1/1 history/future fusion
remains unchanged.


## No-LOO comparison

[Two 1.5B no-LOO ablations](NO_LOO_ABLATIONS.md) retain H2, no shrinkage, EP0 on
both benchmarks, and history/future weights 1/1. They allow the query occurrence
and matching same-trajectory turns into all contextual readouts. The main method
above continues to exclude the whole query trajectory by default (`ccpo_loo=1`).


## WebShop EP0 component comparisons

[Three 1.5B WebShop ablations](WEBSHOP_EP0_ABLATIONS.md) now use the main method's
EP0 protocol: remove future-progress credit, historical-residual credit, or the
context-statistics representation block. Each is a separate one-setting change
from H2 no-shrink EP0; earlier WebShop EP1 component comparisons remain available.


## ALFWorld window allocation comparisons

[Five 1.5B no-shrink EP0 variants](ALFWORLD_WINDOW_ABLATIONS.md) replace the main
2/2 history/future windows with 4/0, 0/4, 3/1, 1/3 and 3/3. Zero windows disable
that actor channel; history 0 also removes prior prompt turns while retaining
the task goal. Context statistics and whole-trajectory LOO remain enabled.
These are prepared ablations; the designated main method remains H2.


## WebShop active-episode peer and representation comparisons

[Three EP1 ablations](WEBSHOP_EP1_ABLATIONS.md) independently allow same-trajectory
peers, remove context statistics, or replace soft weights with clipped cosine.
Each retains H2/no shrinkage and active episode weight 1 on the same 1.5B,
150-step, 15-turn WebShop protocol. They compare against the EP1 control; the
primary method designation above remains EP0. All are prepared only.
