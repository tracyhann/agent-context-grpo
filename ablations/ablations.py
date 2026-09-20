#!/usr/bin/env python3
"""The CCPO-ATTNCRED ablations, per experiments/experiments.md.

Each uses its declared base method (normally `attncred`) on the SAME benchmark,
selected backbone (1.5B by default), 150 steps. The default control for an
ALFWorld ablation is
`ccpo-attncred-alfworld-1.5b`; for a WebShop one it is `ccpo-attncred-ws-1.5b`,
which carries the dense-score target -- so an ablation inherits it and stays paired
with the arm it ablates.

Each entry names the component, the delta, and what the result decides. An entry may
declare `benchmarks` when it is only meaningful on one of them -- by default an ablation
runs on both.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "ccpo"))
import arms                                                   # noqa: E402

ABLATIONS = {
    "future-progress-h2-no-credit-shrinkage-history-only": dict(
        name="CCPO-ATTNCRED-TWO-STEP-NOSHRINK-HISTORY-ONLY",
        tag="fph2-noshrink-history-only",
        title="M10 H2 no shrinkage, history-only, episode weight 0",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("alfworld",),
        delta={'ccpo_lk_fix': 1.0, 'ccpo_progress_weight': 0.0},
        removes="future-progress credit from the actor update and its normalization support",
        asks="how much does history credit contribute alone under the matched no-shrink protocol?",
        watch="History-2 prompts and hidden+context features stay enabled. Raw history/future "
              "remain diagnostic; the disabled channel has exactly zero applied advantage. "
              "Episode weight 0; original edge weight 0; usable context baselines at full strength.",
    ),
    "future-progress-h2-no-credit-shrinkage-future-only": dict(
        name="CCPO-ATTNCRED-TWO-STEP-NOSHRINK-FUTURE-ONLY",
        tag="fph2-noshrink-future-only",
        title="M10 H2 no shrinkage, future-only, episode weight 0",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("alfworld",),
        delta={'ccpo_lk_fix': 1.0, 'ccpo_progress_history_weight': 0.0},
        removes="historical-residual credit from the actor update and its normalization support",
        asks="how much does future credit contribute alone under the matched no-shrink protocol?",
        watch="History-2 prompts and hidden+context features stay enabled. Raw history/future "
              "remain diagnostic; the disabled channel has exactly zero applied advantage. "
              "Episode weight 0; original edge weight 0; usable context baselines at full strength.",
    ),
    "future-progress-h2-no-credit-shrinkage-history-only-active-episode": dict(
        name="CCPO-ATTNCRED-TWO-STEP-NOSHRINK-HISTORY-ONLY-EPISODE",
        tag="fph2-noshrink-history-only-ep",
        title="M11 H2 no shrinkage, history-only, episode weight 1",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("webshop",),
        delta={'ccpo_lk_fix': 1.0, 'ccpo_progress_weight': 0.0, 'ccpo_ep_w': 1.0},
        removes="future-progress credit from the actor update and its normalization support",
        asks="how much does history credit contribute alone under the matched no-shrink protocol?",
        watch="History-2 prompts and hidden+context features stay enabled. Raw history/future "
              "remain diagnostic; the disabled channel has exactly zero applied advantage. "
              "Episode weight 1; original edge weight 0; usable context baselines at full strength.",
    ),
    "future-progress-h2-no-credit-shrinkage-future-only-active-episode": dict(
        name="CCPO-ATTNCRED-TWO-STEP-NOSHRINK-FUTURE-ONLY-EPISODE",
        tag="fph2-noshrink-future-only-ep",
        title="M11 H2 no shrinkage, future-only, episode weight 1",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("webshop",),
        delta={'ccpo_lk_fix': 1.0, 'ccpo_progress_history_weight': 0.0, 'ccpo_ep_w': 1.0},
        removes="historical-residual credit from the actor update and its normalization support",
        asks="how much does future credit contribute alone under the matched no-shrink protocol?",
        watch="History-2 prompts and hidden+context features stay enabled. Raw history/future "
              "remain diagnostic; the disabled channel has exactly zero applied advantage. "
              "Episode weight 1; original edge weight 0; usable context baselines at full strength.",
    ),
    # M10 H2: the shared readout applies the delta to history and both potentials.
    "future-progress-h2-no-context-vector": dict(
        name="CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOCTX",
        tag="fph2-noctx",
        title="M10 H2 using hidden-state features without context statistics",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("alfworld",),
        delta={"ccpo_ctx_w": 0.0},
        removes="the accumulated context-statistics block from similarity features",
        # CTX_W=0 skips the statistics block for history and both potentials.
        # It is equivalent to phi=hidden, which future progress also accepts.
        # Preserve this prepared arm's one-setting delta. M10-NOCTX is H1.
        asks="does the context-statistics vector improve two-step future progress beyond the frozen hidden state?",
        watch="current_phi and nonterminal future_phi have 1536 dimensions on 1.5B; "
              "context-stat perturbations cannot change credit. Kappa=2, support "
              "shrinkage and H2 endpoint grouping are unchanged.",
    ),
    "future-progress-h2-no-credit-shrinkage-no-context-vector": dict(
        name="CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-NOCTX",
        tag="fph2-noshrink-noctx",
        title="M10 H2 with full-strength hidden-only baselines",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("alfworld",),
        delta={"ccpo_lk_fix": 1.0, "ccpo_ctx_w": 0.0},
        removes="both support-based task-prior shrinkage and the context-statistics block",
        # Compose the two existing H2 interventions; frozen hidden features and
        # prompt history remain. CTX_W=0 skips statistics for all three readouts.
        asks="how do credit shrinkage and accumulated context statistics interact in M10 H2?",
        watch="history/current/future usable baselines equal their hidden-only kernel "
              "estimates at lambda_k=1; 1.5B phi has 1536 dimensions. Context-stat "
              "perturbations and kappa cannot change credit; fallback and terminal "
              "conventions remain unchanged.",
    ),
    "future-progress-h2-kappa4": dict(
        name="CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-KAPPAFOUR",
        tag="fph2-kappa4",
        title="M10/M11 H2 with stronger support-based shrinkage (kappa=4)",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("alfworld", "webshop"),
        delta={"ccpo_prior_kappa": 4.0},
        removes="less of the task prior at a given peer support",
        asks="does stronger regularization of sparsely supported context baselines improve H2 future progress?",
        watch="history/current/future lambda_k is J/(J+4) on supported exact groups; "
              "task fallback and terminal endpoint conventions are unchanged.",
    ),
    "future-progress-history1-future1": dict(
        name="CCPO-ATTNCRED-FUTURE-PROGRESS-HISTORYONE-FUTUREONE-WS",
        tag="hist1-fut1",
        title="M11 with one prompt-history turn and one-step future progress",
        base_method="attncred-context-future-progress",
        benchmarks=("webshop",),
        delta={"history_length": 1},
        removes="the second previous observation-action pair from the prompt",
        # The actor and frozen reference consume the same shortened prompt.
        # This does not truncate accumulated context statistics or return targets.
        asks="does a one-turn history with H1 future progress improve M11 over two-turn history?",
        watch="env.history_length=1 reaches WebShop memory.fetch; future horizon=1, "
              "kappa=2, context statistics, whole-trajectory exclusion and fusion "
              "weights remain the M11 defaults. This is not a strict local-only estimator.",
    ),
    "future-progress-history1-future1-no-credit-shrinkage": dict(
        name="CCPO-ATTNCRED-FUTURE-PROGRESS-HISTORYONE-FUTUREONE-NOSHRINK",
        tag="hist1-fut1-noshrink",
        title="M10/M11 with one history turn, one-step future progress and full-strength baselines",
        base_method="attncred-context-future-progress",
        benchmarks=("alfworld", "webshop"),
        delta={"history_length": 1, "ccpo_lk_fix": 1.0},
        removes="the second prompt-history turn and support-based task-prior shrinkage",
        asks="does a one-history/one-future window improve no-shrink context credit on ALFWorld and WebShop?",
        watch="actor and frozen reference receive the previous observation-action pair "
              "plus the current observation; future horizon=1. Accumulated context statistics "
              "and soft exponential weights remain enabled. Usable lambda_k=1, "
              "episode/original-edge weights=0; episode advantage is diagnostic only.",
    ),
    "future-progress-h2-no-credit-shrinkage": dict(
        name="CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK",
        tag="fph2-noshrink",
        title="M10/M11 H2 with full-strength usable context baselines",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("alfworld", "webshop"),
        delta={"ccpo_lk_fix": 1.0},
        removes="support-based shrinkage toward the task prior",
        # lam_u is already 1. Pinning lam_k=1 gives baseline=b_loo; kappa=2
        # remains recorded but cannot affect this weight. Retain prior diagnostics
        # and the existing task-bucket fallback when no exact peer is available.
        asks="does an available context baseline work best at full strength even with only one peer?",
        watch="history/current/future lambda_k is 1 on usable readouts; baseline equals "
              "its kernel estimate. No exact peer still uses the existing task bucket; "
              "no cross-trajectory peer anywhere still provides no usable estimate.",
    ),
    "future-progress-h2-no-credit-shrinkage-active-episode": dict(
        name="CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-ACTIVE-EPISODE",
        tag="fph2-noshrink-ep",
        title="M10/M11 H2 with full-strength baselines and active episode advantage",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("alfworld", "webshop"),
        delta={"ccpo_lk_fix": 1.0, "ccpo_ep_w": 1.0},
        removes="support-based task-prior shrinkage and the exclusion of episode credit from the actor update",
        asks="does restoring the episode channel stabilize or improve full-strength H2 context/future credit?",
        watch="history/current/future usable baselines equal their kernel estimates at lambda_k=1. "
              "Episode weight is 1; applied history + future + episode must equal the actual "
              "masked actor advantage. Context statistics remain enabled; original edge weight stays 0.",
    ),
    "future-progress-h2-no-credit-shrinkage-cosine": dict(
        name="CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-COS",
        tag="fph2-noshrink-cos",
        title="M10/M11 H2 with full-strength cosine context baselines",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("alfworld", "webshop"),
        delta={"ccpo_lk_fix": 1.0, "ccpo_wmode": "cos"},
        removes="support-based task-prior shrinkage and the exponential similarity kernel",
        asks="does clipped cosine weighting of the same context representations improve full-strength H2 credit?",
        watch="history and both potential readouts use max(cos(phi_i,phi_j),0), "
              "with the existing nearest-peer fallback when all weights vanish. "
              "Whitening, context statistics, exact observation groups and whole-trajectory "
              "exclusion stay unchanged; usable lambda_k=1 and episode/original-edge weights=0.",
    ),
    "future-progress-h2-no-credit-shrinkage-active-episode-cosine": dict(
        name="CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-ACTIVE-EPISODE-COS",
        tag="fph2-noshrink-ep-cos",
        title="M10/M11 H2 with full-strength cosine baselines and active episode advantage",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("alfworld", "webshop"),
        delta={"ccpo_lk_fix": 1.0, "ccpo_ep_w": 1.0, "ccpo_wmode": "cos"},
        removes="support-based task-prior shrinkage, the exponential kernel and episode-credit exclusion",
        asks="does active episode supervision improve H2 credit with full-strength cosine context baselines?",
        watch="only episode weight changes from the cosine no-shrink arm. "
              "A=E+N_CC(H+F); N_CC is task standardization on ALFWorld and identity "
              "on WebShop. Context statistics stay enabled, original-edge weight stays 0, "
              "and there is no normalization after episode fusion.",
    ),
    "future-progress-h2-no-credit-shrinkage-30turn": dict(
        name="CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-THIRTYTURN-WS",
        tag="fph2-noshrink-30turn",
        title="M11 H2 full-strength context baselines with a 30-turn WebShop budget",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("webshop",),
        delta={"ccpo_lk_fix": 1.0, "max_steps": 30},
        removes="support-based task-prior shrinkage and the 15-turn episode cutoff",
        asks="does a 30-turn train/eval budget improve M11 H2 without credit shrinkage?",
        watch="env.max_steps=30 applies to training and validation; history and future "
              "horizon remain 2. Context statistics stay enabled, usable lambda_k=1, "
              "and episode/original-edge weights stay 0. Pair with the 15-turn no-shrink arm.",
    ),
    "future-progress-h2-no-credit-shrinkage-active-episode-30turn": dict(
        name="CCPO-ATTNCRED-FUTURE-PROGRESS-TWO-STEP-NOSHRINK-ACTIVE-EPISODE-THIRTYTURN-WS",
        tag="fph2-noshrink-ep-30turn",
        title="M11 H2 full-strength baselines and episode credit with a 30-turn WebShop budget",
        base_method="attncred-context-future-progress-h2",
        benchmarks=("webshop",),
        delta={"ccpo_lk_fix": 1.0, "ccpo_ep_w": 1.0, "max_steps": 30},
        removes="support-based task-prior shrinkage, episode-credit exclusion and the 15-turn cutoff",
        asks="does episode supervision improve full-strength H2 credit under a 30-turn budget?",
        watch="only episode weight changes relative to the 30-turn no-shrink arm; "
              "A=E+H+F under WebShop mean_norm. Both train/eval use 30 turns, while "
              "history/future stay 2/2 and original edge stays 0. No final sum normalization.",
    ),
    # -----------------------------------------------------------------------
    "hard-gate": dict(
        name="CCPO-ATTNCRED-HARDGATE",
        tag="hardgate",
        title="Binary hard gating of within-group comparisons",
        delta={"ccpo_wmode": "hard"},
        removes="the soft kernel",
        # b_loo = Sum w*TGT / Sum w with w = 1[d <= tau] instead of exp(-d/tau).
        # With 0/1 weights that is the plain average of the survivors. Same tau, so
        # both modes select the same neighbourhood and differ only in how they weight
        # inside it. If tau admits nobody the nearest trajectory is kept, so no row
        # loses its baseline.
        asks="does the gain come from ORDERING neighbours by phi-distance, or only "
             "from RESTRICTING which ones count?",
        watch="ccpo/E_w becomes the survival fraction under a 0/1 weight. On ALFWorld "
              "it measured 0.011, i.e. ~1 neighbour in 90 clears tau=0.15*median -- "
              "at that rate the arm is close to a nearest-trajectory baseline via the "
              "fallback, and effect_rel rose to 0.70 against soft's 0.28.",
    ),
    # -----------------------------------------------------------------------
    "no-task-baseline": dict(
        name="CCPO-ATTNCRED-NOTASK",
        tag="notask",
        title="Without task baseline fallback",
        delta={"ccpo_prior_kappa": 0.0},
        removes="the credibility prior b_task",
        # base = lam*b_loo + (1-lam)*b_obs with lam pinned to 1 -> base = b_loo, and
        # with kappa=0 the (J*b_loo + kappa*b_task)/(J+kappa) blend never fires:
        # B_TASK is not even computed. Context b_loo only, exactly as the spec words it.
        asks="does leaning a thinly supported node on the task mean buy anything, or "
             "is the attention readout enough on its own?",
        watch="ccpo/lam_k_mean goes to 1.000 (it is the realised blend weight). "
              "ccpo_backoff_task stays 1, so rows whose node has no sibling are still "
              "credited against the task BUCKET at level 1 -- a different mechanism "
              "from b_task, and the one that keeps live_frac at 1.0. To ablate that "
              "instead, add --set ccpo_backoff_task=0 and expect live_frac to drop "
              "(~8% of rows on ALFWorld, ~20% on WebShop).",
    ),
    # -----------------------------------------------------------------------
    "even-blend": dict(
        name="CCPO-ATTNCRED-EVENBLEND",
        tag="evenblend",
        title="Without evidential-support based credit shrinkage",
        delta={"ccpo_lk_fix": 0.5},
        removes="the support weighting of the prior, not the prior",
        # lam_k = J/(J+kappa) becomes a constant 0.5: base = 0.5*b_loo + 0.5*b_task on
        # every level-0 occurrence, whatever its sibling support. kappa no longer enters
        # the weight -- it only decides that B_TASK is computed, and ccpo_lk_fix forces
        # that on its own, so this is a single-key delta.
        asks="is the Buhlmann credibility form doing the work, or merely the presence "
             "of a task prior at some fixed mixing ratio?",
        watch="ccpo/lam_k_mean pins to 0.500 (attncred runs it at J/(J+2), which moves "
              "with bucket support). Paired against no-task-baseline this separates "
              "'prior exists' from 'prior is weighted by evidence'.",
    ),
    # -----------------------------------------------------------------------
    "no-context-vector": dict(
        name="CCPO-ATTNCRED-NOCTX",
        tag="noctx",
        title="Without context summary vector",
        delta={"ccpo_phi": "hidden"},
        removes="the trajectory-context block of phi",
        # phi is then the whitened last-prompt-token hidden state alone. The prompt
        # carries step_count plus the most recent history_length (2) turns, so it can
        # separate "step 5 from step 15" but not "has this agent already searched here
        # twice" -- n_unique, revisit and progress summarise the whole episode and
        # appear nowhere in the prompt.
        asks="does the whole-episode context earn its place, or does the hidden state "
             "of the recent obs-action pairs already carry it?",
        watch="ccpo/phi_rel_corr. On ALFWorld it rose 0.01 -> 0.24 over training with "
              "the context block in; on WebShop it stayed near 0.02, so this ablation "
              "may be near-free there and expensive on ALFWorld.",
    ),
    # -----------------------------------------------------------------------
    "cosine": dict(
        name="CCPO-ATTNCRED-COS",
        tag="cos",
        title="Cosine similarity rather than the attention formulation",
        delta={"ccpo_wmode": "cos"},
        removes="the exponential kernel",
        # w = max(cos(phi_i, phi_b), 0) with no tau at all: weight falls off linearly
        # in the angle instead of exponentially in the distance, so distant neighbours
        # keep far more weight. Negative cosines clip to 0 (a weighted mean needs
        # non-negative weights); if every neighbour clips away the most similar one is
        # kept, as under hard gating.
        asks="does the exp(-d/tau) kernel earn its place over a plain similarity score?",
        watch="ccpo/E_w rises sharply -- cosine over whitened unit vectors sits near 0 "
              "for unrelated states, against exp(-d/tau) which is ~0.008 on average. "
              "If effect_rel collapses, the kernel's sharpness was the mechanism.",
    ),
    # -----------------------------------------------------------------------
    "no-edge": dict(
        name="CCPO-ATTNCRED-NOEDGE",
        tag="noedge",
        title="Without the edge advantage",
        delta={"ccpo_edge_w": 0.0},
        removes="G2PO's value-gain term",
        # A_CC keeps only the node term, TGT_i - base_i; the second half of (S),
        # z_task(V(next_i) - V(cur_i)), is dropped. V is G2PO's group-aggregated node
        # value, so the edge term is the part that scores MOVEMENT toward the goal --
        # where the action took the agent -- rather than the value of where it stood.
        #
        # NOTE this is `ccpo_edge_w`, not `ccpo_target=nextnode`. The two both involve
        # the successor node and are different knobs: edge_w weights the value-GAIN
        # term added to the step credit, while target=nextnode would change what the
        # node term PREDICTS from the return-to-go to V(next(u)). This ablation leaves
        # the target alone.
        asks="how much of the step credit's effect is the context-conditioned baseline, "
             "and how much is the edge term it is summed with?",
        watch="ccpo/edge_cov drops to 0 and ccpo/adv_cc_absmean falls -- the edge term is "
              "standardised per task, so it contributes ~1 unit of scale that the node "
              "term no longer has beside it. On ALFWorld A_CC is standardised again per "
              "task afterwards, so the arm mostly reweights; on WebShop (mean_norm) "
              "nothing rescales it, and the step channel gets absolutely smaller.",
    ),
    # -----------------------------------------------------------------------
    "no-edge-return-ws": dict(
        name="CCPO-ATTNCRED-NOEDGE-RETURN-WS",
        tag="noedgeret",
        title="Without the edge advantage, on the binary return target (WebShop)",
        delta={"ccpo_edge_w": 0.0, "ccpo_target": "return"},
        benchmarks=("webshop",),
        removes="G2PO's value-gain term AND the dense-score adaptation",
        # A = A_EP + Z(TGT_i - base_i) with TGT the gamma-discounted return-to-go of
        # WebShop's binary 10/0 reward -- the published target. Two keys, deliberately:
        # this is the arm that shows what the method does on WebShop with NOTHING
        # borrowed, neither G2PO's edge term nor the dense score this project added.
        # The episode term stays (A_CC + A_EP), so it is still the standard shape.
        #
        # WebShop only. On ALFWorld ccpo_target is already `return`, so the delta would
        # collapse to plain A6 and the run would be a duplicate.
        asks="on WebShop, with the published binary reward and no edge term, does the "
             "context-conditioned baseline carry anything on its own?",
        watch="the zero-advantage population is the risk: under the binary reward 30-69% "
              "of task groups score zero on EVERY rollout, and a group-relative estimator "
              "computes exactly zero advantage there -- which is why the dense target "
              "exists. Read ccpo/effect_rel and ccpo/live_frac beside the success curve; "
              "a flat curve here with effect_rel > 0 says the target, not the estimator, "
              "was the binding constraint.",
    ),
}


def benchmarks_for(ablation):
    """The benchmarks an ablation is defined on. Both, unless it says otherwise."""
    return tuple(ABLATIONS[ablation].get("benchmarks", ("alfworld", "webshop")))


def build(ablation, benchmark, extra=None, backbone="1.5b"):
    """Build a benchmark/backbone pair; preserve the legacy third-argument overrides."""
    if ablation not in ABLATIONS:
        raise KeyError(f"unknown ablation {ablation!r}; known: {', '.join(ABLATIONS)}")
    spec = ABLATIONS[ablation]
    allowed = benchmarks_for(ablation)
    if benchmark not in allowed:
        raise KeyError(f"{ablation!r} is defined for {', '.join(allowed)} only, "
                       f"not {benchmark!r}")
    _, cfg = arms.build(spec.get("base_method", "attncred"), benchmark, backbone)
    cfg.update(spec["delta"])
    cfg.update(extra or {})
    bench_tag = "alfworld" if benchmark == "alfworld" else "ws"
    return f"ccpo-attncred-abl-{spec['tag']}-{bench_tag}-{backbone}", cfg


def control_name(benchmark, ablation=None, backbone="1.5b"):
    method = ABLATIONS[ablation].get("base_method", "attncred") if ablation else "attncred"
    name, _ = arms.build(method, benchmark, backbone)
    return name
