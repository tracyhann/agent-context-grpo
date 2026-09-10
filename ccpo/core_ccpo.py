# Context-Conditioned Predictive Grouping (CCPO) advantage estimator.
#
# Drops into verl-agent beside gigpo/core_gigpo.py.  GiGPO groups a task's steps
# into HARD clusters of identical observations and normalises inside each; CCPO
# keeps the same observation gate but replaces the hard cluster with a WEIGHTED,
# cross-trajectory leave-one-out baseline whose weights come from trajectory
# context.
#
# Four corrections from the 08-26 diagnostics are baked in:
#   E1  probe/context comparison must not collapse as pi sharpens
#   E2  compare on a SHARED basis, not each occurrence's own top-M
#   E3  scale-free distances -- raw ones group by episode position, not credit
#   E6  cross-trajectory only: 21-47% of occurrences are same-trajectory revisits,
#       where G_v is downstream of a_u and the baseline stops being valid
#
# v1 uses a FROZEN context encoder (the plan's P1/P2 rung).  The learned
# successor-feature variant (P4) is a separate arm: Phase 1 produced no evidence
# that predictive similarity beats a null control, so the cheap rung is the
# scientifically correct thing to run first.
from collections import defaultdict
from difflib import SequenceMatcher
import hashlib
import os
import re

import numpy as np
import torch

# Shrinkage rule for lambda*. "eb" (default) is the positive-part empirical-Bayes
# rule on the realised baseline disagreement; "mse" reproduces the v6 rule.
# Validated offline in experiments/08-27/probe_shrinkage.py against 305k real
# training samples -- see results/probe_shrinkage.txt.
# "eb_hier" is the hierarchical form and the one to prefer: the signal variance is
# estimated once over the batch, the shrinkage is then per occurrence against its
# own noise. "eb_pooled" estimates the shrinkage weight ONCE per batch from the ensemble of
# realised disagreements instead of testing each occurrence against its own noise.
# The per-occurrence test has about two degrees of freedom -- a bucket holds a
# handful of trajectories -- so it demands |b_obs - b_LOO| > sqrt(1/n_eff - 1/J)/rho
# standard deviations before it fires, which a weighted vs unweighted mean of three
# numbers essentially never reaches: measured lambda = 0.000 on every occurrence of
# a real batch. Pooling is the standard empirical-Bayes move (James-Stein): with
# d = signal + noise, 1 - E[Var(d)]/E[d^2] is the fraction of the realised
# disagreement that is real, estimated over thousands of occurrences rather than
# over one bucket.
_SHRINK = os.environ.get("ACG_CCPO_SHRINK", "eb").lower()
# Gate mode. "hard" is the GiGPO/G2PO gate: a bucket is an exact (task,
# observation) match. "global" makes the whole task one bucket and lets the phi
# kernel decide neighbourhood softly, so the hard gate becomes the tau -> 0 limit
# rather than a separate mechanism. Measured offline on gate-probe-20260907
# (6,912 occurrences, both targets): LOO residual R2 0.4579 -> 0.4841 at
# tau_scale 0.15 on return-to-go, 0.7434 -> 0.7604 at 0.25 on nextnode.
_GATE = os.environ.get("ACG_CCPO_GATE", "hard").lower()
# Kernel width as a multiple of the bucket's median phi-distance. Only meaningful
# under the global gate, where it IS the gate: too tight rebuilds exact matching
# with worse statistics (R2 0.30 at 0.02), too loose converges on the uniform task
# mean. The offline optimum is an interior one, which is the whole argument for a
# soft global neighbourhood over a hard local one.
_TAU_ENV = os.environ.get("ACG_CCPO_TAU")
# Scale of the step credit. "task" divides by the per-task sd in the trainer (the
# shipped default, and the WORST of the three options measured). "local" divides by
# the phi-weighted sd over the SAME soft neighbourhood that produced the baseline --
# so the kernel decides membership and scale coherently instead of one of each.
# Measured split-half reliability (gate-probe-20260907, n=6,912): unstandardised
# 0.8299, task 0.8446, G2PO's per-node 0.8968, shuffled-sigma control 0.8860,
# phi-weighted local 0.9711.
_STD_MODE = os.environ.get("ACG_CCPO_STD", "task").lower()
# Floor on sigma. A neighbourhood whose targets are all identical gives sigma 0
# (measured p10 = 0.000), which would divide by ~nothing. Expressed as a fraction of
# the batch-level target sd so it carries no units.
_STD_FLOOR = float(os.environ.get("ACG_CCPO_STD_FLOOR", "0.25"))

# Affinity metric. "hidden" uses the frozen REFERENCE policy's last-prompt-token
# hidden state (supplied by the trainer as phi_feats), whitened batch-wide;
# "bow" is the original hashed bag-of-words FrozenPhi.
#
# Measured on a trained-policy corpus, conflation AUC (0.50 = blind, labels are
# admissible-command sets that no encoder sees):
#     bag-of-words phi          0.568
#     hidden, mean-pooled       0.505   <- pooling dilutes the discriminating line
#     hidden, last-token raw    0.659
#     hidden, last-token +PCA1  0.730
#     hidden, last-token +PCA2  0.771
#     hidden, last-token +PCA3  0.795   <- default
# Leakage control: with all action strings stripped from the context the same
# encoder still scores 0.655-0.688, so this is state inference, not the previous
# step's admissible set leaking through action names.
# "hidden+ctx" concatenates the whitened hidden state with the thermometer-coded
# trajectory context. This is what E4 actually asks for -- phi must be a function
# of BOTH observation and context -- and the plain "hidden" mode does not satisfy
# it. Inside a bucket the observation is constant by construction, and the prompt
# carries only step_count plus the most recent `history_length` (2) turns, so a
# hidden state can separate "step 5 vs step 15" but not "has this agent already
# searched here twice". n_unique, revisit and progress summarise the WHOLE episode
# and appear nowhere in the prompt unless compaction is on.
_PHI_MODE = os.environ.get("ACG_CCPO_PHI", "hidden").lower()

# Weight on the context block when it is concatenated onto the hidden state. Both
# blocks are L2-normalised first, so this is a genuine mixing ratio.
_CTX_W = float(os.environ.get("ACG_CCPO_CTX_W", "1.0"))

# Number of principal directions removed before distances are taken. Decoder-LM
# embeddings are anisotropic: the leading directions encode register ("this is
# ALFWorld prose"), not state. Unsupervised -- no labels enter.
_WHITEN_K = int(os.environ.get("ACG_CCPO_WHITEN", "3"))

# rho is the estimator's declared confidence in its distances and enters as
# B_eff = rho * bias_prior * sigma. AUC calibrates it: rho = 2*(AUC - 0.5), so a
# blind metric (AUC 0.5) gives rho = 0, lambda -> 0 and an exact fallback to the
# uniform G2PO baseline. It shipped hardcoded at 1.0 while the metric in use was
# at chance; 0.795 gives 0.59, which puts lambda ~ 0.26 at the observed var_gain.
_RHO = float(os.environ.get("ACG_CCPO_RHO", "0.59"))


def whiten_feats(F, k=None):
    """Centre, remove the top-k principal directions, L2-normalise. Batch-local."""
    k = _WHITEN_K if k is None else k
    X = np.asarray(F, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return X
    X = X - X.mean(0, keepdims=True)
    if k > 0 and X.shape[0] > k:
        try:
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            X = X - X @ Vt[:k].T @ Vt[:k]
        except np.linalg.LinAlgError:
            pass                      # un-whitened distances still beat the bag
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)

_TOK = re.compile(r"[a-z0-9]+")


def _seed_of(s):
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


class FrozenPhi:
    """phi(o, c): frozen random projection of observation tokens + context tags.

    Must be a function of BOTH observation and context (E4).  An observation-only
    phi is constant inside a bucket and provably cannot split anything -- we ship
    that as `context_weight=0.0`, the null control every ablation needs.
    """

    def __init__(self, dim=64, nbuck=4096, seed=0, context_weight=1.0,
                 use_memory=False, mem_weight=1.0):
        rng = np.random.default_rng(seed)
        self.P = rng.standard_normal((nbuck, dim)) / np.sqrt(dim)
        self.dim, self.nbuck, self.cw = dim, nbuck, context_weight
        self.use_memory, self.mw = use_memory, mem_weight

    def _bag(self, toks, nbuck=None):
        nbuck = nbuck or self.nbuck
        v = np.zeros(nbuck)
        for w in toks:
            v[_seed_of(w) % nbuck] += 1.0
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    # Ordinal context features get a THERMOMETER encoding, not a hashed tag.
    # Hashing "t1", "t15", "t28" sends them to unrelated random buckets, so step 1
    # and step 28 come out no more distant than step 15 and step 16 -- measured
    # corr(|delta t|, phi-distance) = -0.187, i.e. the weights were near-uniform
    # and our advantage tracked G2PO at r=0.998.  Thermometer coding makes
    # distance in phi monotone in distance in context.
    _BINS = 12

    def _thermo(self, val, lo, hi):
        v = np.zeros(self._BINS)
        f = (float(val) - lo) / max(hi - lo, 1e-9)
        k = int(np.clip(f, 0.0, 1.0) * self._BINS)
        v[:max(k, 0)] = 1.0
        return v

    def __call__(self, obs, ctx):
        o = self._bag(_TOK.findall(str(obs).lower()))
        z_o = o @ self.P
        if self.cw <= 0:
            z = z_o
        else:
            c = np.concatenate([
                self._thermo(ctx["t"], 0, 30),
                self._thermo(ctx.get("n_unique", ctx.get("n_unique_obs", 0)), 0, 25),
                self._thermo(ctx.get("progress", ctx.get("progress_frac", 0.0)), 0.0, 1.0),
                np.array([float(ctx.get("revisit", ctx.get("revisit_count", 0)) > 0)]),
            ])
            if self.use_memory:
                # SUMMARISED MEMORY as a frozen bag -- isolates "does memory help"
                # from "does a learned encoder help".  Without this cell the
                # learned-vs-frozen contrast changes two things at once.
                mt = []
                # mem_digest: the compaction digest (agent_system/memory/compact.py).
                # Inside a hard bucket the observation block is constant, so this is
                # the only block that can differentiate siblings whose hidden state
                # differs -- the blindness measured on 2026-09-03 (AUC 0.483).
                for k in ("mem_where", "mem_visited", "mem_carrying", "mem_digest"):
                    for item in (ctx.get(k) or []):
                        mt += _TOK.findall(str(item).lower())
                # hash memory DIRECTLY into a dim-sized bag; slicing a 4096-wide
                # bag to its first 64 entries yields all-zeros almost always.
                mbag = self._bag(mt, nbuck=self.dim) if mt else np.zeros(self.dim)
                c = np.concatenate([c, self.mw * mbag])
            n_c = np.linalg.norm(c)
            c = c / n_c if n_c > 0 else c
            n_o = np.linalg.norm(z_o)
            z_o = z_o / n_o if n_o > 1e-12 else z_o
            z = np.concatenate([z_o, self.cw * c])
        n = np.linalg.norm(z)
        return z / n if n > 1e-12 else z


def derive_context(anchor_obs, index, traj_index):
    """Per-occurrence trajectory context, reconstructed from what verl already
    plumbs through the batch.  Phase 0 found `n_unique_obs` (p=0.008) and `t`
    (p=0.011) the most predictive context features on ALFWorld, and both are
    recoverable from the observation sequence alone -- so CCPO needs no new
    fields in the data pipeline.

    NOTE: revisit and n_unique count EXACT observation repeats, while the gate may
    be clustering near-identical observations (ACG_CCPO_SIM > 0). Two observations
    the gate calls one state would still count as two distinct states here. This is
    inert under the shipped configuration -- with phi=hidden the distances come from
    phi_feats and ctx is never consulted -- but it would matter for the bag-of-words
    ablation run with a fuzzy gate, and should be reconciled before reading that
    arm."""
    order = defaultdict(list)
    for i, tj in enumerate(traj_index):
        order[tj].append(i)
    ctx = [None] * len(anchor_obs)
    for tj, idxs in order.items():
        seen, uniq = set(), 0
        for t, i in enumerate(idxs):
            h = str(anchor_obs[i])
            rev = 1 if h in seen else 0
            if not rev:
                seen.add(h); uniq += 1
            ctx[i] = dict(t=t, n_unique=uniq, revisit=rev,
                          progress=uniq / max(t + 1, 1))
    return ctx


# ---------------------------------------------------------------------------
# Gate and target options. Every one defaults to the behaviour that shipped
# before this revision, so each is a single-flag ablation (the convention the
# compaction and budget-forcing components already follow).
# ---------------------------------------------------------------------------

# What the step credit is computed ON.
#   "return"   (default) the step's own gamma-discounted return-to-go, as GiGPO
#              and the previous CCPO revisions use.
#   "nextnode" G2PO's successor value V(next(u)) -- the group-pooled value of the
#              node the action moved INTO. See g2po_node_values below.
# The distinction matters more than the baseline does: a return-to-go carries
# every downstream accident of one trajectory, so conditioning the BASELINE on
# context cannot remove noise that lives in the TARGET.
_TARGET = os.environ.get("ACG_CCPO_TARGET", "return").lower()

# Observation gate. 0.0 keeps the byte-exact match. A value in (0,1) clusters
# observations by SequenceMatcher ratio within a task, as GiGPO's
# build_step_group(enable_similarity=True) does at 0.95 -- one changed character
# stops being a different bucket.
_SIM = float(os.environ.get("ACG_CCPO_SIM", "0.0"))

# Coarse backoff level. When >0, occurrences whose level-0 bucket has fewer than
# min_traj distinct trajectories get a second chance in a LOOSER similarity
# cluster (reference pool = the whole coarse bucket, credit only to the rows
# level 0 could not serve). ~44% of buckets are singletons and currently take
# A^CC = 0, i.e. no step signal at all.
#
# Deliberately NOT a task-level fallback: at task level the bucket holds
# arbitrary observations, b_obs collapses to the task mean return, and A^CC
# would largely restate A^EP -- the trajectory-level credit this method exists
# to refine. A looser state match keeps the "same-ish situation" semantics.
_SIM_BACKOFF = float(os.environ.get("ACG_CCPO_SIM_BACKOFF", "0.0"))

# rho discount applied at the coarse level: a looser gate is a less trustworthy
# metric, and rho is exactly where metric confidence enters lambda*.
_BACKOFF_RHO = float(os.environ.get("ACG_CCPO_BACKOFF_RHO", "0.5"))

# Weight on G2PO's edge (value-gain) term, A^EC. 0 disables it.
_EDGE_W = float(os.environ.get("ACG_CCPO_EDGE_W", "0.0"))

# Matches actor_rollout_ref.actor.invalid_action_penalty_coef. Applied to the
# successor value inside the step term only -- the episode term already carries
# the penalty through token_level_rewards, exactly as in G2PO, where
# compute_group_aggregation_values runs BEFORE apply_invalid_action_penalty and
# writes to non_tensor_batch['step_rewards'], which that penalty never touches.
_INVALID_PEN = float(os.environ.get("ACG_CCPO_INVALID_PEN", "0.1"))

_TERM_OK = ("__terminal__", "success")
_TERM_BAD = ("__terminal__", "failure")


def are_similar(a, b, threshold=0.95):
    """GiGPO's observation-similarity test (gigpo/core_gigpo.py:72).

    The two quick_ratio guards are upper bounds on ratio() and cost O(n) against
    ratio()'s O(n^2); they only ever reject pairs ratio() would also reject.
    """
    if a == b:
        return True
    sm = SequenceMatcher(None, a, b)
    if sm.real_quick_ratio() < threshold or sm.quick_ratio() < threshold:
        return False
    return sm.ratio() >= threshold


def visited_signatures(anchor_obs, traj_index):
    """Per-occurrence hash of the SET of distinct observations the trajectory has
    seen up to and including this step.

    In a deterministic environment entered from a fixed start, the set of states
    visited so far is a sufficient statistic for the current state -- the node
    identity GiGPO and G2PO lack, since they key on the current observation alone.
    Order-invariant on purpose: it keeps buckets populated rather than shattering
    them by path.

    Measured on gate-probe-20260907 this gate has the best LOO residual R2 under
    both targets (+0.059 on return-to-go) but leaves 94.7% of occurrences with no
    second trajectory to compare against -- precise state identity simply has no
    support in a 128-trajectory batch. It is therefore only usable WITH the
    observation gate as a backoff level, which is what ACG_CCPO_GATE=sig does.
    """
    order = defaultdict(list)
    for i, tj in enumerate(traj_index):
        order[str(tj)].append(i)
    sig = [None] * len(anchor_obs)
    for _tj, idxs in order.items():
        seen = set()
        for i in idxs:
            seen.add(str(anchor_obs[i]))
            sig[i] = hashlib.sha1("|".join(sorted(seen)).encode()).hexdigest()[:12]
    return sig


def cluster_keys(anchor_obs, index, sim_thresh=0.0):
    """Bucket key per occurrence: (task, observation) or (task, cluster).

    sim_thresh <= 0 reproduces the exact-match gate. Above 0 the clustering is
    greedy first-fit against each cluster's representative, as in GiGPO's
    build_step_group -- order-dependent and non-transitive, which is a property
    of that algorithm, kept here so the two gates stay comparable.
    """
    n = len(anchor_obs)
    keys = np.empty(n, dtype=object)
    if _GATE == "global":
        # One bucket per task: every occurrence is a candidate neighbour of every
        # other, and exp(-d/tau) does the gating. Nothing falls dead for want of
        # an exact string match, which is what the hard gate loses ~35% of the
        # batch to.
        for i in range(n):
            keys[i] = (str(index[i]),)
        return keys
    if sim_thresh <= 0.0:
        for i in range(n):
            keys[i] = (str(index[i]), str(anchor_obs[i]))
        return keys
    for task in {str(x) for x in index}:
        reps = []
        for i in range(n):
            if str(index[i]) != task:
                continue
            o = str(anchor_obs[i])
            for rep, k in reps:
                if are_similar(o, rep, sim_thresh):
                    keys[i] = k
                    break
            else:
                k = (task, "c%d" % len(reps))
                reps.append((o, k))
                keys[i] = k
    return keys


def g2po_node_values(anchor_obs, index, traj_index, episode_rewards,
                     gamma=0.95, success_reward=10.0):
    """G2PO's group-aggregation state values and successor links.

    Ports compute_group_aggregation_values (baselines/G2PO/g2po/core_g2po.py:65):

        V(g) = mean over visits (i,t) to g of  gamma^(T_i - t) * R_i

    where T_i is the visiting trajectory's length and R_i its episode reward, so
    a node reached close to a success scores high and the same node reached late
    in a failure scores low. Averaging over every sibling that touched the
    observation is where G2PO's variance reduction comes from -- no single
    trajectory's return sets a node's value.

    Returns (VAL, NODE, NEXT). NEXT[i] is the node of the next step in the same
    trajectory, or the terminal sentinel: V = success_reward on success, 0 on
    failure. Requires episode_rewards to be constant within a trajectory, which
    gather_rollout_data guarantees (it writes the final total to every step).
    """
    n = len(anchor_obs)
    NODE = np.empty(n, dtype=object)
    NEXT = np.empty(n, dtype=object)
    VAL = {_TERM_OK: float(success_reward), _TERM_BAD: 0.0}

    order = defaultdict(list)
    for i in range(n):
        order[str(traj_index[i])].append(i)

    acc, cnt = defaultdict(float), defaultdict(int)
    for _tj, ids in order.items():
        task = str(index[ids[0]])
        T = len(ids)
        R = float(episode_rewards[ids[0]])
        term = _TERM_OK if abs(R - success_reward) < 1e-9 else _TERM_BAD
        for t, i in enumerate(ids):
            key = (task, str(anchor_obs[i]))
            NODE[i] = key
            acc[key] += (gamma ** (T - t)) * R
            cnt[key] += 1
            NEXT[i] = (task, str(anchor_obs[ids[t + 1]])) if t + 1 < T else term
    for k in acc:
        VAL[k] = acc[k] / max(cnt[k], 1)
    return VAL, NODE, NEXT


def _successor_values(VAL, NEXT, n):
    return np.array([VAL.get(NEXT[i], 0.0) for i in range(n)], dtype=np.float64)


def g2po_step_advantage(index, VAL, NODE, NEXT, is_action_valid=None,
                        invalid_penalty=None, mode="mean_std_norm",
                        epsilon=1e-6):
    """G2PO's step advantage, ported for use as a DIAGNOSTIC reference.

    Two components (baselines/G2PO/g2po/core_g2po.py:135):
      1. successor value, standardised WITHIN the node (self-inclusive);
         the invalid-action penalty applies here only
      2. value gain V(next) - V(current), standardised across the task; the
         unpenalised successor value is used, as in the reference

    This is what "corr(A_CC, A_G2PO)" should be measured against. The quantity
    the previous revisions reported under that name was G[i] - mean(G[bucket]),
    which is GiGPO's step advantage in mean_norm mode -- see r_vs_gigpo.
    """
    n = len(NODE)
    pen = _INVALID_PEN if invalid_penalty is None else float(invalid_penalty)
    inval = np.zeros(n) if is_action_valid is None else \
        (1.0 - np.asarray(is_action_valid, dtype=float).reshape(-1))
    v_raw = _successor_values(VAL, NEXT, n)
    v_pen = v_raw - pen * inval
    out = np.zeros(n, dtype=np.float64)

    members = defaultdict(list)
    for i in range(n):
        members[NODE[i]].append(i)
    for ids in members.values():
        if len(ids) < 2:
            continue                       # G2PO leaves singleton nodes at 0
        v = v_pen[ids]
        out[ids] = (v - v.mean()) / (v.std(ddof=1) + epsilon) \
            if mode == "mean_std_norm" else (v - v.mean())

    gain = v_raw - np.array([VAL.get(NODE[i], 0.0) for i in range(n)])
    for t in {str(x) for x in index}:
        ids = [i for i in range(n) if str(index[i]) == t]
        if len(ids) < 2:
            continue
        g = gain[ids]
        out[ids] += (g - g.mean()) / (g.std(ddof=1) + epsilon) \
            if mode == "mean_std_norm" else (g - g.mean())
    return out


def ccpo_step_advantage(step_rewards, response_mask, anchor_obs, index,
                        traj_index, phi, tau_scale=1.0, min_traj=2,
                        epsilon=1e-6, return_diag=False, ctx_override=None,
                        rho=None, bias_prior=0.30, shrink=None,
                        aff_labels=None, step_tag="", phi_feats=None,
                        episode_rewards=None, is_action_valid=None,
                        gamma=0.95, success_reward=10.0, target=None,
                        edge_w=None, sim=None, sim_backoff=None):
    """Weighted cross-trajectory leave-one-out step advantage.

    phi_feats: optional (n, d) array of per-sample affinity features. When given
    and ACG_CCPO_PHI=hidden, distances are taken on these (whitened batch-wide)
    instead of on phi(obs, ctx). The trainer supplies the frozen reference
    policy's last-prompt-token hidden state, which costs no extra forward pass.

    episode_rewards / is_action_valid: required for target="nextnode", for the
    edge term, and for the r_vs_g2po diagnostic. Without them the estimator
    silently keeps the return-to-go target and reports r_vs_g2po = nan.

    shrink / target / sim / sim_backoff / edge_w default to their ACG_CCPO_*
    environment values when None. (They were previously read from the
    environment unconditionally, so an explicitly passed `shrink` was ignored.)
    """
    rho = _RHO if rho is None else float(rho)
    shrink = _SHRINK if shrink is None else str(shrink).lower()
    target = _TARGET if target is None else str(target).lower()
    edge_w = _EDGE_W if edge_w is None else float(edge_w)
    sim = _SIM if sim is None else float(sim)
    sim_backoff = _SIM_BACKOFF if sim_backoff is None else float(sim_backoff)

    dev = step_rewards.device
    scores = (step_rewards * response_mask).sum(-1) if step_rewards.dim() > 1 else step_rewards
    G = scores.detach().float().cpu().numpy().astype(np.float64)
    n = len(G)
    ctx = ctx_override if ctx_override is not None else derive_context(anchor_obs, index, traj_index)

    # ---- G2PO node values, when the trainer supplied what they need ---------
    VAL = NODE = NEXT = None
    if episode_rewards is not None:
        VAL, NODE, NEXT = g2po_node_values(anchor_obs, index, traj_index,
                                           episode_rewards, gamma, success_reward)

    # ---- the target the step credit is computed on --------------------------
    if target == "nextnode" and VAL is not None:
        inval = np.zeros(n) if is_action_valid is None else \
            (1.0 - np.asarray(is_action_valid, dtype=float).reshape(-1))
        TGT = _successor_values(VAL, NEXT, n) - _INVALID_PEN * inval
    else:
        if target == "nextnode":
            print("[ccpo] target=nextnode requested but episode_rewards is None; "
                  "falling back to the return-to-go target", flush=True)
            target = "return"
        TGT = G

    adv = np.zeros(n, dtype=np.float64)
    lam_all, neff_all, w_all, _rec = [], [], [], []
    _rel_all, _rel_slope = [], []
    _eff_all = []
    _dump = os.environ.get("ACG_CCPO_DUMP")
    _rows = [] if _dump else None

    # Whitening is batch-wide, not per bucket: the directions being removed are
    # corpus-level register, and estimating them inside a 4-member bucket would
    # delete the very variation the gate is meant to see.
    PHI = None
    if _PHI_MODE.startswith("hidden") and phi_feats is not None:
        _pf = phi_feats.detach().float().cpu().numpy() if hasattr(phi_feats, "detach") \
            else np.asarray(phi_feats, dtype=float)
        if _pf.ndim == 2 and _pf.shape[0] == n:
            PHI = whiten_feats(_pf)
            if _PHI_MODE == "hidden+ctx" and _CTX_W > 0:
                # The trajectory context the prompt cannot carry: how many distinct
                # states this rollout has seen, whether it is standing somewhere it
                # has already been, and how much of its turn budget bought progress.
                # Thermometer-coded so phi-distance is monotone in context distance;
                # hashing "t1"/"t15"/"t28" would send them to unrelated buckets.
                _p = FrozenPhi()
                _c = np.stack([np.concatenate([
                    _p._thermo(ctx[i]["t"], 0, 30),
                    _p._thermo(ctx[i]["n_unique"], 0, 25),
                    _p._thermo(ctx[i]["progress"], 0.0, 1.0),
                    np.array([float(ctx[i]["revisit"] > 0)]),
                ]) for i in range(n)])
                _cn = np.maximum(np.linalg.norm(_c, axis=1, keepdims=True), 1e-9)
                PHI = np.concatenate([PHI, _CTX_W * (_c / _cn)], axis=1)
                _pn = np.maximum(np.linalg.norm(PHI, axis=1, keepdims=True), 1e-9)
                PHI = PHI / _pn

    # ---- reference estimators on the SAME batch, for the diagnostics --------
    # GiGPO: uniform, self-inclusive mean over the EXACT anchor bucket, on the
    # return-to-go (gigpo/core_gigpo.py:334, mean_norm mode -- the repo default).
    # This is the quantity earlier revisions mislabelled "A_G2PO".
    ADV_GIGPO = np.zeros(n, dtype=np.float64)
    _exact = defaultdict(list)
    for i in range(n):
        _exact[(str(index[i]), str(anchor_obs[i]))].append(i)
    for _ids in _exact.values():
        if len(_ids) > 1:
            _a = np.asarray(_ids)
            ADV_GIGPO[_a] = G[_a] - G[_a].mean()
    ADV_G2PO = g2po_step_advantage(index, VAL, NODE, NEXT, is_action_valid) \
        if VAL is not None else None

    # ---- gate levels --------------------------------------------------------
    if _GATE == "sig":
        # Level 0: (task, visited-set signature) -- precise state identity.
        # Level 1: (task, observation) -- the baselines' gate, as a backoff for the
        # ~95% of occurrences level 0 cannot serve. The multi-level machinery below
        # already credits each occurrence at the FINEST level that can serve it, so
        # this keeps the precision where it is affordable and the standard gate
        # everywhere else. rho is reduced at level 1 exactly as for the similarity
        # backoff, because a coarser match deserves less metric confidence.
        _sig = visited_signatures(anchor_obs, traj_index)
        _k0 = np.empty(n, dtype=object)
        for _i in range(n):
            _k0[_i] = (str(index[_i]), _sig[_i])
        levels = [(0, _k0, rho),
                  (1, cluster_keys(anchor_obs, index, 0.0), rho * _BACKOFF_RHO)]
    else:
        levels = [(0, cluster_keys(anchor_obs, index, sim), rho)]
        if sim_backoff > 0.0:
            levels.append((1, cluster_keys(anchor_obs, index, sim_backoff),
                           rho * _BACKOFF_RHO))

    assigned = np.zeros(n, dtype=bool)
    level_of = np.full(n, -1, dtype=np.int64)

    for lvl, keys, rho_l in levels:
        buckets = defaultdict(list)
        for i in range(n):
            buckets[keys[i]].append(i)

        for _bkey, idx in buckets.items():
            # At level>0 the reference pool is the whole coarse bucket, but only
            # rows the finer level could not serve receive credit -- so a lone
            # occurrence borrows populated neighbours instead of being dropped.
            todo = [a for a in range(len(idx)) if not assigned[idx[a]]]
            if not todo:
                continue
            trajs = [str(traj_index[i]) for i in idx]
            if len(set(trajs)) < min_traj:
                continue
            _bhash = hashlib.sha1(str(_bkey).encode()).hexdigest()[:10]
            F = PHI[idx] if PHI is not None else \
                np.stack([phi(anchor_obs[i], ctx[i]) for i in idx])
            # Gram identity rather than the (n, n, d) broadcast difference: the
            # episode-start observation is shared by every trajectory in a task, so
            # that bucket holds all of them, and with a 1536-d hidden-state phi the
            # broadcast form allocates n^2 * d floats -- 200 MB at n=128 and 3 GB at
            # n=500, which the similarity gate makes reachable. Exact, not an
            # approximation: ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v.
            _sq = (F * F).sum(1)
            D = np.sqrt(np.maximum(_sq[:, None] + _sq[None, :] - 2.0 * (F @ F.T), 0.0))
            off = D[np.triu_indices(len(idx), 1)]
            tau = float(np.median(off)) if off.size and np.median(off) > 1e-9 else 1.0
            tau *= max(float(_TAU_ENV) if _TAU_ENV else tau_scale, 1e-6)

            # ---- grouping relevance: does phi-distance predict return distance? -
            # lambda only detects whether phi shifts the MEAN of the baseline, so a
            # phi that genuinely orders neighbours by similarity but happens not to
            # move that mean reads as zero signal. This is the direct question, per
            # bucket, over the pairs that actually feed the estimator: if phi is
            # informative, pairs that are close in phi should differ less in target.
            # Reward-free in the sense that matters -- it is a DIAGNOSTIC, never fed
            # back into the weights, so the leave-one-out argument is untouched.
            if len(idx) > 3:
                _iu = np.triu_indices(len(idx), 1)
                _dp = D[_iu]
                _gp = np.abs(TGT[np.asarray(idx)][_iu[0]] - TGT[np.asarray(idx)][_iu[1]])
                if _dp.std() > 1e-9 and _gp.std() > 1e-9:
                    _rel_all.append(float(np.corrcoef(_dp, _gp)[0, 1]))
                    # variance of the target explained by phi-distance, within bucket
                    _sl = np.polyfit(_dp, _gp, 1)[0]
                    _rel_slope.append(float(_sl))

            for a in todo:
                i = idx[a]
                other = [b for b in range(len(idx)) if trajs[b] != trajs[a]]
                if not other:
                    continue
                w = np.exp(-D[a, other] / tau)
                w_all.extend(w.tolist())
                # aggregate within each reference trajectory first, then across
                # them: the independent unit is the trajectory, not the occurrence
                per = defaultdict(lambda: [0.0, 0.0])
                for k, b in enumerate(other):
                    per[trajs[b]][0] += w[k] * TGT[idx[b]]
                    per[trajs[b]][1] += w[k]
                alpha = np.array([v[1] for v in per.values()])
                gbar = np.array([v[0] / max(v[1], 1e-12) for v in per.values()])
                ne = (alpha.sum() ** 2) / max((alpha ** 2).sum(), 1e-12)
                b_loo = float((alpha * gbar).sum() / max(alpha.sum(), 1e-12))
                # ---- bias-variance shrinkage, uncertainty-aware --------------
                # n_eff alone is the WRONG rule: concentrated weights are
                # precisely what context-conditioning is for, yet they LOWER
                # n_eff and so trigger fallback to the uniform baseline. n_eff
                # sees only the variance cost of conditioning, never the bias
                # benefit.
                #
                # `other` holds LOCAL positions into idx and must be mapped back
                # to global rows before indexing the target, as the b_loo loop
                # above does. Without the map this read rows 0..len(idx)-1 of the
                # whole batch; b_obs carries weight (1-lambda) on every sample
                # and is the entire estimator wherever lambda=0.
                J = float(len(per))
                oth = [idx[b] for b in other]
                # phi-weighted sd over the SAME neighbourhood and the SAME weights
                # that produced b_loo. Uses b_loo as the centre so the mean and the
                # scale are consistent with one another.
                _wv = w / max(w.sum(), 1e-12)
                _sig = float(np.sqrt(max((_wv * (TGT[oth] - b_loo) ** 2).sum(), 0.0)))
                s2 = float(np.var(TGT[oth])) if len(oth) > 1 else 0.0
                var_gain = max(1.0 / max(ne, 1e-9) - 1.0 / max(J, 1e-9), 0.0)
                b_obs = float(TGT[oth].mean())                # Record the occurrence; the shrinkage weight is applied below,
                # because "eb_pooled" needs the whole batch before it can choose one.
                _rec.append(dict(i=i, lvl=lvl, bhash=_bhash, b_loo=b_loo, b_obs=b_obs,
                                 s2=s2, ne=ne, J=J, var_gain=var_gain, rho=rho_l,
                                 sig=_sig))
                assigned[i] = True
                level_of[i] = lvl

    # ---- shrinkage ---------------------------------------------------------
    # n_eff alone is the WRONG rule: concentrated weights are precisely what
    # context-conditioning is for, yet they LOWER n_eff and so trigger fallback to
    # the uniform baseline. n_eff sees only the variance cost of conditioning,
    # never the bias benefit.
    # Both rules are ALWAYS evaluated and reported; only the selected one is
    # applied. The per-occurrence rule degenerating to lambda = 0 is exactly the
    # failure that makes CCPO the uniform baseline in disguise, and it is invisible
    # unless the alternative is measured on the same batch.
    lam_pooled = None
    _lam_pooled_obs = float("nan")
    _tau2 = float("nan")
    if _rec:
        _d2 = np.array([(r["rho"] * (r["b_obs"] - r["b_loo"])) ** 2 for r in _rec])
        _vd = np.array([r["rho"] ** 2 * r["s2"] * r["var_gain"] for r in _rec])
        _den = float(_d2.mean())
        _lam_pooled_obs = 0.0 if _den <= 1e-12 else max(0.0, min(1.0, 1.0 - float(_vd.mean()) / _den))
        # Signal variance, by method of moments over the batch:
        #   E[d^2] = tau^2 + E[Var(d)]   =>   tau^2 = max(0, E[d^2] - E[Var(d)])
        # This is the quantity the ensemble can estimate and a single bucket cannot.
        _tau2 = max(0.0, _den - float(_vd.mean()))
        _lam_eb_obs = np.array([
            0.0 if abs(r["rho"] * (r["b_obs"] - r["b_loo"])) < 1e-12
            else max(0.0, min(1.0, 1.0 - (r["s2"] * r["var_gain"])
                              / ((r["rho"] * (r["b_obs"] - r["b_loo"])) ** 2)))
            for r in _rec])
    else:
        _lam_eb_obs = np.zeros(1)
    if shrink == "eb_pooled" and _rec:
        # James-Stein / Efron-Morris: with d = signal + noise and
        # Var(noise) = s2*(1/n_eff - 1/J) known per occurrence,
        #   lambda = max(0, 1 - E[Var(d)] / E[d^2])
        # is the fraction of the realised disagreement that is real signal,
        # estimated over every live occurrence in the batch rather than over the
        # two or three trajectories of a single bucket.
        lam_pooled = _lam_pooled_obs

    _tgt_sd = float(np.std(TGT)) if len(TGT) > 1 else 1.0
    if _tgt_sd < 1e-9:
        _tgt_sd = 1.0
    for _r in _rec:
        i, b_loo, b_obs = _r["i"], _r["b_loo"], _r["b_obs"]
        s2, var_gain, rho_l = _r["s2"], _r["var_gain"], _r["rho"]
        if shrink == "one":
            # FORCE lam = 1: full weight on the phi-weighted leave-one-out baseline,
            # OVERRIDING the empirical-Bayes estimate.
            #
            # Why this exists: under the hard gate the EB shrinkage collapses to
            # lam ~ 0.011 because tau^2 measures 0.00e+00 -- the estimator finds no
            # between-bucket variance for the context term to explain, and correctly
            # shrinks it away. That makes "CCPO on a refined partition" untestable:
            # the estimator switches context off before it can be evaluated.
            #
            # `one` overrides that verdict so the question can be asked directly. It is
            # NOT a defensible production setting -- it forces a term the data says is
            # worthless. Use it only to test whether context conditioning helps when it
            # is made to operate on a GOOD partition (H-Q predicts it will hurt:
            # phi-weighting inside a bucket measured 2-10 R^2 points worse than uniform).
            lam = 1.0
        elif _GATE == "global":
            # b_obs here is the uniform mean over the whole task, which measured
            # 0.4248 against the hard gate's 0.4579 -- WORSE. The quantity that
            # measured better (0.4841) is the phi-weighted b_loo. Shrinking toward
            # b_obs would therefore move the estimator toward the worse baseline,
            # and since lam has been exactly 0.000 in every run to date that is
            # precisely where it would land. Under a global gate the soft kernel
            # IS the gate, so there is nothing to shrink toward and lam = 1.
            lam = 1.0
        elif shrink == "eb_hier":
            # Hierarchical empirical Bayes (Efron-Morris). tau^2 comes from the
            # whole batch, which is where the statistical power is; the shrinkage
            # is then per occurrence against ITS OWN noise:
            #     lam_i = tau^2 / (tau^2 + Var_i)
            # "eb" is this with tau^2 estimated from one bucket (about two degrees
            # of freedom, so lam collapses to 0); "eb_pooled" is this with every
            # Var_i replaced by its batch mean (power, but no adaptivity). This
            # keeps both: concentrated, well-supported neighbourhoods shrink less
            # than thin ones, and the decision of how much signal exists at all is
            # made once, over thousands of occurrences.
            _vd_i = (rho_l ** 2) * s2 * var_gain
            lam = 0.0 if (_tau2 + _vd_i) <= 1e-12 else float(_tau2 / (_tau2 + _vd_i))
        elif lam_pooled is not None:
            lam = lam_pooled
        elif shrink == "mse":
            # Kept for reproducing v6. Writing the bias prior in sd units
            # (B = 0.30*sigma) makes s2 cancel outright, so this reduces to
            # 0.09/(0.09+var_gain): a pure function of n_eff and J, i.e. exactly
            # the n_eff rule the comment above rejects. Measured on 305k samples:
            # corr(lam,|d|) = -0.239.
            B_eff = rho_l * bias_prior * (np.sqrt(s2) + 1e-9)
            lam = float((B_eff ** 2) / ((B_eff ** 2) + s2 * var_gain + 1e-12))
        else:
            # Positive-part empirical Bayes on this occurrence's own disagreement.
            #   d = rho*(b_obs - b_LOO),  Var(d) = s2*(1/n_eff - 1/J)
            #   lam = max(0, 1 - Var(d)/d^2)
            # rho scales d before it meets its own noise, so an uninformative
            # metric cannot fire and rho->0 is an exact fallback to the uniform
            # baseline. Note this test has about two degrees of freedom and
            # measured lam = 0.000 on a real batch -- see "eb_pooled".
            _d = rho_l * (b_obs - b_loo)
            _vd1 = s2 * var_gain
            lam = 0.0 if abs(_d) < 1e-12 else float(1.0 - _vd1 / (_d * _d))
        lam = min(1.0, max(0.0, lam))
        adv[i] = TGT[i] - (lam * b_loo + (1 - lam) * b_obs)
        if _STD_MODE == "local":
            # Coherent standardisation: the kernel that chose the neighbourhood also
            # sets the scale. The floor keeps a degenerate all-identical
            # neighbourhood from exploding the credit.
            adv[i] = adv[i] / max(_r["sig"], _STD_FLOOR * _tgt_sd)
        lam_all.append(lam)
        neff_all.append(_r["ne"])
        # effect: how far our credit departs from the uniform baseline. If this is
        # ~0 the estimator is that baseline in disguise.
        _eff = lam * (b_obs - b_loo)
        _eff_all.append(abs(_eff))
        if _rows is not None:
            _rows.append((
                str(index[i]), str(traj_index[i]), _r["bhash"], _r["lvl"],
                float(G[i]), float(TGT[i]), float(lam), float(b_loo),
                float(b_obs), float(_r["ne"]), float(_r["J"]), float(adv[i]),
                float(ADV_GIGPO[i]),
                "" if ADV_G2PO is None else float(ADV_G2PO[i]),
                float(_eff),
                "" if aff_labels is None else str(aff_labels[i]),
            ))

    # ---- edge term (G2PO component 2), off by default -----------------------
    # V(next) - V(current), standardised per task. This is the half of G2PO that
    # scores movement toward the goal rather than the value of standing still.
    edge_cov = 0.0
    if edge_w > 0 and VAL is not None:
        v_next = _successor_values(VAL, NEXT, n)
        v_cur = np.array([VAL.get(NODE[i], 0.0) for i in range(n)])
        gain = v_next - v_cur
        for t in {str(x) for x in index}:
            ids = [i for i in range(n) if str(index[i]) == t]
            if len(ids) > 1:
                g = gain[ids]
                adv[np.asarray(ids)] += edge_w * (g - g.mean()) / (g.std(ddof=1) + epsilon)
        edge_cov = 1.0

    out = torch.tensor(adv, dtype=step_rewards.dtype, device=dev)
    if not return_diag:
        return out

    def _corr(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        if int(m.sum()) < 3 or x[m].std() < 1e-12 or y[m].std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(x[m], y[m])[0, 1])

    _eff = np.array(_eff_all, dtype=float) if _eff_all else np.zeros(1)
    _live = assigned
    _adv_scale = float(np.abs(adv[_live]).mean()) if _live.any() else float("nan")
    _r_gigpo = _corr(adv[_live], ADV_GIGPO[_live])
    _r_g2po = float("nan") if ADV_G2PO is None else _corr(adv[_live], ADV_G2PO[_live])

    # ---- grouping dump: the RAW inputs to the gate, for offline re-bucketing ---
    # Separate from the CSV above so that schema stays stable. Written only when
    # ACG_CCPO_GDUMP is set, because it carries full observation text and is
    # ~100x larger per step. This is what lets an alternative gate be evaluated
    # without spending a training run on it.
    _gdump = os.environ.get("ACG_CCPO_GDUMP")
    if _gdump:
        try:
            import json as _json
            with open(_gdump, "a") as _fh:
                for i in range(n):
                    _fh.write(_json.dumps({
                        "step": str(step_tag),
                        "uid": str(index[i]),
                        "traj_uid": str(traj_index[i]),
                        "t": ctx[i]["t"],
                        "n_unique": ctx[i]["n_unique"],
                        "revisit": ctx[i]["revisit"],
                        "progress": ctx[i]["progress"],
                        "obs": str(anchor_obs[i]),
                        "G": float(G[i]),
                        "target": float(TGT[i]),
                    }) + "\n")
        except OSError:
            pass

    if _rows:
        try:
            _new = not os.path.exists(_dump)
            with open(_dump, "a") as _fh:
                if _new:
                    _fh.write("step,uid,traj_uid,bucket,level,G,target,lam,b_loo,"
                              "b_obs,n_eff,J,adv_cc,adv_gigpo,adv_g2po,effect,aff\n")
                for _r in _rows:
                    _fh.write(str(step_tag) + "," + ",".join(str(x) for x in _r) + "\n")
        except OSError:
            pass

    # ACG length-bias diagnostic. If A^CC itself correlates with response length,
    # our own step credit is rewarding verbosity -- a confound to rule out before
    # blaming the reward or the KL anchor for the length growth.
    _lens = response_mask.sum(-1).detach().cpu().numpy().astype(float)
    _m = np.isfinite(adv) & (_lens > 0)
    _corr_len = _corr(adv[_m], _lens[_m])
    if _dump:
        # Separate file: these rows have a different schema from the per-sample
        # ones above, and appending both to one path made the dump unparseable.
        try:
            with open(_dump + ".len.csv", "a") as _fh:
                for _l, _a in zip(_lens[_m], adv[_m]):
                    _fh.write(f"{int(_l)},{_a:.6f}\n")
        except OSError:
            pass

    return out, dict(
        lam_u_mean=float(np.mean(lam_all)) if lam_all else 0.0,
        lam_u_gt50=float(np.mean(np.array(lam_all) > 0.5)) if lam_all else 0.0,
        lam_pooled=(float(lam_pooled) if lam_pooled is not None else float('nan')),
        # what each rule WOULD give on this batch, whichever is in force
        lam_pooled_obs=float(_lam_pooled_obs), tau2=float(_tau2),
        # grouping relevance: corr(phi distance, |target difference|) within a
        # bucket, averaged over buckets. >0 means phi-similar pairs really do have
        # more similar targets -- the method's premise, measured directly and
        # independently of whether phi moves the baseline's mean.
        phi_rel_corr=float(np.mean(_rel_all)) if _rel_all else float("nan"),
        phi_rel_gt0=float(np.mean(np.array(_rel_all) > 0)) if _rel_all else float("nan"),
        phi_rel_slope=float(np.mean(_rel_slope)) if _rel_slope else float("nan"),
        lam_eb_obs=float(np.mean(_lam_eb_obs)),
        lam_eb_obs_gt0=float(np.mean(_lam_eb_obs > 0)),
        n_eff_mean=float(np.mean(neff_all)) if neff_all else 0.0,
        E_w=float(np.mean(w_all)) if w_all else float("nan"),
        live_frac=float(_live.mean()), live_mask=_live,
        n_buckets=len(_exact),
        lvl1_frac=float((level_of == 1).mean()),
        phi_mode=(_PHI_MODE if PHI is not None else "bow"), rho=rho,
        target=target, sim=sim, sim_backoff=sim_backoff,
        edge_w=edge_w, edge_cov=edge_cov,
        acc_len_corr=_corr_len,
        effect_mean=float(_eff.mean()), effect_p90=float(np.percentile(_eff, 90)),
        effect_rel=float(_eff.mean() / _adv_scale)
        if _adv_scale and np.isfinite(_adv_scale) and _adv_scale > 1e-12 else float("nan"),
        r_vs_gigpo=_r_gigpo, r_vs_g2po=_r_g2po,
    )
