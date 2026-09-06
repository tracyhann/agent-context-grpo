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
import hashlib
import os
import re

import numpy as np
import torch

# Shrinkage rule for lambda*. "eb" (default) is the positive-part empirical-Bayes
# rule on the realised baseline disagreement; "mse" reproduces the v6 rule.
# Validated offline in experiments/08-27/probe_shrinkage.py against 305k real
# training samples -- see results/probe_shrinkage.txt.
_SHRINK = os.environ.get("ACG_CCPO_SHRINK", "eb").lower()

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
_PHI_MODE = os.environ.get("ACG_CCPO_PHI", "hidden").lower()

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
    fields in the data pipeline."""
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


def ccpo_step_advantage(step_rewards, response_mask, anchor_obs, index,
                        traj_index, phi, tau_scale=1.0, min_traj=2,
                        epsilon=1e-6, return_diag=False, ctx_override=None,
                        rho=None, bias_prior=0.30, shrink='mse', progress_w=0.0,
                        aff_labels=None, step_tag="", phi_feats=None):
    """Weighted cross-trajectory leave-one-out step advantage.

    phi_feats: optional (n, d) array of per-sample affinity features. When given
    and ACG_CCPO_PHI=hidden, distances are taken on these (whitened batch-wide)
    instead of on phi(obs, ctx). The trainer supplies the frozen reference
    policy's last-prompt-token hidden state, which costs no extra forward pass.
    """
    rho = _RHO if rho is None else rho
    dev = step_rewards.device
    scores = (step_rewards * response_mask).sum(-1) if step_rewards.dim() > 1 else step_rewards
    G = scores.detach().float().cpu().numpy()
    n = len(G)
    ctx = ctx_override if ctx_override is not None else derive_context(anchor_obs, index, traj_index)

    adv = np.zeros(n, dtype=np.float64)
    lam_all, neff_all, w_all, live = [], [], [], 0
    # diagnostics: effect size vs the uniform baseline, and the G2PO comparison
    _eff_all, _g2po_pairs = [], []
    _dump = os.environ.get("ACG_CCPO_DUMP")
    _rows = [] if _dump else None

    # Whitening is batch-wide, not per bucket: the directions being removed are
    # corpus-level register, and estimating them inside a 4-member bucket would
    # delete the very variation the gate is meant to see.
    PHI = None
    if _PHI_MODE == "hidden" and phi_feats is not None:
        _pf = phi_feats.detach().float().cpu().numpy() if hasattr(phi_feats, "detach") \
            else np.asarray(phi_feats, dtype=float)
        if _pf.ndim == 2 and _pf.shape[0] == n:
            PHI = whiten_feats(_pf)

    buckets = defaultdict(list)
    for i in range(n):
        buckets[(str(index[i]), str(anchor_obs[i]))].append(i)

    for _bkey, idx in buckets.items():
        _bkey = hashlib.sha1(str(_bkey).encode()).hexdigest()[:10]
        trajs = [str(traj_index[i]) for i in idx]
        if len(set(trajs)) < min_traj:
            continue                                   # backoff: A^CC = 0
        F = PHI[idx] if PHI is not None else \
            np.stack([phi(anchor_obs[i], ctx[i]) for i in idx])
        D = np.sqrt(np.maximum(((F[:, None, :] - F[None, :, :]) ** 2).sum(-1), 0.0))
        off = D[np.triu_indices(len(idx), 1)]
        tau = float(np.median(off)) if off.size and np.median(off) > 1e-9 else 1.0
        tau *= max(tau_scale, 1e-6)

        for a, i in enumerate(idx):
            other = [b for b in range(len(idx)) if trajs[b] != trajs[a]]
            if not other:
                continue
            w = np.exp(-D[a, other] / tau)
            w_all.extend(w.tolist())
            # aggregate within each reference trajectory first, then across them:
            # the independent unit is the trajectory, not the occurrence
            per = defaultdict(lambda: [0.0, 0.0])
            for k, b in enumerate(other):
                per[trajs[b]][0] += w[k] * G[idx[b]]
                per[trajs[b]][1] += w[k]
            alpha = np.array([v[1] for v in per.values()])
            gbar = np.array([v[0] / max(v[1], 1e-12) for v in per.values()])
            ne = (alpha.sum() ** 2) / max((alpha ** 2).sum(), 1e-12)
            b_loo = float((alpha * gbar).sum() / max(alpha.sum(), 1e-12))
            # shrink toward the plain observation baseline when support is thin.
            # NOTE: an SNR gate lived here and was removed -- it used
            # std(d)/mean(d), a coefficient of variation that sits near 0.4 even
            # for pure noise, and silently scaled every lambda by ~0.34.
            # ---- bias-variance shrinkage, uncertainty-aware -------------------
            # n_eff alone is the WRONG rule: concentrated weights are precisely
            # what context-conditioning is for, yet they LOWER n_eff and so
            # trigger fallback to the uniform (G2PO) baseline.  n_eff sees only
            # the variance cost of conditioning, never the bias benefit.
            #
            # Shrink by comparing the two estimators' MSE instead:
            #   b_obs : low variance (all J trajectories), HIGH bias  = the
            #           false-merge bias, measured at 0.26-0.36 sd in Phase 0
            #   b_LOO : low bias IF the metric is trustworthy, variance ~ s2/n_eff
            #
            #   lambda* = (rho*B)^2 / ( (rho*B)^2 + s2*(1/n_eff - 1/J) )
            #
            # rho in [0,1] is confidence in the distances: only a metric that is
            # actually informative realises the bias reduction, so an unreliable
            # metric discounts B and collapses lambda to 0 (= G2PO). Limits are
            # all correct: rho->0 or B->0 gives b_obs; n_eff->J gives b_LOO=b_obs;
            # s2->0 gives full conditioning.
            J = float(len(per))
            # `other` holds LOCAL positions into idx, so it must be mapped back to
            # global rows before indexing G -- as the b_loo loop above already does
            # via G[idx[b]]. Without the map, G[other] reads rows 0..len(idx)-1 of
            # the whole batch: arbitrary trajectories with no relation to this
            # bucket. That made b_obs uncorrelated (r=+0.051) with the baseline it
            # was meant to be, and on the ~70% of samples where lambda=0, b_obs IS
            # the entire estimator.
            oth = [idx[b] for b in other]
            s2 = float(np.var(G[oth])) if len(oth) > 1 else 0.0
            var_gain = max(1.0 / max(ne, 1e-9) - 1.0 / max(J, 1e-9), 0.0)
            b_obs = float(G[oth].mean())
            if _SHRINK == "mse":
                # The rule above, kept for reproducing v6. Note that writing the
                # bias prior in sd units (B = 0.30*sigma) makes s2 cancel outright,
                # so this reduces to 0.09/(0.09+var_gain): a pure function of
                # n_eff and J, i.e. exactly the n_eff rule the comment rejects.
                # Measured on 305k real samples: corr(lam,|d|) = -0.239 (it shrinks
                # hardest where the correction is largest) and corr(lam,n_eff/J)
                # = +0.947. It also fires at lam=0.136 on the 295k samples whose
                # disagreement sits below its own sampling noise.
                B_eff = rho * bias_prior * (np.sqrt(s2) + 1e-9)
                lam = float((B_eff ** 2) / ((B_eff ** 2) + s2 * var_gain + 1e-12))
            else:
                # Positive-part empirical Bayes on the REALISED disagreement.
                #   d      = b_obs - b_LOO
                #   Var(d) = s2 * (1/n_eff - 1/J)
                #   lam    = max(0, 1 - Var(d)/(rho*d)^2)
                # Scale-free, and positively correlated with |d| by construction
                # (+0.212 measured). Zeroes the below-noise band exactly and gives
                # lam=0.875 where d exceeds 2 sigma.
                #
                # rho carries the same meaning as in the MSE rule: only a metric
                # that is actually informative realises the disagreement as bias
                # reduction, so rho scales d before it is compared to its own
                # noise. Without this factor an uninformative phi still fires
                # wherever d happens to clear its noise band -- precisely the
                # noise-injection mode this whole revision exists to remove --
                # and the rho -> 0 limit stops reducing to the uniform baseline.
                _d = rho * (b_obs - b_loo)
                _vd = s2 * var_gain
                lam = 0.0 if abs(_d) < 1e-12 else float(1.0 - _vd / (_d * _d))
            lam = min(1.0, max(0.0, lam))
            adv[i] = G[i] - (lam * b_loo + (1 - lam) * b_obs)
            lam_all.append(lam); neff_all.append(ne); live += 1
            # ---- per-sample diagnostics --------------------------------------
            # effect: how far our credit departs from the uniform (G2PO-style)
            # baseline. If this is ~0 the estimator is the baseline in disguise,
            # and no amount of training can separate the two arms.
            _eff = lam * (b_obs - b_loo)
            _eff_all.append(abs(_eff))
            # G2PO's own step advantage on the SAME bucket: uniform pooling that
            # INCLUDES self (their node value leaks the trajectory's own return).
            _adv_g2po = float(G[i] - G[idx].mean())
            _g2po_pairs.append((float(adv[i]), _adv_g2po))
            if _rows is not None:
                _rows.append((
                    str(index[i]), str(traj_index[i]), _bkey,
                    float(G[i]), float(lam), float(b_loo), float(b_obs),
                    float(ne), float(J), float(adv[i]), _adv_g2po, float(_eff),
                    "" if aff_labels is None else str(aff_labels[i]),
                ))

    # ---- edge / progress term (from the G2PO reference implementation) -------
    # G2PO adds V(next node) - V(current node), normalised per task. Ours uses
    # LEAVE-ONE-OUT node values (other trajectories only), so a trajectory's own
    # outcome never enters the value used to judge its own action -- avoiding the
    # same-trajectory leakage present in the released G2PO code. Terminal steps:
    # the next value is the realised terminal return (an env reward, legitimate).
    prog_cov = 0.0
    if progress_w > 0:
        G_np = step_rewards.detach().cpu().numpy().astype(float).reshape(-1)
        bkt = [(str(index[i]), str(anchor_obs[i])) for i in range(n)]
        members = defaultdict(list)
        for i in range(n):
            members[bkt[i]].append(i)
        order = defaultdict(list)
        for i in range(n):
            order[str(traj_index[i])].append(i)
        nxt = {a_: b_ for ids in order.values() for a_, b_ in zip(ids, ids[1:])}
        def loo_value(b, me):
            vals = [G_np[j] for j in members.get(b, []) if str(traj_index[j]) != me]
            return float(np.mean(vals)) if vals else None
        prog = np.full(n, np.nan)
        for i in range(n):
            me = str(traj_index[i])
            v_cur = loo_value(bkt[i], me)
            if v_cur is None:
                continue
            v_next = loo_value(bkt[nxt[i]], me) if i in nxt else G_np[i]
            if v_next is None:
                continue
            prog[i] = v_next - v_cur
        for t in {str(x) for x in index}:
            ids = [i for i in range(n) if str(index[i]) == t and np.isfinite(prog[i])]
            if len(ids) > 1:
                v = prog[ids]; mu_, sd_ = v.mean(), v.std()
                for i in ids:
                    adv[i] += progress_w * (prog[i] - mu_) / (sd_ + epsilon)
        prog_cov = float(np.isfinite(prog).mean())

    out = torch.tensor(adv, dtype=step_rewards.dtype, device=dev)
    if not return_diag:
        return out

    # ACG length-bias diagnostic. If A^CC itself correlates with response length,
    # our own step credit is rewarding verbosity -- a confound that has to be ruled
    # out before blaming the reward or the KL anchor for the length growth.
    # ACG_CCPO_DUMP=<path> additionally appends per-sample (len, A^CC) for scatters.
    # effect size of the estimator vs the uniform baseline, and how closely our
    # advantage tracks G2PO's on the same batch. r ~ 1.0 means the two arms cannot
    # be separated by any experiment, however long.
    _eff = np.array(_eff_all, dtype=float) if _eff_all else np.zeros(1)
    if len(_g2po_pairs) > 2:
        _o = np.array([p[0] for p in _g2po_pairs]); _g = np.array([p[1] for p in _g2po_pairs])
        _r_g2po = float(np.corrcoef(_o, _g)[0, 1]) if _o.std() > 1e-12 and _g.std() > 1e-12 else float("nan")
        _adv_scale = float(np.abs(_o).mean())
    else:
        _r_g2po, _adv_scale = float("nan"), float("nan")
    if _rows:
        try:
            _new = not os.path.exists(_dump)
            with open(_dump, "a") as _fh:
                if _new:
                    _fh.write("step,uid,traj_uid,bucket,G,lam,b_loo,b_obs,n_eff,J,"
                              "adv_cc,adv_g2po,effect,aff\n")
                for _r in _rows:
                    _fh.write(str(step_tag) + "," + ",".join(str(x) for x in _r) + "\n")
        except OSError:
            pass

    _lens = response_mask.sum(-1).detach().cpu().numpy().astype(float)
    _acc = np.asarray(adv, dtype=float)
    _m = np.isfinite(_acc) & (_lens > 0)
    if int(_m.sum()) > 2 and _acc[_m].std() > 1e-9 and _lens[_m].std() > 1e-9:
        _corr = float(np.corrcoef(_acc[_m], _lens[_m])[0, 1])
    else:
        _corr = float("nan")
    _dump = os.environ.get("ACG_CCPO_DUMP")
    if _dump:
        try:
            with open(_dump, "a") as _fh:
                for _l, _a in zip(_lens[_m], _acc[_m]):
                    _fh.write(f"{int(_l)},{_a:.6f}\n")
        except OSError:
            pass

    return out, dict(
        lam_u_mean=float(np.mean(lam_all)) if lam_all else 0.0,
        lam_u_gt50=float(np.mean(np.array(lam_all) > 0.5)) if lam_all else 0.0,
        n_eff_mean=float(np.mean(neff_all)) if neff_all else 0.0,
        E_w=float(np.mean(w_all)) if w_all else float("nan"),
        live_frac=live / max(n, 1), n_buckets=len(buckets), progress_cov=prog_cov,
        phi_mode=("hidden" if PHI is not None else "bow"), rho=rho,
        acc_len_corr=_corr,
        # Q3 diagnostics
        effect_mean=float(_eff.mean()), effect_p90=float(np.percentile(_eff, 90)),
        effect_rel=float(_eff.mean() / _adv_scale) if _adv_scale and _adv_scale > 1e-12 else float("nan"),
        r_vs_g2po=_r_g2po,
    )
