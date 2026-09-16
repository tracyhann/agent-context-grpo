"""Learned predictive phi + memory latents + learned affinity.

This is the full method from idea.md, replacing the frozen stand-in that was
benchmarked earlier. Three components, all trained REWARD-FREE -- no return, no
advantage, no reward ever enters the grouping, which is what keeps the LOO
baseline valid.

  LearnedPhi   z = f_theta(observation tokens, SUMMARISED MEMORY tokens, progress)
               Memory is the component idea.md names as the differentiator over
               HGPO (needs step-divisible exact suffixes) and G2PO (sees only
               what is inside o_t). Earlier runs omitted it entirely.

  MuPsi        mu_psi(z, a) = E[ sum_k lambda^{k-1} phi(chi_{t+k}) | z, a ]
               action-conditioned successor features, TD-trained on ordinary
               observed transitions.

  LearnedAffinity   d(u,v) from a learned metric over the predicted-consequence
               sets, rather than a fixed exp(-d/tau) kernel on raw distances.

Trained jointly by TD on transitions. The target is stop-gradient'd, and phi is
updated through the TD loss only -- so the representation is shaped by "what
happens next", never by "what reward came".
"""
import numpy as np
import torch
import torch.nn as nn

from ccpo.core_ccpo import _seed_of, _TOK


def hashed_bag(tokens, nbuck):
    v = np.zeros(nbuck, dtype=np.float32)
    for w in tokens:
        v[_seed_of(w) % nbuck] += 1.0
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def featurise(obs, ctx, n_obs=2048, n_mem=1024):
    """Raw inputs to LearnedPhi: observation bag, MEMORY bag, progress scalars."""
    o = hashed_bag(_TOK.findall(str(obs).lower()), n_obs)
    mem_tokens = []
    for k in ("mem_where", "mem_visited", "mem_carrying"):
        for item in (ctx.get(k) or []):
            mem_tokens += _TOK.findall(str(item).lower())
    m = hashed_bag(mem_tokens, n_mem)
    s = np.array([
        ctx.get("t", 0) / 30.0,
        ctx.get("n_unique", ctx.get("n_unique_obs", 0)) / 25.0,
        float(ctx.get("revisit", ctx.get("revisit_count", 0)) > 0),
        float(ctx.get("progress", ctx.get("progress_frac", 0.0))),
        ctx.get("mem_n_seen", 0) / 15.0,
        ctx.get("mem_n_visited", 0) / 15.0,
        len(ctx.get("mem_carrying") or []) / 3.0,
    ], dtype=np.float32)
    return o, m, s


class LearnedPhi(nn.Module):
    def __init__(self, n_obs=2048, n_mem=1024, n_scalar=7, d=64, hid=256):
        super().__init__()
        self.obs = nn.Linear(n_obs, hid)
        self.mem = nn.Linear(n_mem, hid)       # <- memory latent
        self.sca = nn.Linear(n_scalar, 32)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hid + hid + 32, hid),
                                  nn.ReLU(), nn.Linear(hid, d))

    def forward(self, o, m, s):
        z = self.head(torch.cat([self.obs(o), self.mem(m), self.sca(s)], -1))
        return z / (z.norm(dim=-1, keepdim=True) + 1e-8)


class MuPsi(nn.Module):
    """Action-conditioned successor features over the LEARNED phi."""
    def __init__(self, d=64, d_act=32, hid=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + d_act, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, d))

    def forward(self, z, a):
        return self.net(torch.cat([z, a], -1))


class LearnedAffinity(nn.Module):
    """Learned metric on the predicted-consequence gap.

    It has its OWN objective (see ContextEncoder.fit_affinity): regress the gap
    between two occurrences' PREDICTED consequences onto the divergence of their
    ACTUALLY OBSERVED discounted future features.  Without that, the module sits
    downstream of the mu_psi TD loss but is never called by it -- so its weights
    stay at initialisation and the "learned metric" is a random projection. That
    was the state of the first `learned` arm.

    Target uses observed FEATURES, never reward or return, so the grouping stays
    reward-free and the LOO baseline argument survives.

    Symmetric by construction: the input is the per-dimension mean |P_u - P_v|.
    """
    def __init__(self, d=64, hid=96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    @staticmethod
    def gap_features(Pu, Pv):
        """Symmetric summary of two consequence sets -> 2d vector."""
        g = (Pu - Pv).abs()
        return torch.cat([g.mean(0), g.amax(0)], -1)

    def forward(self, feats):
        return torch.nn.functional.softplus(self.net(feats)).squeeze(-1)

    def dist(self, gap_feats):
        return self.forward(gap_feats)


class ContextEncoder:
    """Owns phi + mu_psi + affinity, and their joint reward-free TD training."""

    def __init__(self, d=64, d_act=32, lam=0.9, lr=1e-3, device="cpu", seed=0):
        torch.manual_seed(seed)
        self.dev = device
        self.phi = LearnedPhi(d=d).to(device)
        self.mu = MuPsi(d=d, d_act=d_act).to(device)
        self.aff = LearnedAffinity(d=d).to(device)
        self.phi_t = LearnedPhi(d=d).to(device); self.phi_t.load_state_dict(self.phi.state_dict())
        self.mu_t = MuPsi(d=d, d_act=d_act).to(device); self.mu_t.load_state_dict(self.mu.state_dict())
        self.opt = torch.optim.Adam(
            list(self.phi.parameters()) + list(self.mu.parameters()) + list(self.aff.parameters()), lr=lr)
        self.lam, self.d = lam, d

    def _t(self, x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=self.dev)

    def fit(self, trans, epochs=4, bs=256, verbose=False):
        """trans: dicts with o,m,s (now), a, o2,m2,s2 (next), a2 list, q2 weights, done.
        REWARD-FREE: nothing here reads a return."""
        if not trans:
            return float("nan")
        O, M, S = self._t([x["o"] for x in trans]), self._t([x["m"] for x in trans]), self._t([x["s"] for x in trans])
        A = self._t([x["a"] for x in trans])
        O2, M2, S2 = self._t([x["o2"] for x in trans]), self._t([x["m2"] for x in trans]), self._t([x["s2"] for x in trans])
        D = self._t([[float(x["done"])] for x in trans])
        maxA = max(len(x["a2"]) for x in trans)
        NA = np.zeros((len(trans), maxA, A.shape[1]), dtype=np.float32)
        NQ = np.zeros((len(trans), maxA), dtype=np.float32)
        for i, x in enumerate(trans):
            k = len(x["a2"])
            if k:
                NA[i, :k] = np.asarray(x["a2"], dtype=np.float32); NQ[i, :k] = np.asarray(x["q2"], dtype=np.float32)
        NA, NQ = self._t(NA), self._t(NQ)
        losses, step = [], 0
        for _ in range(epochs):
            perm = torch.randperm(len(trans), device=self.dev)
            for i in range(0, len(trans), bs):
                idx = perm[i:i + bs]; b = idx.shape[0]
                with torch.no_grad():
                    z2t = self.phi_t(O2[idx], M2[idx], S2[idx])
                    exp_mu = (self.mu_t(z2t.unsqueeze(1).expand(-1, maxA, -1).reshape(b * maxA, -1),
                                        NA[idx].reshape(b * maxA, -1)).reshape(b, maxA, self.d)
                              * NQ[idx].unsqueeze(-1)).sum(1)
                    y = z2t + self.lam * (1.0 - D[idx]) * exp_mu
                z = self.phi(O[idx], M[idx], S[idx])
                pred = self.mu(z, A[idx])
                loss = ((pred - y) ** 2).mean()
                self.opt.zero_grad(); loss.backward(); self.opt.step()
                losses.append(float(loss.detach())); step += 1
                if step % 100 == 0:
                    self.phi_t.load_state_dict(self.phi.state_dict())
                    self.mu_t.load_state_dict(self.mu.state_dict())
        self.phi_t.load_state_dict(self.phi.state_dict()); self.mu_t.load_state_dict(self.mu.state_dict())
        return float(np.mean(losses[-50:]))

    # ------------------------------------------------------------------ affinity
    @torch.no_grad()
    def observed_successor(self, traj_feats, lam=None):
        """Phi_t = sum_k lam^{k-1} phi(chi_{t+k}) from the ACTUALLY OBSERVED
        continuation of one trajectory.  Reward-free: features only.
        Uses the target phi so the regression target does not chase the encoder."""
        lam = self.lam if lam is None else lam
        O, M, S = (self._t([f[k] for f in traj_feats]) for k in range(3))
        Z = self.phi_t(O, M, S)
        out = torch.zeros_like(Z)
        acc = torch.zeros(Z.shape[1], device=self.dev)
        for t in range(len(traj_feats) - 1, -1, -1):
            acc = Z[t] + lam * acc
            out[t] = acc
        return out

    def fit_affinity(self, pair_data, epochs=3, bs=128):
        """Train d_aff(P_u,P_v) to predict ||Phi_u - Phi_v||, the divergence of the
        two occurrences' observed futures.  pair_data: (gap_feats, target) pairs."""
        if len(pair_data) < 16:
            return float("nan")
        X = torch.stack([g for g, _ in pair_data]).to(self.dev)
        y = self._t([t for _, t in pair_data])
        y = y / (y.mean() + 1e-8)                      # scale-free target
        opt = torch.optim.Adam(self.aff.parameters(), lr=1e-3)
        losses = []
        for _ in range(epochs):
            perm = torch.randperm(len(X), device=self.dev)
            for i in range(0, len(X), bs):
                idx = perm[i:i + bs]
                pred = self.aff(X[idx])
                loss = ((pred - y[idx]) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss.detach()))
        return float(np.mean(losses[-30:]))

    @torch.no_grad()
    def consequence_set(self, o, m, s, acts):
        """P_u: centred, scale-free set of predicted consequences over shared actions."""
        z = self.phi(self._t([o]), self._t([m]), self._t([s]))
        Z = z.expand(len(acts), -1)
        mu = self.mu(Z, self._t(acts))
        mu = mu - mu.mean(0, keepdim=True)
        n = mu.norm()
        return mu / n if n > 1e-8 else mu

    @torch.no_grad()
    def distance(self, Pu, Pv):
        return float(self.aff.dist(LearnedAffinity.gap_features(Pu, Pv)[None, :]).item())
