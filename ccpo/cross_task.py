"""Batch-wide soft contextual peers, without task or observation matching."""
import numpy as np


def peer_records(features, targets, tasks, trajectories, tau_scale, min_traj=2,
                 value_targets=None):
    """Exact exponential kernel, excluding the query's complete trajectory.

    Retain occurrence weighting and trajectory-level effective support. A single
    distance matrix avoids the old Python loop over every query/peer pair. At the
    ALFWorld ceiling (16 tasks x 8 rollouts x 50 turns), float64 distances take
    313 MiB; bandwidth selection temporarily adds half that much. No top-k,
    approximate neighbors, task prior, or task fallback is used.
    """
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    tasks = np.asarray(tasks).astype(str)
    trajectories = np.asarray(trajectories).astype(str)
    if not np.isfinite(tau_scale) or tau_scale <= 0:
        raise ValueError('Cross-task soft grouping requires a finite positive tau scale')
    n = len(targets)
    _, traj_codes = np.unique(trajectories, return_inverse=True)
    task_names, task_codes = np.unique(tasks, return_inverse=True)
    n_traj = len(np.unique(traj_codes))
    if n_traj < max(2, min_traj):
        return [], dict(E_w=float('nan'), tau=float('nan'),
                        phi_rel_corr=float('nan'), phi_rel_slope=float('nan'))

    # In-place Gram distances avoid an (n,n,d) temporary.
    distances = features @ features.T
    distances *= -2
    square_norm = np.einsum('ij,ij->i', features, features)
    distances += square_norm[:, None]
    distances += square_norm[None, :]
    np.maximum(distances, 0., out=distances)
    np.sqrt(distances, out=distances)
    # Preserve the existing bandwidth rule: all distinct row pairs in the pool,
    # including same-trajectory pairs. Exclusion applies to the readout.
    upper = np.concatenate([distances[i, i+1:] for i in range(n-1)])
    median = float(np.median(upper, overwrite_input=True))
    del upper
    tau = (median if median > 1e-9 else 1.) * max(tau_scale, 1e-6)

    records = []
    weight_sum = 0.
    pair_count = 0
    # Streaming moments for the existing distance/target-difference diagnostic.
    sx = sy = sxx = syy = sxy = 0.
    for i in range(n):
        peers = np.flatnonzero(traj_codes != traj_codes[i])
        d = distances[i, peers]
        weight_sum += float(np.exp(-d / tau).sum())
        pair_count += len(peers)
        # A common factor cancels. Subtract the nearest distance to prevent all
        # weights underflowing when the kernel is narrow.
        weights = np.exp(-(d - d.min()) / tau)
        weights /= weights.sum()
        labels = targets[peers]
        baseline = float(weights @ labels)
        trajectory_mass = np.bincount(traj_codes[peers], weights=weights, minlength=n_traj)
        effective = 1. / float(trajectory_mass @ trajectory_mass)
        support = float(n_traj - 1)
        variance = float(np.var(labels))
        records.append(dict(i=i, lvl=0, bhash='cross_task', b_loo=baseline,
            b_obs=float(labels.mean()), s2=variance, ne=effective, J=support,
            var_gain=max(1./effective - 1./support, 0.), rho=0.,
            sig=float(np.sqrt(max(float(weights @ ((labels-baseline)**2)), 0.))),
            peer_rows=float(len(peers)), self_mass=0., same_traj_mass=0.,
            cross_task_mass=float(weights[task_codes[peers] != task_codes[i]].sum()),
            peer_tasks=float(np.count_nonzero(np.bincount(task_codes[peers], minlength=len(task_names)))), kernel_tau=tau,
            value_loo=None if value_targets is None else float(weights @ value_targets[peers]),
            value_obs=None if value_targets is None else float(value_targets[peers].mean())))
        x = distances[i, i+1:]
        y = abs(targets[i+1:] - targets[i])
        sx += x.sum(); sy += y.sum()
        sxx += x @ x; syy += y @ y; sxy += x @ y
    count = n * (n-1) / 2
    vx = max(sxx - sx*sx/count, 0.)
    vy = max(syy - sy*sy/count, 0.)
    cov = sxy - sx*sy/count
    informative = n > 3 and vx/count > 1e-18 and vy/count > 1e-18
    return records, dict(E_w=weight_sum/pair_count, tau=tau,
        phi_rel_corr=float(cov / np.sqrt(vx*vy)) if informative else float('nan'),
        phi_rel_slope=float(cov / vx) if informative else float('nan'))
