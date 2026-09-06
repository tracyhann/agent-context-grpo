# ccpo-nextnode-20260906

**Question.** `ccpo-base` showed the estimator is inert: conditioning the
*baseline* on phi changes it by less than sampling noise (lambda = 0.000,
`effect_rel` = 0.000, `r_vs_gigpo` = +0.977), and not for want of support — ~5
effective neighbours, 90% of samples credited. Does changing the **target** from
the step's own return-to-go to G2PO's successor node value produce signal where
changing the baseline could not?

**Why this and not more support.** `d = b_obs - b_LOO` reweights a handful of the
*same* neighbour returns. Return-to-go on ALFWorld is near-binary (0 or 10), so
`Var(d) = s^2 (1/n_eff - 1/J)` is large and reweighting five high-variance numbers
cannot move their mean beyond that noise. `V(next node)` is pooled over every
sibling visit to that node, so `s^2` is far smaller, `Var(d)` shrinks, and the same
relative reweighting becomes detectable. The write-up's own argument, now with a
measurement behind it: conditioning the baseline cannot remove noise living in the
target.

**Arm.** `ccpo_target=nextnode`, `ccpo_shrink=eb_hier`. Differs from
`ccpo-base-20260906` in `ccpo_target` alone.

**Watch, in order.**
1. `ccpo/lam_u_mean` and `ccpo/effect_rel`. If still 0, the method does not work on
   this benchmark under any variant reachable here, and that is the result.
2. `ccpo/r_vs_g2po` should now be high (the target is G2PO's) and
   `ccpo/r_vs_gigpo` should fall well below +0.977.
3. `response_length/mean` — `V(g)` carries trajectory-length information through
   `gamma^(T-t)`, so it could reward shorter or longer episodes. `ccpo/acc_len_corr`
   is the direct check.
4. Only then the success rate.

**Result.** _pending_

**Reading.** _pending_
