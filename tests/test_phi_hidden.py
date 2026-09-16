#!/usr/bin/env python3
"""Verification for the hidden-state phi wiring.

Test 1 (CPU): the estimator takes phi_feats, whitens batch-wide, and produces a
different advantage from the bag-of-words path on the same returns.

Test 2 (1 GPU): the forward hook on the final norm yields exactly the hidden
state at the LAST PROMPT TOKEN -- position (seqlen - response_length - 1) -- and
agrees with output_hidden_states[-1] computed independently. This is the index
that would silently misattribute every affinity vector if it were off by one.
"""
import os
import sys

# The Rust tokenizer builds a rayon pool sized from nproc; on a box whose
# cgroup pid budget is already spent by a training run that fails outright.
os.environ.setdefault("RAYON_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "verl-agent"))
sys.path.insert(0, os.path.join(ROOT, "docker", "fa_stub"))   # flash-attn has no sm_120 build


def _default_ckpt():
    """The base model this repo trains, or ACG_TEST_CKPT to point elsewhere."""
    env = os.environ.get("ACG_TEST_CKPT")
    if env:
        return env
    snap = os.path.join(os.environ.get("HF_HOME", os.path.join(ROOT, "hf")), "hub",
                        "models--Qwen--Qwen2.5-1.5B-Instruct", "snapshots")
    if os.path.isdir(snap):
        return os.path.join(snap, sorted(os.listdir(snap))[0])
    return "Qwen/Qwen2.5-1.5B-Instruct"


CKPT = _default_ckpt()


def test_estimator():
    """A conflated bucket: identical observation text, two latent states with
    different returns. This is the situation the method exists for."""
    from ccpo.core_ccpo import FrozenPhi, ccpo_step_advantage, whiten_feats
    rng = np.random.default_rng(0)
    n, d = 48, 64
    traj = np.array([f"t{i // 8}" for i in range(n)])
    uid = np.array(["task0"] * n)
    obs = np.array([f"you are in room {i % 3}. there is a table." for i in range(n)])
    mask = torch.ones(n, 12)

    latent = (np.arange(n) // 3) % 2          # hidden state, invisible in the text
    G = torch.tensor(latent * 1.0 + rng.normal(size=n) * 0.1, dtype=torch.float32)

    # Anisotropy modelled the way it appears in decoder embeddings: the leading
    # K directions carry register (~100x the variance) and the state signal sits
    # BELOW them. That ordering is the whole premise -- whitening at k removes
    # the top k, so the fixture needs at least k nuisance directions or the test
    # is asking whitening to preserve something it is defined to remove.
    K = 3
    feats = rng.normal(size=(n, d)) * 0.05
    for j in range(K):
        v = np.zeros(d); v[j] = 1.0
        feats = feats + rng.normal(size=n)[:, None] * v * 10.0
    v_state = np.zeros(d); v_state[K] = 1.0
    feats = feats + latent[:, None] * v_state * 1.0

    def sep(F):
        F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-9)
        D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=-1)
        same = D[(latent[:, None] == latent[None, :]) & ~np.eye(n, dtype=bool)]
        diff = D[latent[:, None] != latent[None, :]]
        return diff.mean() / max(same.mean(), 1e-9)

    r_raw, r_wht = sep(feats), sep(whiten_feats(feats))
    W = whiten_feats(feats)
    assert abs(np.linalg.norm(W, axis=1).mean() - 1.0) < 1e-6, "not L2-normalised"
    print(f"  d(diff)/d(same): raw {r_raw:.3f} -> whitened {r_wht:.3f}  "
          f"{'OK (whitening recovers the state direction)' if r_wht > r_raw else 'FAIL'}")

    common = dict(step_rewards=G, response_mask=mask, anchor_obs=obs,
                  index=uid, traj_index=traj, phi=FrozenPhi(), return_diag=True)
    a_bow, dg_bow = ccpo_step_advantage(**common)
    a_hid, dg_hid = ccpo_step_advantage(**common, phi_feats=torch.tensor(feats))
    for tag, dg in (("bow", dg_bow), ("hidden", dg_hid)):
        print(f"  {tag:<6} phi={dg['phi_mode']:<6} rho={dg['rho']:.2f} "
              f"lam={dg['lam_u_mean']:.3f} E[w]={dg['E_w']:.3f} "
              f"effect={dg['effect_mean']:.4f} r_vs_gigpo={dg['r_vs_gigpo']:.3f}")
    assert dg_hid["phi_mode"] == "hidden", "phi_feats ignored"
    assert dg_bow["phi_mode"] == "bow"

    # The point of the whole method. A baseline that sees the latent state
    # predicts the state-conditional return, so the state-driven component drops
    # OUT of the advantage and only the within-state deviation is credited. The
    # gap between the two latent groups' mean advantage should therefore SHRINK:
    # a blind baseline leaves A ~ latent - pooled_mean, i.e. a gap of ~1.0.
    def gap(a):
        a = a.numpy()
        return abs(a[latent == 1].mean() - a[latent == 0].mean())
    g_bow, g_hid = gap(a_bow), gap(a_hid)
    print(f"  latent-group gap in A: bow {g_bow:.4f} -> hidden {g_hid:.4f}  "
          f"{'OK (state component removed)' if g_hid < g_bow else 'FAIL'}")

    _, dg0 = ccpo_step_advantage(**common, phi_feats=torch.tensor(feats), rho=0.0)
    print(f"  rho=0 -> lam {dg0['lam_u_mean']:.4f}  "
          f"{'OK (exact fallback to the uniform bucket baseline)' if dg0['lam_u_mean'] < 1e-6 else 'FAIL'}")
    return r_wht > r_raw and g_hid < g_bow and dg0["lam_u_mean"] < 1e-6


def test_hook():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class Stub:
        """Minimal stand-in exposing the two methods under test."""
        use_remove_padding = False
        from verl.workers.actor.dp_actor import DataParallelPPOActor as _A
        _acg_final_norm = _A._acg_final_norm
        _acg_hidden_hook = _A._acg_hidden_hook

    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    stub = Stub(); stub.actor_module = model

    mod = stub._acg_final_norm()
    print(f"  final norm resolved: {type(mod).__name__ if mod is not None else None}")
    assert mod is not None, "could not locate the final norm"

    prompts = ["You are in the kitchen. You see a fridge 1 and a cabinet 2.",
               "You are in the middle of a room holding an apple."]
    responses = ["<action>go to fridge 1</action>", "<action>open cabinet 2</action>"]
    penc = tok(prompts, return_tensors="pt", padding=True, padding_side="left")
    renc = tok(responses, return_tensors="pt", padding=True, padding_side="right")
    input_ids = torch.cat([penc["input_ids"], renc["input_ids"]], 1).cuda()
    attn = torch.cat([penc["attention_mask"], renc["attention_mask"]], 1).cuda()
    response_length = renc["input_ids"].shape[1]

    h = mod.register_forward_hook(stub._acg_hidden_hook)
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn,
                    output_hidden_states=True, use_cache=False)
    h.remove()

    hooked = stub._acg_hook_out
    pos = hooked.shape[1] - response_length - 1
    pooled = hooked[:, pos, :].float().cpu()
    reference = out.hidden_states[-1][:, pos, :].float().cpu()
    err = float((pooled - reference).abs().max())
    print(f"  seqlen {hooked.shape[1]}, prompt {penc['input_ids'].shape[1]}, "
          f"response {response_length} -> pooled position {pos}")
    print(f"  max |hook - hidden_states[-1]| at that position = {err:.2e}  "
          f"{'OK' if err < 1e-3 else 'FAIL'}")

    last_prompt_tok = tok.decode(input_ids[0, pos])
    first_resp_tok = tok.decode(input_ids[0, pos + 1])
    print(f"  token at pos   = {last_prompt_tok!r}  (last prompt token)")
    print(f"  token at pos+1 = {first_resp_tok!r}  (first response token)")
    ok_index = pos == penc["input_ids"].shape[1] - 1
    print(f"  index identity: pos == prompt_len - 1  {'OK' if ok_index else 'FAIL'}")
    return err < 1e-3 and ok_index


if __name__ == "__main__":
    print("test 1: estimator accepts phi_feats and whitens")
    t1 = test_estimator()
    print("\ntest 2: hook pools the last prompt token")
    t2 = test_hook() if torch.cuda.is_available() else None
    print(f"\ntest 1 {'PASS' if t1 else 'FAIL'}"
          f" | test 2 {'PASS' if t2 else ('SKIPPED (no GPU)' if t2 is None else 'FAIL')}")
    sys.exit(0 if t1 and t2 is not False else 1)
