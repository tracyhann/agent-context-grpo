"""Context compaction: a compact, domain-agnostic digest of an episode's history.

One representation serves three consumers:
  * the policy   -- the digest goes in the prompt, so the agent stops re-deriving
                    state inside <think>
  * phi          -- a block hashed from the digest varies WITHIN a hard bucket,
                    where the observation block is constant by construction and
                    the four history scalars are blind to hidden state
  * the gate     -- a coarse state signature refines conflated buckets

Nothing here parses domain vocabulary. The two primitives are:

  null action   an action that left the observation unchanged (obs_{t+1} == obs_t).
                In any environment a failed/no-op action leaves the world as it was,
                so this needs no "Nothing happens" regex.

  state proxy   in a deterministic environment reached from a fixed start, the set
                of SUCCESSFUL (non-null) actions is a sufficient statistic for the
                current state. Order-invariance is a deliberate coarsening: it keeps
                refined buckets populated.
"""

import hashlib
from typing import Any, Dict, List, Optional, Sequence

# The digest is the one expendable block in the prompt, so it is explicitly
# delimited: when the chat-templated prompt overruns max_prompt_length the
# rollout collector trims digest lines from the end (least valuable: null-effect
# and oldest, given the sort in build_digest) instead of losing the whole step.
DIGEST_HEADER = "What you have already tried and found:\n"
DIGEST_FOOTER = "(end of summary)"


def _norm(s: str) -> str:
    return " ".join(str(s).split()).strip()


def pair_history(records: Sequence[Dict[str, Any]],
                 obs_key: str = "text_obs",
                 action_key: str = "action") -> List[Dict[str, Any]]:
    """Turn [(obs_t, action_t), ...] into [(action_t -> obs_{t+1}, was_null), ...]."""
    out = []
    for i in range(len(records) - 1):
        before = _norm(records[i].get(obs_key, ""))
        after = _norm(records[i + 1].get(obs_key, ""))
        action = _norm(records[i].get(action_key, ""))
        if not action:
            continue
        out.append({"action": action, "outcome": after, "null": after == before})
    return out


def state_signature(records: Sequence[Dict[str, Any]],
                    obs_key: str = "text_obs",
                    action_key: str = "action") -> str:
    """Coarse hash of the set of successful actions -- the generic state proxy."""
    eff = {p["action"] for p in pair_history(records, obs_key, action_key)
           if not p["null"]}
    if not eff:
        return "s0"
    h = hashlib.sha1("|".join(sorted(eff)).encode()).hexdigest()[:12]
    return f"s{h}"


def build_digest(records: Sequence[Dict[str, Any]],
                 budget_tokens: int = 512,
                 tokenizer: Optional[Any] = None,
                 obs_key: str = "text_obs",
                 action_key: str = "action",
                 max_outcome_chars: int = 160) -> str:
    """Compact the history into a budgeted digest string.

    Priority when evicting: informative discoveries are kept over failures, and
    recent entries over old ones. Failures are collapsed with a count, which is
    what keeps a truncation-induced run of no-ops from flushing out real findings.
    """
    pairs = pair_history(records, obs_key, action_key)
    if not pairs:
        return ""

    merged: Dict[tuple, Dict[str, Any]] = {}
    for order, p in enumerate(pairs):
        key = (p["action"], p["outcome"])
        slot = merged.get(key)
        if slot is None:
            merged[key] = {"count": 1, "order": order, **p}
        else:
            slot["count"] += 1
            slot["order"] = order  # most recent occurrence decides recency

    entries = list(merged.values())
    # EVICTION priority: informative before no-effect, recent before old. This
    # decides what survives the budget, NOT the order it is shown in -- the two are
    # different concerns and conflating them meant the agent read its own history
    # backwards, most recent action first, which inverts the causality of a
    # sequential plan.
    entries.sort(key=lambda e: (e["null"], -e["order"]))

    def render(e: Dict[str, Any]) -> str:
        rep = f" (x{e['count']})" if e["count"] > 1 else ""
        if e["null"]:
            return f"- {e['action']} -> no effect{rep}"
        return f"- {e['action']} -> {e['outcome'][:max_outcome_chars]}{rep}"

    def n_tokens(text: str) -> int:
        if tokenizer is None:
            return max(1, len(text) // 4)  # rough fallback, only for tests
        return len(tokenizer(text).input_ids)

    header = DIGEST_HEADER
    kept: List[tuple] = []
    used = n_tokens(header)
    for e in entries:
        line = render(e)
        cost = n_tokens(line) + 1
        if used + cost > budget_tokens:
            continue  # skip this one, a later cheaper line may still fit
        kept.append((e["order"], line))
        used += cost
    if not kept:
        return ""
    # Render CHRONOLOGICALLY. What survived the budget was chosen by priority
    # above; the agent still has to read it as a narrative of what it did.
    kept = [ln for _, ln in sorted(kept, key=lambda t: t[0])]
    return header + "\n".join(kept) + "\n" + DIGEST_FOOTER


def fits(prompt: str, tokenizer: Any, max_tokens: int, margin: int = 64) -> bool:
    """Guard used by the degradation ladder in the prompt builder."""
    return len(tokenizer(prompt).input_ids) <= max_tokens - margin
