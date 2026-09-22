#!/usr/bin/env python3
"""Patch a verl-agent tree for full-coverage evaluation and per-episode dumps.

    census/apply_census_patch.py /workspace/baselines/verl-agent
    census/apply_census_patch.py /workspace/baselines/G2PO
    census/apply_census_patch.py --check /workspace/baselines/verl-agent

Idempotent: re-running on a patched tree reports "already applied" and changes
nothing. Run it once per tree -- grpo/gigpo/hgpo share baselines/verl-agent, g2po
has its own.

WHAT IT CHANGES (three files, all evaluation-only; no training path is touched)

1. env_package/alfworld/envs.py   ACG_ALF_CENSUS=1 pins worker i to game i.
2. env_package/webshop/envs.py    ACG_WS_CENSUS=1 walks the goal pool in order.
3. multi_turn_rollout/rollout_loop.py
                                  ACG_VAL_DUMP=<path> writes one JSON line per
                                  episode: task id, turns, success.

WHY (1) AND (2) ARE NEEDED

Neither environment hands out its held-out tasks exactly once. ALFWorld gives each
worker its own shuffled copy of the pool and draws from it; WebShop samples without
replacement WITHIN a reset but redraws independently on the next one. Either way an
N-episode pass over a pool of M lands on roughly M*(1-(1-1/M)^N) distinct tasks and
repeats the rest -- about 84 of ALFWorld's 140, about 63% of WebShop's 500.

That is not a cosmetic issue. Repeated tasks are counted twice, so the reported
number is a re-weighted average over a sample, not the split average, and the
weights change with the seed. It also means the *number of validation envs* silently
changes which games are played: 128 envs x 1 reset and 64 envs x 2 resets are
different samples, not two chunks of one.

With coverage pinned, every task is played exactly once and **sampling variance is
exactly zero**. There is then no draw to vary, so env.seed stops being the knob that
matters and the decoding seed (`+actor_rollout_ref.rollout.seed`) becomes the only
source of run-to-run difference.

WHY (3) IS NEEDED

verl only ever logs the MEAN success rate -- `_validate()` asserts the per-row values
are identical because they are a broadcast scalar. Without per-episode records a
census cannot be re-drawn offline, and a sampled evaluation has to be paid for in GPU
time. With the dump, `census/census_draw.py` reproduces any draw for free.
"""
import argparse
import os
import sys

# (relative path, marker that means "already patched", anchor, replacement)
EDITS = [
    (
        "agent_system/environments/env_package/alfworld/envs.py",
        "ACG_ALF_CENSUS",
        """        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        for i in range(self.num_processes):
            worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
            self.workers.append(worker)""",
        '''        # Create Ray remote actors instead of processes
        env_worker = ray.remote(**resources_per_worker)(AlfworldWorker)
        self.workers = []
        self.last_task_ids = None
        if os.environ.get("ACG_ALF_CENSUS", "") and not is_train:
            # CENSUS MODE: pin worker i to game i so the split is covered exactly
            # once. By default each worker shuffles the whole pool and draws from
            # it, so a 140-env pass lands on ~84 distinct games and repeats the
            # rest. AlfworldWorker does base_env.init_env(batch_size=1) and
            # init_env registers base_env.game_files, so handing worker i a
            # shallow copy whose game_files is [game_i] fixes its game.
            games = list(base_env.game_files)
            if self.num_processes != len(games):
                print(f"[alf-census] {self.num_processes} envs vs {len(games)} games "
                      f"-- set data.val_batch_size={len(games)} for exact coverage",
                      flush=True)
            assigned = [games[i % len(games)] for i in range(self.num_processes)]
            for i, g in enumerate(assigned):
                b = copy.copy(base_env)
                b.game_files = [g]
                b.num_games = 1
                self.workers.append(env_worker.remote(config, seed + i, b))
            # task dir + trial dir. The task dir ALONE is not unique -- the split
            # holds three tasks with two trials each, so 140 games carry only 137
            # distinct task-dir names and a colliding id silently merges two games.
            self.last_task_ids = ["/".join(os.path.normpath(g).split(os.sep)[-3:-1])
                                  for g in assigned]
        else:
            for i in range(self.num_processes):
                worker = env_worker.remote(config, seed + (i // self.group_n), base_env)
                self.workers.append(worker)''',
    ),
    (
        "agent_system/environments/env_package/webshop/envs.py",
        "ACG_WS_CENSUS",
        """    def reset(self):
        idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        idx = np.repeat(idx, self.group_n).tolist()""",
        '''    def reset(self):
        if os.environ.get("ACG_WS_CENSUS", ""):
            # CENSUS MODE: walk goal_idxs in order across successive resets so every
            # held-out goal is played exactly once. The default draw samples without
            # replacement WITHIN a reset but redraws independently on the next one,
            # so over N rounds it covers only ~63% of the pool and repeats the rest.
            n = len(self.goal_idxs)
            start = getattr(self, "_census_cursor", 0)
            idx = np.array([self.goal_idxs[(start + i) % n] for i in range(self.env_num)])
            self._census_cursor = (start + self.env_num) % n
        else:
            idx = self._rng.choice(self.goal_idxs, size=self.env_num, replace=False)
        # Which task each env is playing, for the per-episode dump.
        self.last_task_ids = [int(v) for v in np.asarray(idx).ravel()]
        idx = np.repeat(idx, self.group_n).tolist()''',
    ),
    (
        "agent_system/multi_turn_rollout/rollout_loop.py",
        "ACG_VAL_DUMP",
        """        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings""",
        '''        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards, 
                    episode_lengths=episode_lengths,
                    )

        _dump = os.environ.get("ACG_VAL_DUMP", "")
        if _dump:
            # Per-episode outcomes keyed by task identity. verl logs only the MEAN
            # success rate, so without this a census cannot be re-drawn offline.
            # Intended for val_only runs, where every rollout is a validation one.
            try:
                import json as _json
                _raw = getattr(envs, "envs", None)
                _ids = getattr(_raw, "last_task_ids", None)
                with open(_dump, "a") as _f:
                    for _i in range(len(episode_lengths)):
                        _row = {"task": (_ids[_i] if _ids is not None and _i < len(_ids)
                                         else None),
                                "turns": int(episode_lengths[_i])}
                        for _k, _v in success.items():
                            # Only per-episode series. ALFWorld's per-task-type keys
                            # hold one entry per episode OF THAT TYPE, so they are
                            # shorter than the batch and indexing them by batch
                            # position pairs a row with another episode's result.
                            # The type is recoverable from the task id anyway.
                            if len(_v) == len(episode_lengths):
                                _row[_k] = float(_v[_i])
                        _f.write(_json.dumps(_row) + "\\n")
            except Exception as _e:                      # never fail a run over logging
                print(f"[val-dump] failed: {_e}", flush=True)

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings''',
    ),
]

IMPORTS = [
    ("agent_system/environments/env_package/alfworld/envs.py", "import ray\n",
     "import copy\nimport os\nimport ray\n", "import copy"),
    ("agent_system/environments/env_package/webshop/envs.py", "import ray\n",
     "import os\nimport ray\n", "import os"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree", help="path to a verl-agent / G2PO checkout")
    ap.add_argument("--check", action="store_true", help="report status, change nothing")
    a = ap.parse_args()

    rc = 0
    for rel, marker, anchor, repl in EDITS:
        p = os.path.join(a.tree, rel)
        if not os.path.exists(p):
            print(f"  MISSING  {rel}"); rc = 1; continue
        s = open(p).read()
        if marker in s:
            print(f"  ok       {rel}  (already applied)"); continue
        if a.check:
            print(f"  TODO     {rel}"); rc = 1; continue
        if anchor not in s:
            print(f"  FAILED   {rel}: anchor not found -- upstream has changed, patch "
                  f"by hand (see this file's docstring)"); rc = 1; continue
        open(p, "w").write(s.replace(anchor, repl, 1))
        print(f"  patched  {rel}")

    for rel, anchor, repl, marker in IMPORTS:
        p = os.path.join(a.tree, rel)
        if not os.path.exists(p):
            continue
        s = open(p).read()
        if marker in s.split("\n\n")[0] or a.check:
            continue
        if anchor in s:
            open(p, "w").write(s.replace(anchor, repl, 1))
            print(f"  imports  {rel}")

    if not a.check:
        print("\nVerify with:  python -c \"import ast;ast.parse(open(F).read())\"  on each file,")
        print("then run scripts/census.sh.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
