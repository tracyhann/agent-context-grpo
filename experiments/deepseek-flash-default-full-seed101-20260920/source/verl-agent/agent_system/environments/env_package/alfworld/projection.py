# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List
import re

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    originals = list(actions)
    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # check reasoning format. Qwen3 with enable_thinking=False carries the
        # opening <think> block in the PROMPT (empty prefill), so the response
        # cannot contain a literal <think> tag - measured 0/32 at temp 1.0 while
        # 32/32 produced valid <action> blocks. Requiring response-side think
        # tags would pin valid_action_ratio to exactly 0 and RL could never
        # bootstrap. Accept either the classic <think>...</think> pair or the
        # prefill convention (no opening tag). Identical for every arm.
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx != -1 and think_end_idx == -1:
            valids[i] = 0   # opened a think block but never closed it

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    if not getattr(alfworld_projection, "_logged", 0):
        alfworld_projection._logged = 1
        import re as _re
        n = len(actions)
        has_tag = sum(1 for i in range(n) if valids[i] or "<action" in (actions[i] or ""))
        admiss = sum(1 for i in range(n) if valids[i] and actions[i] in [a.lower() for a in (action_pools[i] or [])])
        fails = [i for i in range(n) if not valids[i]]
        bad = [(i, originals[i][:70], originals[i][-25:]) for i in fails[:4]]
        print(f"[proj-debug] n={n} parser_valid={sum(valids)} admissible={admiss} | fail idx={fails[:12]}", flush=True)
        for b in bad:
            print(f"[proj-debug] FAIL i={b[0]} head={b[1]!r} tail={b[2]!r}", flush=True)
    return actions, valids
