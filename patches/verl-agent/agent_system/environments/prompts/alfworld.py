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

# --------------------- ALFWorld --------------------- #
ALFWORLD_TEMPLATE_NO_HIS = """
You are an expert agent operating in the ALFRED Embodied Environment.
Your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation, concisely. This reasoning process MUST be enclosed within <think> </think> tags. 
Your response must contain BOTH parts: first a <think> block with your concise reasoning, then an <action> block with one action copied EXACTLY, character for character, from the admissible actions list. Do not rephrase, shorten, or invent actions. Never omit the <action> part. Example response: <think>I need to find an apple first; the fridge is a likely place, so I will check it.</think><action>go to fridge 1</action>
"""

ALFWORLD_TEMPLATE = """
You are an expert agent operating in the ALFRED Embodied Environment. Your task is to: {task_description}
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}
Your admissible actions of the current situation are: [{admissible_actions}].

Now it's your turn to take an action.
You should first reason step-by-step about the current situation, concisely. This reasoning process MUST be enclosed within <think> </think> tags. 
Your response must contain BOTH parts: first a <think> block with your concise reasoning, then an <action> block with one action copied EXACTLY, character for character, from the admissible actions list. Do not rephrase, shorten, or invent actions. Never omit the <action> part. Example response: <think>I need to find an apple first; the fridge is a likely place, so I will check it.</think><action>go to fridge 1</action>
"""