"""
task_medium.py — Task 2: Dual Variable Interaction

TWO hidden variables vary simultaneously, creating confounded failure signals.
The agent must distinguish between causes and choose the correct strategy.

Difficulty: MEDIUM
    - Two failure modes active at once
    - Moderate step budget
    - Probe signals partially overlap — agent must reason jointly
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

from env.environment import BearRaelEnv

TASK_NAME       = "dual_variable_interaction"
MAX_STEPS       = 20
MEDIUM_SEED_BASE = 2000


# Which pairs of variables can be simultaneously problematic
VARIABLE_PAIRS = [
    ("friction",   "alignment"),
    ("friction",   "stiffness"),
    ("alignment",  "stiffness"),
]


def make_task_env(seed: int = MEDIUM_SEED_BASE, pair_override: Optional[Tuple[str, str]] = None) -> BearRaelEnv:
    """
    Returns a reset environment for Task 2.

    Two variables are set to problematic values; the specific pair is chosen
    deterministically from the seed.
    """
    rng = random.Random(seed)
    pair = pair_override or rng.choice(VARIABLE_PAIRS)

    kwargs: dict = dict(
        friction_level  = 0.3,
        alignment_error = 0.01,
        stiffness       = 0.7,
    )

    for var in pair:
        if var == "friction":
            kwargs["friction_level"]  = rng.uniform(0.65, 0.90)
        elif var == "alignment":
            kwargs["alignment_error"] = rng.uniform(0.028, 0.045)
        else:
            kwargs["stiffness"]       = rng.uniform(0.12, 0.35)

    env = BearRaelEnv()
    env.reset(
        seed=seed,
        task='medium',  # short name for grader registry
        max_steps=MAX_STEPS,
        **kwargs,
    )
    env._episode_info["disturbed_variables"] = list(pair)
    return env


def describe() -> dict:
    return {
        "name":        TASK_NAME,
        "difficulty":  "medium",
        "max_steps":   MAX_STEPS,
        "description": (
            "Two hidden variables are simultaneously set to problematic values "
            "(e.g. high friction AND bad alignment). The agent must probe both "
            "dimensions, reason about their interaction, and select actions that "
            "address both failure modes to complete the insertion."
        ),
        "success_criteria": "task_progress >= 0.95 within step budget",
    }
