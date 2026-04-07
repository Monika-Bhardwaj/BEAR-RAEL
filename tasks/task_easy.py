"""
task_easy.py — Task 1: Single Variable Diagnosis

Only ONE hidden variable is perturbed; the other two are fixed at benign values.
The agent must identify the single issue and resolve it.

Difficulty: EASY
    - Only one failure mode active
    - Generous step budget
    - Strong probe signal
"""

from __future__ import annotations

import random
from typing import Optional

from env.environment import BearRaelEnv, Action, Observation

TASK_NAME       = "single_variable_diagnosis"
MAX_STEPS       = 18
EASY_SEED_BASE  = 1000


def make_task_env(seed: int = EASY_SEED_BASE, variable_override: Optional[str] = None) -> BearRaelEnv:
    """
    Returns a reset environment for Task 1.

    Exactly one of the three hidden variables is set to a problematic value.
    The disturbed variable is chosen deterministically from the seed.
    """
    rng = random.Random(seed)
    choice = variable_override or rng.choice(["friction", "alignment", "stiffness"])

    kwargs: dict = dict(
        friction_level  = 0.3,   # benign default
        alignment_error = 0.01,  # benign default
        stiffness       = 0.7,   # benign default
    )

    if choice == "friction":
        kwargs["friction_level"] = rng.uniform(0.65, 0.95)   # HIGH friction
    elif choice == "alignment":
        kwargs["alignment_error"] = rng.uniform(0.028, 0.048) # BAD alignment
    else:  # stiffness
        kwargs["stiffness"] = rng.uniform(0.1, 0.35)          # LOW stiffness

    env = BearRaelEnv()
    env.reset(
        seed=seed,
        task='easy',  # short name for grader registry
        max_steps=MAX_STEPS,
        **kwargs,
    )
    # Store chosen variable for grader
    env._episode_info["disturbed_variable"] = choice
    return env


def describe() -> dict:
    return {
        "name":        TASK_NAME,
        "difficulty":  "easy",
        "max_steps":   MAX_STEPS,
        "description": (
            "Exactly one hidden variable (friction / alignment / stiffness) is "
            "set to a problematic value. The agent must identify which one via "
            "probing and then successfully complete the peg insertion."
        ),
        "success_criteria": "task_progress >= 0.95 within step budget",
    }
