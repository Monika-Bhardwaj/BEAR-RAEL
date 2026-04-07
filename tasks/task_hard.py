"""
task_hard.py — Task 3: Full Debugging Under Noise

ALL three hidden variables vary, and observation noise is amplified.
The agent must perform efficient diagnosis (minimal probes) and act fast.

Difficulty: HARD
    - All three failure modes potentially active
    - Tight step budget (penalises over-probing severely)
    - Amplified sensor noise degrades probe signal quality
    - True test of belief-based adaptive strategy
"""

from __future__ import annotations

import random

from env.environment import BearRaelEnv
from env.dynamics import DynamicsEngine

TASK_NAME        = "full_debugging_noisy"
MAX_STEPS        = 22          # same budget, but 3 problems to solve
HARD_SEED_BASE   = 3000
NOISE_SCALE      = 0.06        # 3× default noise


def make_task_env(seed: int = HARD_SEED_BASE) -> BearRaelEnv:
    """
    Returns a reset environment for Task 3.

    All three variables are independently sampled from their full ranges
    (they may or may not be problematic — full uncertainty from the start).
    Noise is amplified.
    """
    rng = random.Random(seed)

    # All variables sampled from their FULL ranges — agent has no prior knowledge
    friction_level  = rng.uniform(0.1, 1.0)
    alignment_error = rng.uniform(0.0, 0.05)
    stiffness       = rng.uniform(0.1, 1.0)

    env = BearRaelEnv()
    # Inject higher noise before reset
    env._dynamics = DynamicsEngine(noise_scale=NOISE_SCALE)

    env.reset(
        seed=seed,
        task='hard',  # short name for grader registry
        max_steps=MAX_STEPS,
        friction_level  = friction_level,
        alignment_error = alignment_error,
        stiffness       = stiffness,
    )
    env._episode_info["noise_scale"] = NOISE_SCALE
    return env


def describe() -> dict:
    return {
        "name":        TASK_NAME,
        "difficulty":  "hard",
        "max_steps":   MAX_STEPS,
        "description": (
            "All three hidden variables (friction, alignment, stiffness) are "
            "independently sampled from their full ranges, and sensor noise is "
            "amplified 3×. The agent must use minimal probing to update beliefs "
            "efficiently, then execute the correct insertion strategy before the "
            "step budget is exhausted."
        ),
        "success_criteria": "task_progress >= 0.95 within step budget, bonus for efficiency",
    }
