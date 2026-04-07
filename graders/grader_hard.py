"""
grader_hard.py — Grader for Task 3: Full Debugging Under Noise

score ∈ [0.0, 1.0], deterministic.

Hard grader penalises over-probing much more aggressively and puts a higher
premium on efficiency. Belief accuracy is the largest single component because
reaching the correct conclusion under noise is the core challenge.

    success_component     0.40
    efficiency_score      0.20   (steep — every wasted step costs more)
    belief_accuracy_score 0.30
    probe_penalty        -0.15   (harsh redundancy penalty)
    noise_bonus           0.05   (extra credit: succeeded despite 3× noise)
"""

from __future__ import annotations

import math
from typing import Any, Dict


MAX_STEPS = 20


def grade(env_state: Dict[str, Any]) -> Dict[str, float]:
    hidden        = env_state.get("hidden_state", {})
    belief        = env_state.get("belief", [0.5, 0.5, 0.5])
    step_count    = env_state.get("step_count", MAX_STEPS)
    task_progress = env_state.get("task_progress", 0.0)
    probe_counts  = env_state.get("probe_counts", {})
    episode_info  = env_state.get("episode_info", {})
    noise_scale   = episode_info.get("noise_scale", 0.02)

    # ------------------------------------------------------------------
    # 1. Success component (0 to 0.40)
    # ------------------------------------------------------------------
    success = task_progress >= 0.95
    if success:
        # Exponential efficiency bonus on top of base
        step_fraction = step_count / MAX_STEPS
        success_component = 0.40 * (1.0 + 0.5 * math.exp(-3 * step_fraction))
        success_component = min(success_component, 0.40)
    else:
        success_component = task_progress * 0.30

    # ------------------------------------------------------------------
    # 2. Efficiency (0 to 0.20) — steeper penalty per step
    # ------------------------------------------------------------------
    steps_used_fraction = step_count / MAX_STEPS
    efficiency_score = max(0.0, 0.20 * (1.0 - steps_used_fraction) ** 1.5)

    # ------------------------------------------------------------------
    # 3. Belief accuracy (0 to 0.30)
    # ------------------------------------------------------------------
    true_high_friction  = bool(hidden.get("true_high_friction",  False))
    true_bad_alignment  = bool(hidden.get("true_bad_alignment",  False))
    true_low_stiffness  = bool(hidden.get("true_low_stiffness",  False))

    p_hf, p_ba, p_ls = belief[0], belief[1], belief[2]

    def match(p: float, truth: bool) -> float:
        return p if truth else (1.0 - p)

    raw_ba = (
        match(p_hf, true_high_friction)
        + match(p_ba, true_bad_alignment)
        + match(p_ls, true_low_stiffness)
    ) / 3.0
    belief_accuracy_score = 0.30 * raw_ba

    # ------------------------------------------------------------------
    # 4. Redundant probe penalty (0 to -0.15)
    # ------------------------------------------------------------------
    redundant_probes = sum(max(cnt - 1, 0) for cnt in probe_counts.values())
    probe_penalty = -min(0.15, redundant_probes * 0.04)

    # ------------------------------------------------------------------
    # 5. Noise robustness bonus (0 to 0.05)
    # ------------------------------------------------------------------
    noise_bonus = 0.05 if (success and noise_scale >= 0.05) else 0.0

    # ------------------------------------------------------------------
    # Final score
    # ------------------------------------------------------------------
    score = (
        success_component
        + efficiency_score
        + belief_accuracy_score
        + probe_penalty
        + noise_bonus
    )
    score = max(0.0, min(1.0, score))

    return {
        "score":                 round(score, 4),
        "success_component":     round(success_component, 4),
        "efficiency_score":      round(efficiency_score, 4),
        "belief_accuracy_score": round(belief_accuracy_score, 4),
        "probe_penalty":         round(probe_penalty, 4),
        "noise_bonus":           round(noise_bonus, 4),
        "details": {
            "task_progress":    task_progress,
            "step_count":       step_count,
            "success":          success,
            "redundant_probes": redundant_probes,
            "belief":           belief,
            "hidden_state":     hidden,
            "noise_scale":      noise_scale,
        },
    }
