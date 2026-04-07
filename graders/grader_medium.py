"""
grader_medium.py — Grader for Task 2: Dual Variable Interaction

score ∈ [0.0, 1.0], deterministic.

Components mirror Task 1 but success_component requires handling both variables.
Belief accuracy is weighted more heavily because dual inference is harder.

    success_component     0.45
    efficiency_score      0.15
    belief_accuracy_score 0.30
    probe_penalty        -0.10
"""

from __future__ import annotations

from typing import Any, Dict


MAX_STEPS = 18


def grade(env_state: Dict[str, Any]) -> Dict[str, float]:
    hidden        = env_state.get("hidden_state", {})
    belief        = env_state.get("belief", [0.5, 0.5, 0.5])
    step_count    = env_state.get("step_count", MAX_STEPS)
    task_progress = env_state.get("task_progress", 0.0)
    probe_counts  = env_state.get("probe_counts", {})

    # ------------------------------------------------------------------
    # 1. Success component (0 to 0.45)
    # ------------------------------------------------------------------
    success = task_progress >= 0.95
    success_component = 0.45 if success else task_progress * 0.35

    # ------------------------------------------------------------------
    # 2. Efficiency (0 to 0.15)
    # ------------------------------------------------------------------
    steps_used_fraction = step_count / MAX_STEPS
    efficiency_score = max(0.0, 0.15 * (1.0 - steps_used_fraction))

    # ------------------------------------------------------------------
    # 3. Belief accuracy (0 to 0.30) — doubled weight vs easy task
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
    # 4. Redundant probe penalty (0 to -0.10)
    # ------------------------------------------------------------------
    redundant_probes = sum(max(cnt - 1, 0) for cnt in probe_counts.values())
    probe_penalty = -min(0.10, redundant_probes * 0.025)

    # ------------------------------------------------------------------
    # Final score
    # ------------------------------------------------------------------
    score = success_component + efficiency_score + belief_accuracy_score + probe_penalty
    score = max(0.0, min(1.0, score))

    return {
        "score":                 round(score, 4),
        "success_component":     round(success_component, 4),
        "efficiency_score":      round(efficiency_score, 4),
        "belief_accuracy_score": round(belief_accuracy_score, 4),
        "probe_penalty":         round(probe_penalty, 4),
        "details": {
            "task_progress":    task_progress,
            "step_count":       step_count,
            "success":          success,
            "redundant_probes": redundant_probes,
            "belief":           belief,
            "hidden_state":     hidden,
        },
    }
