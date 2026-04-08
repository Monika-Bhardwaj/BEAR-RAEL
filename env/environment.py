"""
environment.py — BEAR-RAEL OpenEnv environment.

Implements the full OpenEnv interface:
    reset(seed, task)  → Observation
    step(action)       → (Observation, float, bool, dict)
    state()            → dict
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from pydantic import Field
from models import Observation, Action, Reward

from env.dynamics import DynamicsEngine
from env.belief import BeliefState


# Models are now in models.py at root


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS = {
    "insert", "adjust_position", "increase_force",
    "probe_friction", "probe_alignment", "probe_stiffness",
    "commit_solution",
}

STEP_PENALTY         = -0.02
REDUNDANT_PROBE_PEN  = -0.10
INVALID_ACTION_PEN   = -0.15
SUCCESS_BONUS        =  1.00


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class BearRaelEnv:
    """
    BEAR-RAEL: Bayesian Embodied Autonomous Robotics Lab environment.

    The agent must diagnose hidden physical properties (friction, alignment,
    stiffness) via diagnostic probes and then execute the correct insertion
    strategy to complete the manipulation task.
    """

    DEFAULT_MAX_STEPS = 20

    def __init__(self) -> None:
        self._dynamics = DynamicsEngine(noise_scale=0.02)
        self._belief   = BeliefState()

        self._task: str       = "easy"
        self._max_steps: int  = self.DEFAULT_MAX_STEPS
        self._seed: int       = 42
        self._done: bool      = False
        self._last_action_str: str = "none"
        self._cumulative_reward: float = 0.0
        self._rewards_history: List[float] = []
        self._step_count: int = 0

        # Episode-level info for graders
        self._episode_info: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int = 42,
        task: str = "easy",
        max_steps: Optional[int] = None,
        # Allow task-level fixture overrides (used by task_*.py)
        friction_level:  Optional[float] = None,
        alignment_error: Optional[float] = None,
        stiffness:       Optional[float] = None,
    ) -> Observation:

        self._seed       = seed
        self._task       = task
        self._max_steps  = max_steps or self.DEFAULT_MAX_STEPS
        self._done       = False
        self._last_action_str = "none"
        self._cumulative_reward = 0.0
        self._rewards_history   = []
        self._step_count = 0

        self._dynamics.reset(
            seed=seed,
            friction_level=friction_level,
            alignment_error=alignment_error,
            stiffness=stiffness,
        )
        self._belief.reset()

        self._episode_info = {
            "task":          task,
            "seed":          seed,
            "hidden_state":  self._dynamics.hidden_state_dict(),
            "start_time":    time.time(),
        }

        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self._done:
            obs = self._build_observation()
            return obs, 0.0, True, {"error": "episode_already_done"}

        prev_belief = self._belief.as_list()
        prev_progress = self._dynamics.state.task_progress

        # Dispatch action
        reward_breakdown, error_msg = self._dispatch(action)

        self._step_count += 1
        self._dynamics.state.step_count = self._step_count

        # Progress reward
        curr_progress = self._dynamics.state.task_progress
        progress_delta = curr_progress - prev_progress
        reward_breakdown.progress_delta = progress_delta

        # Info gain bonus (only for probe actions)
        if action.action_type.startswith("probe_"):
            reward_breakdown.info_gain_bonus = round(
                self._belief.info_gain(prev_belief) * 0.5, 4
            )

        # Success check
        success = self._dynamics.is_success() or (
            action.action_type == "commit_solution"
            and self._dynamics.state.task_progress >= 0.75
        )
        if success:
            efficiency = max(1.0 - self._step_count / self._max_steps, 0.0)
            reward_breakdown.success_bonus = SUCCESS_BONUS + efficiency * 0.5

        # Episode termination
        self._done = success or self._step_count >= self._max_steps

        # Total reward
        total = (
            reward_breakdown.progress_delta * 3.0
            + reward_breakdown.info_gain_bonus
            + reward_breakdown.success_bonus
            + reward_breakdown.step_penalty
            + reward_breakdown.redundant_probe_penalty
            + reward_breakdown.invalid_action_penalty
        )
        # Clamp to [-1, 2] to keep scale sane
        total = max(-1.0, min(2.0, total))
        reward_breakdown.total = round(total, 4)

        self._cumulative_reward += total
        self._rewards_history.append(total)

        obs = self._build_observation()
        info = {
            "reward_breakdown": reward_breakdown.model_dump(),
            "success":          success,
            "error":            error_msg,
            "hidden_state":     self._dynamics.hidden_state_dict(),  # for grader
            "belief":           self._belief.as_list(),
        }

        return obs, total, self._done, info

    def state(self) -> dict:
        """Return full internal state (for debugging / grader access)."""
        s = self._dynamics.state
        return {
            "task":              self._task,
            "seed":              self._seed,
            "step_count":        self._step_count,
            "max_steps":         self._max_steps,
            "done":              self._done,
            "cumulative_reward": self._cumulative_reward,
            "task_progress":     s.task_progress,
            "position":          list(s.position),
            "hidden_state":      self._dynamics.hidden_state_dict(),
            "belief":            self._belief.as_list(),
            "probe_counts": {
                "probe_friction":   self._belief.probe_count("probe_friction"),
                "probe_alignment":  self._belief.probe_count("probe_alignment"),
                "probe_stiffness":  self._belief.probe_count("probe_stiffness"),
            },
            "episode_info":      self._episode_info,
        }

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: Action) -> tuple[Reward, Optional[str]]:
        reward = Reward(
            total=0.0,
            progress_delta=0.0,
            info_gain_bonus=0.0,
            success_bonus=0.0,
            step_penalty=STEP_PENALTY,
            redundant_probe_penalty=0.0,
            invalid_action_penalty=0.0,
        )
        error_msg = None

        atype  = action.action_type
        params = action.parameters or {}

        if atype not in VALID_ACTIONS:
            reward.invalid_action_penalty = INVALID_ACTION_PEN
            self._last_action_str = f"{atype}(INVALID)"
            return reward, f"unknown_action:{atype}"

        # ----- Task actions -----
        if atype == "insert":
            fm = float(params.get("force_magnitude", 1.0))
            pos, force, contact, _ = self._dynamics.apply_insert(force_magnitude=fm)
            self._last_action_str = f"insert(force_magnitude={fm:.2f})"

        elif atype == "adjust_position":
            # Accept dx/dy (canonical) AND delta_x/delta_y (LLM alias)
            _dx = params.get("dx") or params.get("delta_x") or params.get("x") or 0.0
            _dy = params.get("dy") or params.get("delta_y") or params.get("y") or 0.0
            dx = float(_dx)
            dy = float(_dy)
            pos, force, contact, _ = self._dynamics.apply_adjust_position(dx=dx, dy=dy)
            self._last_action_str = f"adjust_position(dx={dx:.3f},dy={dy:.3f})"

        elif atype == "increase_force":
            delta = float(params.get("delta", 0.3))
            pos, force, contact, _ = self._dynamics.apply_increase_force(delta=delta)
            self._last_action_str = f"increase_force(delta={delta:.2f})"

        # ----- Probe actions -----
        elif atype == "probe_friction":
            prev_count = self._belief.probe_count("probe_friction")
            force, signal = self._dynamics.probe_friction()
            self._belief.update_from_probe("probe_friction", signal)
            if prev_count > 0:
                reward.redundant_probe_penalty = REDUNDANT_PROBE_PEN
            self._last_action_str = f"probe_friction(signal={signal:.3f})"

        elif atype == "probe_alignment":
            prev_count = self._belief.probe_count("probe_alignment")
            force, signal = self._dynamics.probe_alignment()
            self._belief.update_from_probe("probe_alignment", signal)
            if prev_count > 0:
                reward.redundant_probe_penalty = REDUNDANT_PROBE_PEN
            self._last_action_str = f"probe_alignment(signal={signal:.3f})"

        elif atype == "probe_stiffness":
            prev_count = self._belief.probe_count("probe_stiffness")
            force, signal = self._dynamics.probe_stiffness()
            self._belief.update_from_probe("probe_stiffness", signal)
            if prev_count > 0:
                reward.redundant_probe_penalty = REDUNDANT_PROBE_PEN
            self._last_action_str = f"probe_stiffness(signal={signal:.3f})"

        # ----- Meta action -----
        elif atype == "commit_solution":
            self._last_action_str = "commit_solution()"

        return reward, error_msg

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        s     = self._dynamics.state
        flags = self._dynamics.compute_anomaly_flags()

        # Velocity approximated from last two force readings
        hist = s.force_history
        if len(hist) >= 2:
            vel = [hist[-1][i] - hist[-2][i] for i in range(3)]
        else:
            vel = [0.0, 0.0, 0.0]

        return Observation(
            position         = list(s.position),
            velocity         = vel,
            force            = hist[-1] if hist else [0.0, 0.0, 0.0],
            contact_detected = s.contact_detected,
            task_progress    = s.task_progress,
            anomaly_flags    = flags,
            last_action      = self._last_action_str,
            step_count       = self._step_count,
            belief           = self._belief.as_list(),
        )
