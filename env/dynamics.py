"""
dynamics.py — Physics simulation for BEAR-RAEL.

Hidden state variables (never directly observable):
    friction_level  ∈ [0.1, 1.0]
    alignment_error ∈ [0.0, 0.05]
    stiffness       ∈ [0.1, 1.0]

Exposed only through force/position/contact signals and probe responses.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Hidden physical state
# ---------------------------------------------------------------------------

@dataclass
class PhysicalState:
    friction_level:  float = 0.5
    alignment_error: float = 0.02
    stiffness:       float = 0.5

    # Mutable world state (observable indirectly)
    position:        List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    velocity:        List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    task_progress:   float       = 0.0
    step_count:      int         = 0

    # Contact bookkeeping
    contact_detected: bool       = False
    consecutive_contact: int     = 0
    force_history:   List[List[float]] = field(default_factory=list)

    # Thresholds (fixed physics constants)
    FRICTION_HIGH_THRESH:   float = 0.6
    ALIGNMENT_BAD_THRESH:   float = 0.025
    STIFFNESS_LOW_THRESH:   float = 0.4
    MAX_PROGRESS:           float = 1.0
    INSERTION_DEPTH:        float = 0.10   # metres


# ---------------------------------------------------------------------------
# Noise helper
# ---------------------------------------------------------------------------

def _gauss(mean: float, std: float, rng: random.Random) -> float:
    return rng.gauss(mean, std)


# ---------------------------------------------------------------------------
# Dynamics engine
# ---------------------------------------------------------------------------

class DynamicsEngine:
    """
    Computes observable signals given hidden physical state and action.

    All randomness routed through a seeded Random instance so that
    deterministic reset(seed) is possible.
    """

    def __init__(self, noise_scale: float = 0.02):
        self.noise_scale = noise_scale
        self._rng = random.Random(42)
        self.state: PhysicalState = PhysicalState()

    # ------------------------------------------------------------------
    # Seeded reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int = 42,
        friction_level:  Optional[float] = None,
        alignment_error: Optional[float] = None,
        stiffness:       Optional[float] = None,
    ) -> None:
        self._rng = random.Random(seed)
        rng = self._rng

        self.state = PhysicalState(
            friction_level  = friction_level  if friction_level  is not None else rng.uniform(0.1, 1.0),
            alignment_error = alignment_error if alignment_error is not None else rng.uniform(0.0, 0.05),
            stiffness       = stiffness       if stiffness       is not None else rng.uniform(0.1, 1.0),
            position        = [0.0, 0.0, 0.0],
            velocity        = [0.0, 0.0, 0.0],
            task_progress   = 0.0,
            step_count      = 0,
        )

    # ------------------------------------------------------------------
    # Task actions
    # ------------------------------------------------------------------

    def apply_insert(self, force_magnitude: float = 1.0) -> Tuple[List[float], List[float], bool, float]:
        """
        Attempt downward insertion.  Returns (position, force_vec, contact, progress_delta).
        """
        s = self.state
        n = self.noise_scale

        # Effective downward force reduced by friction
        friction_resistance = s.friction_level * force_magnitude * 0.8
        effective_force_z   = max(force_magnitude - friction_resistance, 0.05)

        # Alignment error creates lateral drift and resisting lateral force
        drift_x = s.alignment_error * _gauss(1.0, 0.1, self._rng) * force_magnitude
        drift_y = s.alignment_error * _gauss(0.5, 0.1, self._rng) * force_magnitude

        # Stiffness affects deformation (how much z moves per unit force)
        dz = (effective_force_z / (s.friction_level + 0.1)) * s.stiffness * 0.02
        dz = max(dz, 0.0)

        s.position[0] += drift_x + _gauss(0.0, n, self._rng)
        s.position[1] += drift_y + _gauss(0.0, n, self._rng)
        s.position[2] += dz

        # Force vector
        force_z = -effective_force_z + _gauss(0.0, n * 0.5, self._rng)
        force_x = s.alignment_error * 10.0 * _gauss(1.0, 0.2, self._rng)
        force_y = s.alignment_error *  5.0 * _gauss(1.0, 0.2, self._rng)
        force = [force_x, force_y, force_z]

        contact = s.position[2] > 0.005
        s.contact_detected = contact

        # Progress toward insertion depth
        prev_progress = s.task_progress
        s.task_progress = min(s.position[2] / s.INSERTION_DEPTH, 1.0)
        progress_delta = s.task_progress - prev_progress

        s.force_history.append(force)
        s.step_count += 1
        return list(s.position), force, contact, progress_delta

    def apply_adjust_position(self, dx: float = 0.0, dy: float = 0.0) -> Tuple[List[float], List[float], bool, float]:
        """
        Lateral repositioning.  Reduces alignment error effect temporarily.
        """
        s = self.state
        n = self.noise_scale

        # Adjustment partially corrects alignment — actually update the state
        correction = min(math.sqrt(dx**2 + dy**2), 0.05)
        prev_align = s.alignment_error
        effective_align = max(s.alignment_error - correction * 0.5, 0.0)
        s.alignment_error = effective_align  # BUG FIX: persist the correction

        s.position[0] += dx + _gauss(0.0, n, self._rng)
        s.position[1] += dy + _gauss(0.0, n, self._rng)

        force_x = (prev_align - effective_align) * 8.0
        force_y = (prev_align - effective_align) * 4.0
        force = [force_x + _gauss(0.0, n, self._rng),
                 force_y + _gauss(0.0, n, self._rng),
                 0.0]

        contact = s.position[2] > 0.005
        s.force_history.append(force)
        s.step_count += 1
        return list(s.position), force, contact, 0.0

    def apply_increase_force(self, delta: float = 0.3) -> Tuple[List[float], List[float], bool, float]:
        """
        Boost applied insertion force — risky with high friction.
        """
        s = self.state
        n = self.noise_scale

        boosted_force = 1.0 + delta
        friction_resistance = s.friction_level * boosted_force * 0.9
        effective_force_z = max(boosted_force - friction_resistance, 0.0)

        dz = (effective_force_z / (s.friction_level + 0.1)) * s.stiffness * 0.025
        s.position[2] += dz + _gauss(0.0, n * 0.3, self._rng)

        force_z = -(boosted_force + _gauss(0.0, n, self._rng))
        force_x = s.alignment_error * 12.0 + _gauss(0.0, n, self._rng)
        force_y = s.alignment_error *  6.0 + _gauss(0.0, n, self._rng)
        force = [force_x, force_y, force_z]

        contact = s.position[2] > 0.005
        s.contact_detected = contact

        prev_progress = s.task_progress
        s.task_progress = min(s.position[2] / s.INSERTION_DEPTH, 1.0)
        progress_delta = s.task_progress - prev_progress

        s.force_history.append(force)
        s.step_count += 1
        return list(s.position), force, contact, progress_delta

    # ------------------------------------------------------------------
    # Probe actions — produce signal_strength ∈ [0, 1]
    # ------------------------------------------------------------------

    def probe_friction(self) -> Tuple[List[float], float]:
        """
        Lateral sweep: high friction → high resistance → large |force_x|.
        Returns (force_vec, signal_strength).
        """
        s = self.state
        n = self.noise_scale * 0.5

        # Signal: normalised friction level
        signal = s.friction_level                       # ∈ [0.1, 1.0]
        force_x = signal * 5.0 + _gauss(0.0, n, self._rng)
        force_y = signal * 2.0 + _gauss(0.0, n, self._rng)
        force   = [force_x, force_y, _gauss(0.0, n, self._rng)]

        s.step_count += 1
        s.force_history.append(force)
        return force, min(signal, 1.0)

    def probe_alignment(self) -> Tuple[List[float], float]:
        """
        Oscillation test: misalignment → asymmetric lateral forces.
        Returns (force_vec, signal_strength).
        """
        s = self.state
        n = self.noise_scale * 0.5

        # Signal: normalised alignment error (0..0.05 → 0..1)
        signal = s.alignment_error / 0.05              # ∈ [0, 1]
        asymmetry = signal * 4.0
        force_x = asymmetry + _gauss(0.0, n, self._rng)
        force_y = asymmetry * 0.5 + _gauss(0.0, n, self._rng)
        force   = [force_x, force_y, _gauss(0.0, n, self._rng)]

        s.step_count += 1
        s.force_history.append(force)
        return force, min(signal, 1.0)

    def probe_stiffness(self) -> Tuple[List[float], float]:
        """
        Compression test: low stiffness → large displacement for small force.
        Returns (force_vec, signal_strength).
        """
        s = self.state
        n = self.noise_scale * 0.5

        # Low stiffness → high signal (signal inverted)
        signal = 1.0 - s.stiffness                     # ∈ [0, 0.9]
        deformation_z = signal * 0.03
        force_z = -(0.5 - signal * 0.4) + _gauss(0.0, n, self._rng)
        force   = [_gauss(0.0, n, self._rng), _gauss(0.0, n, self._rng), force_z]

        # Tiny position change from compression
        s.position[2] += deformation_z * 0.01
        s.step_count += 1
        s.force_history.append(force)
        return force, min(signal, 1.0)

    # ------------------------------------------------------------------
    # Anomaly flags derived from recent force history
    # ------------------------------------------------------------------

    def compute_anomaly_flags(self) -> Dict[str, bool]:
        history = self.state.force_history[-5:] if self.state.force_history else []

        # High force: max |fz| in recent history
        max_fz = max((abs(f[2]) for f in history), default=0.0)
        high_force = max_fz > 3.0

        # No progress: task_progress barely moved
        no_progress = self.state.task_progress < 0.05 and self.state.step_count > 3

        # Unstable contact: variance in fz signal
        if len(history) >= 3:
            fz_vals = [f[2] for f in history]
            mean_fz = sum(fz_vals) / len(fz_vals)
            var_fz = sum((v - mean_fz) ** 2 for v in fz_vals) / len(fz_vals)
            unstable = var_fz > 0.5
        else:
            unstable = False

        return {
            "high_force":        high_force,
            "no_progress":       no_progress,
            "unstable_contact":  unstable,
        }

    # ------------------------------------------------------------------
    # Success check
    # ------------------------------------------------------------------

    def is_success(self) -> bool:
        """
        Insertion succeeds when progress reaches 0.95.
        Alignment quality is captured by the grader's belief_accuracy_score,
        not here — otherwise bad alignment prevents ever completing the task.
        """
        s = self.state
        return s.task_progress >= 0.95

    # ------------------------------------------------------------------
    # Expose hidden state for graders (NOT for agent observation)
    # ------------------------------------------------------------------

    def hidden_state_dict(self) -> Dict[str, float]:
        s = self.state
        return {
            "friction_level":  s.friction_level,
            "alignment_error": s.alignment_error,
            "stiffness":       s.stiffness,
            "true_high_friction":  s.friction_level  > s.FRICTION_HIGH_THRESH,
            "true_bad_alignment":  s.alignment_error > s.ALIGNMENT_BAD_THRESH,
            "true_low_stiffness":  s.stiffness       < s.STIFFNESS_LOW_THRESH,
        }
