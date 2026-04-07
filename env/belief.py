"""
belief.py — BEAR Component: Bayesian belief update over hidden physical properties.

Belief vector: [p_high_friction, p_bad_alignment, p_low_stiffness]
Updates are deterministic given probe outcomes, ensuring reproducibility.
"""

import math
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------

def entropy(probs: List[float]) -> float:
    """Shannon entropy of a probability list (nats)."""
    h = 0.0
    for p in probs:
        p = max(p, 1e-9)
        h -= p * math.log(p)
    return h


def binary_entropy(p: float) -> float:
    """Binary entropy H(p) in nats."""
    p = max(min(p, 1.0 - 1e-9), 1e-9)
    return -(p * math.log(p) + (1 - p) * math.log(1 - p))


# ---------------------------------------------------------------------------
# Belief state
# ---------------------------------------------------------------------------

class BeliefState:
    """
    Maintains and updates a belief over three binary hidden-variable hypotheses:
        [0] p_high_friction   — friction_level > 0.6
        [1] p_bad_alignment   — alignment_error > 0.025
        [2] p_low_stiffness   — stiffness < 0.4

    Updates are deterministic: given a probe signal strength, we apply a
    likelihood-ratio update clamped to [0.05, 0.95].
    """

    PRIOR = [0.5, 0.5, 0.5]

    # Likelihood P(signal | hypothesis=True) and P(signal | hypothesis=False)
    # These drive the Bayesian update.
    _LIKELIHOOD_TRUE  = 0.85   # probe correctly signals the condition
    _LIKELIHOOD_FALSE = 0.20   # probe fires even when condition absent

    def __init__(self):
        self.probs: List[float] = list(self.PRIOR)
        self._probe_history: Dict[str, int] = {
            "probe_friction": 0,
            "probe_alignment": 0,
            "probe_stiffness": 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self.probs = list(self.PRIOR)
        self._probe_history = {k: 0 for k in self._probe_history}

    def as_list(self) -> List[float]:
        return list(self.probs)

    def total_entropy(self) -> float:
        return sum(binary_entropy(p) for p in self.probs)

    def info_gain(self, prev_probs: List[float]) -> float:
        """KL-divergence style gain: reduction in total entropy."""
        prev_h = sum(binary_entropy(p) for p in prev_probs)
        new_h  = sum(binary_entropy(p) for p in self.probs)
        return max(prev_h - new_h, 0.0)

    # ------------------------------------------------------------------
    # Belief updates triggered by probe outcomes
    # ------------------------------------------------------------------

    def update_from_probe(
        self,
        probe_type: str,
        signal_strength: float,   # 0.0 = absent, 1.0 = strong positive
    ) -> Tuple[List[float], float]:
        """
        Update one belief component based on probe signal strength.

        signal_strength is continuous [0, 1] produced by dynamics.py.
        We threshold at 0.5 as a soft Bayesian likelihood weighting.

        Returns (new_probs, info_gain_nats).
        """
        prev = list(self.probs)

        idx = {"probe_friction": 0, "probe_alignment": 1, "probe_stiffness": 2}.get(probe_type)
        if idx is None:
            return self.probs, 0.0

        self._probe_history[probe_type] = self._probe_history.get(probe_type, 0) + 1

        # Soft likelihood weighting: signal_strength interpolates likelihoods
        lh_true  = self._LIKELIHOOD_FALSE + signal_strength * (self._LIKELIHOOD_TRUE  - self._LIKELIHOOD_FALSE)
        lh_false = self._LIKELIHOOD_FALSE + signal_strength * (self._LIKELIHOOD_TRUE  - self._LIKELIHOOD_FALSE) * 0.1 + self._LIKELIHOOD_FALSE * 0.9

        # Bayesian update on the single component
        p = self.probs[idx]
        numerator   = lh_true  * p
        denominator = numerator + lh_false * (1.0 - p)
        p_new = numerator / (denominator + 1e-12)

        # Clamp to avoid degenerate posteriors
        self.probs[idx] = max(0.05, min(0.95, p_new))

        gain = self.info_gain(prev)
        return list(self.probs), gain

    def probe_count(self, probe_type: str) -> int:
        return self._probe_history.get(probe_type, 0)

    def total_probes(self) -> int:
        return sum(self._probe_history.values())

    # ------------------------------------------------------------------
    # Grader helper: belief accuracy vs true hidden state
    # ------------------------------------------------------------------

    def belief_accuracy_score(
        self,
        true_high_friction: bool,
        true_bad_alignment: bool,
        true_low_stiffness: bool,
    ) -> float:
        """
        Returns a [0, 1] score measuring how well the belief matches reality.
        Penalises confident wrong beliefs; rewards calibrated correct beliefs.
        """
        ground_truth = [true_high_friction, true_bad_alignment, true_low_stiffness]
        scores = []
        for p, truth in zip(self.probs, ground_truth):
            # p is belief in 'True'; truth is actual bool
            match = p if truth else (1.0 - p)
            scores.append(match)
        return sum(scores) / len(scores)
