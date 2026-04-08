from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class Observation(BaseModel):
    position:         List[float]       = Field(..., description="[x, y, z] end-effector position in metres")
    velocity:         List[float]       = Field(..., description="[vx, vy, vz] estimated velocity")
    force:            List[float]       = Field(..., description="[fx, fy, fz] measured contact force (N)")
    contact_detected: bool              = Field(..., description="True if end-effector is in contact with surface")
    task_progress:    float             = Field(..., ge=0.0, le=1.0, description="Fraction of insertion completed")
    anomaly_flags:    Dict[str, bool]   = Field(..., description="Detected anomalies: high_force / no_progress / unstable_contact")
    last_action:      str               = Field(..., description="String representation of the last executed action")
    step_count:       int               = Field(..., description="Number of steps taken in this episode")
    belief:           List[float]       = Field(..., description="[p_high_friction, p_bad_alignment, p_low_stiffness]")

class Action(BaseModel):
    action_type: str                    = Field(..., description=(
        "One of: insert | adjust_position | increase_force | "
        "probe_friction | probe_alignment | probe_stiffness | commit_solution"
    ))
    parameters:  Dict[str, float]       = Field(default_factory=dict, description=(
        "Action parameters. "
        "insert: {force_magnitude: float}. "
        "adjust_position: {dx: float, dy: float}. "
        "increase_force: {delta: float}. "
        "Probes and commit_solution: {} (empty)."
    ))

class Reward(BaseModel):
    total:                 float = Field(..., description="Total reward for this step")
    progress_delta:        float = Field(..., description="Change in task_progress this step")
    info_gain_bonus:       float = Field(..., description="Entropy reduction bonus from probing")
    success_bonus:         float = Field(..., description="Bonus awarded on successful completion")
    step_penalty:          float = Field(..., description="Per-step penalty")
    redundant_probe_penalty: float = Field(..., description="Penalty for re-probing same type")
    invalid_action_penalty: float  = Field(..., description="Penalty for malformed / unsupported action")

class ResetRequest(BaseModel):
    seed:  int  = 42
    task:  str  = "easy"

class StepRequest(BaseModel):
    action_type: str
    parameters:  Dict[str, float] = {}

class GradeResponse(BaseModel):
    task:   str
    scores: Dict[str, Any]
