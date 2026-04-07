"""
server.py — FastAPI server exposing the BEAR-RAEL environment via REST.

Endpoints:
    POST /reset          — reset(seed, task) → Observation
    POST /step           — step(action) → StepResponse
    GET  /state          — state() → dict
    GET  /health         — liveness probe
    GET  /tasks          — list available tasks + descriptions
    POST /grade          — grade current episode state
"""

from __future__ import annotations

import sys
import os

from typing import Any, Dict, Optional

# Ensure project root is on path when running from api/ subdirectory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from env.environment import BearRaelEnv, Action, Observation
from tasks import task_easy, task_medium, task_hard
from graders import grader_easy, grader_medium, grader_hard

app = FastAPI(
    title="BEAR-RAEL OpenEnv",
    description=(
        "Bayesian Embodied Autonomous Robotics Lab — "
        "an RL environment for robotics debugging via active probing."
    ),
    version="1.0.0",
)

# Single shared environment instance (stateful per session)
_env: BearRaelEnv = BearRaelEnv()

TASK_REGISTRY = {
    "easy":   task_easy,
    "medium": task_medium,
    "hard":   task_hard,
}

GRADER_REGISTRY = {
    "easy":   grader_easy,
    "medium": grader_medium,
    "hard":   grader_hard,
}


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed:  int  = 42
    task:  str  = "easy"


class StepRequest(BaseModel):
    action_type: str
    parameters:  Dict[str, float] = {}


class StepResponse(BaseModel):
    observation: Observation
    reward:      float
    done:        bool
    info:        Dict[str, Any]


class GradeResponse(BaseModel):
    task:   str
    scores: Dict[str, float]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head><title>BEAR-RAEL OpenEnv</title></head>
        <body style="font-family: sans-serif; text-align: center; margin-top: 50px;">
            <h1>🤖 BEAR-RAEL OpenEnv</h1>
            <p>Your Bayesian Robotics API is <strong>RUNNING</strong>!</p>
            <p>STATUS: <span style="color: green;">OK 200</span></p>
            <hr style="width: 50%;" />
            <p><a href="/docs" style="display:inline-block; padding: 10px 20px; background-color: #007BFF; color: white; text-decoration: none; border-radius: 5px;">View API Documentation</a></p>
        </body>
    </html>
    """


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": "bear-rael"}


@app.get("/tasks")
def list_tasks() -> dict:
    return {
        name: module.describe()
        for name, module in TASK_REGISTRY.items()
    }


@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest) -> Observation:
    if req.task not in TASK_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{req.task}'. Valid tasks: {list(TASK_REGISTRY)}",
        )
    task_module = TASK_REGISTRY[req.task]
    obs = task_module.make_task_env(seed=req.seed)
    # Replace shared env with the one configured by the task module
    global _env
    _env = obs  # make_task_env returns a reset BearRaelEnv
    return _env._build_observation()


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest) -> StepResponse:
    action = Action(action_type=req.action_type, parameters=req.parameters)
    obs, reward, done, info = _env.step(action)
    # info may contain non-serialisable types; coerce to plain dict
    safe_info = _safe_dict(info)
    return StepResponse(observation=obs, reward=reward, done=done, info=safe_info)


@app.get("/state")
def state() -> dict:
    return _safe_dict(_env.state())


# Map full task names → short registry keys
_TASK_NAME_MAP = {
    "single_variable_diagnosis": "easy",
    "dual_variable_interaction":  "medium",
    "full_debugging_noisy":       "hard",
    "easy":   "easy",
    "medium": "medium",
    "hard":   "hard",
}

@app.post("/grade", response_model=GradeResponse)
def grade() -> GradeResponse:
    env_state = _env.state()
    raw_task  = env_state.get("task", "easy")
    task_name = _TASK_NAME_MAP.get(raw_task, raw_task)
    grader    = GRADER_REGISTRY.get(task_name)

    if grader is None:
        raise HTTPException(status_code=400,
            detail=f"No grader for task '{raw_task}' (mapped: '{task_name}'). "
                   f"Valid keys: {list(GRADER_REGISTRY.keys())}")

    scores = grader.grade(env_state)
    return GradeResponse(task=task_name, scores=scores)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_dict(d: Any) -> Any:
    """Recursively convert non-JSON-serialisable values to strings."""
    if isinstance(d, dict):
        return {k: _safe_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_safe_dict(v) for v in d]
    if isinstance(d, bool):
        return d
    if isinstance(d, (int, float, str, type(None))):
        return d
    return str(d)


# ---------------------------------------------------------------------------
# Dev entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.server:app", host="0.0.0.0", port=7860, reload=False)
