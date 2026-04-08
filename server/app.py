import sys
import os
from typing import Any, Dict, Optional

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from models import Observation, Action, ResetRequest, StepRequest, GradeResponse
from env.environment import BearRaelEnv
from tasks import task_easy, task_medium, task_hard
from graders import grader_easy, grader_medium, grader_hard

app = FastAPI(
    title="BEAR-RAEL OpenEnv",
    description="Bayesian Embodied Autonomous Robotics Lab — an RL environment for robotics debugging.",
    version="1.0.0",
)

# Single shared environment instance
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

_TASK_NAME_MAP = {
    "single_variable_diagnosis": "easy",
    "dual_variable_interaction":  "medium",
    "full_debugging_noisy":       "hard",
    "easy":   "easy",
    "medium": "medium",
    "hard":   "hard",
}

@app.get("/", response_class=HTMLResponse)
def read_root():
    return """
    <html>
        <head><title>BEAR-RAEL OpenEnv</title></head>
        <body style="font-family: sans-serif; text-align: center; margin-top: 50px;">
            <h1>🤖 BEAR-RAEL OpenEnv</h1>
            <p>Status: <span style="color: green;">RUNNING</span></p>
            <p><a href="/docs" style="display:inline-block; padding: 10px 20px; background-color: #007BFF; color: white; text-decoration: none; border-radius: 5px;">API Docs</a></p>
        </body>
    </html>
    """

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "env": "bear-rael"}

@app.get("/tasks")
def list_tasks() -> dict:
    return {name: module.describe() for name, module in TASK_REGISTRY.items()}

@app.post("/reset", response_model=Observation)
def reset(req: ResetRequest = None) -> Observation:
    req = req or ResetRequest()
    if req.task not in TASK_REGISTRY:
        raise HTTPException(status_code=400, detail=f"Unknown task '{req.task}'")
    
    task_module = TASK_REGISTRY[req.task]
    env_instance = task_module.make_task_env(seed=req.seed)
    global _env
    _env = env_instance
    return _env._build_observation()

@app.post("/step", response_model=Any) # Manual response model for flexibility
def step(req: StepRequest) -> dict:
    action = Action(action_type=req.action_type, parameters=req.parameters)
    obs, reward, done, info = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": float(reward),
        "done": bool(done),
        "info": _safe_dict(info)
    }

@app.get("/state")
def state() -> dict:
    return _safe_dict(_env.state())

@app.post("/grade", response_model=GradeResponse)
def grade() -> GradeResponse:
    env_state = _env.state()
    raw_task  = env_state.get("task", "easy")
    task_name = _TASK_NAME_MAP.get(raw_task, raw_task)
    grader    = GRADER_REGISTRY.get(task_name)

    if grader is None:
        raise HTTPException(status_code=400, detail="No grader found")

    scores = grader.grade(env_state)
    return GradeResponse(task=task_name, scores=scores)

def _safe_dict(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: _safe_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_safe_dict(v) for v in d]
    if isinstance(d, (int, float, str, bool, type(None))):
        return d
    return str(d)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
