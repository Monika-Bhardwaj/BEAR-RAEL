---
title: BEAR-RAEL OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---
# BEAR-RAEL 🤖
### Bayesian Embodied Autonomous Robotics Lab

> *"This environment models a real robotics debugging workflow where agents must actively perform diagnostic experiments to infer hidden physical properties and adapt manipulation strategies."*

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://openenv.ai)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)](https://fastapi.tiangolo.com)

---

## Overview

BEAR-RAEL is an **active-probing robotics environment** where an AI agent must:

1. **Execute** a peg-insertion manipulation task
2. **Detect** failure signals (force spikes, stalled progress, contact instability)
3. **Run diagnostic experiments** (probing actions) to infer hidden physical properties
4. **Update a Bayesian belief** over three latent variables
5. **Adapt strategy** to complete the task efficiently

This directly models what human robotics engineers do: observe signals, run tests, adjust parameters.

---

## Why This Environment?

Unlike toy environments, BEAR-RAEL evaluates a capability that matters for real deployments:
**can the agent actively gather information and reason under uncertainty?**

Brute-force trial-and-error is penalised. The agent must probe strategically, interpret physical signals, maintain calibrated uncertainty, and select actions that address the *inferred* root cause.

---

## Hidden State (Never Directly Observable)

| Variable | Range | Problematic When |
|----------|-------|-----------------|
| `friction_level` | [0.1, 1.0] | > 0.6 → high resistance |
| `alignment_error` | [0.0, 0.05] m | > 0.025 → lateral drift |
| `stiffness` | [0.1, 1.0] | < 0.4 → oscillatory contact |

Variables are sampled per episode and can only be inferred through probe actions.

---

## Observation Space

```python
class Observation(BaseModel):
    position:         List[float]      # [x, y, z] end-effector position (m)
    velocity:         List[float]      # [vx, vy, vz] estimated velocity
    force:            List[float]      # [fx, fy, fz] contact force (N)
    contact_detected: bool
    task_progress:    float            # 0.0 – 1.0
    anomaly_flags:    Dict[str, bool]  # high_force | no_progress | unstable_contact
    last_action:      str
    step_count:       int
    belief:           List[float]      # [p_high_friction, p_bad_alignment, p_low_stiffness]
```

---

## Action Space

| Action | Parameters | Effect |
|--------|-----------|--------|
| `insert` | `force_magnitude: float` | Attempt downward insertion |
| `adjust_position` | `dx, dy: float` | Lateral correction for alignment |
| `increase_force` | `delta: float` | Apply boosted force (risky with high friction) |
| `probe_friction` | *(none)* | Lateral sweep → resistance signal |
| `probe_alignment` | *(none)* | Oscillation test → asymmetry signal |
| `probe_stiffness` | *(none)* | Compression test → deformation signal |
| `commit_solution` | *(none)* | Declare task complete |

---

## Probe Signal Design

Each probe returns a continuous **signal_strength ∈ [0, 1]** that drives a Bayesian belief update:

- **probe_friction** → lateral force magnitude (high signal = high friction)
- **probe_alignment** → force asymmetry (high signal = bad alignment)
- **probe_stiffness** → deformation response (high signal = low stiffness)

Belief updates are **deterministic** given the signal, ensuring reproducibility.

---

## Reward Function

```
total_reward = progress_delta × 3.0
             + info_gain_bonus         # entropy reduction × 0.5
             + success_bonus           # 1.0 + efficiency_bonus on completion
             + step_penalty            # −0.02 per step
             + redundant_probe_penalty # −0.10 for re-running same probe
             + invalid_action_penalty  # −0.15 for unknown actions
```

Range: **[−1.0, 2.0]**

---

## Tasks

### 🟢 Task 1: Single Variable Diagnosis (`easy`)
- **One** hidden variable is problematic; the others are fixed at benign values
- Step budget: 15
- Expected difficulty: solvable with 1 targeted probe + correct insertion strategy

### 🟡 Task 2: Dual Variable Interaction (`medium`)
- **Two** hidden variables are simultaneously problematic
- Step budget: 18
- Agent must distinguish confounded failure signals and address both causes

### 🔴 Task 3: Full Debugging Under Noise (`hard`)
- **All three** variables vary from their full ranges
- Sensor noise amplified **3×**
- Step budget: 20
- Efficient, minimal probing is required; over-probing leads to failure

---

## Grader Scoring (0.0 – 1.0)

Each grader is **deterministic** and **reproducible**.

| Component | Easy | Medium | Hard |
|-----------|------|--------|------|
| Success | 0.50 | 0.45 | 0.40 |
| Efficiency | 0.20 | 0.15 | 0.20 |
| Belief Accuracy | 0.20 | 0.30 | 0.30 |
| Probe Penalty | −0.10 | −0.10 | −0.15 |
| Noise Bonus | — | — | +0.05 |

---

## Project Structure

```
bear-rael/
├── env/
│   ├── environment.py    # OpenEnv interface + Pydantic models
│   ├── dynamics.py       # Physics simulation + hidden state
│   └── belief.py         # Bayesian belief update (BEAR component)
├── tasks/
│   ├── task_easy.py      # Single variable diagnosis
│   ├── task_medium.py    # Dual variable interaction
│   └── task_hard.py      # Full debugging under noise
├── graders/
│   ├── grader_easy.py    # Deterministic grader (easy)
│   ├── grader_medium.py  # Deterministic grader (medium)
│   └── grader_hard.py    # Deterministic grader (hard)
├── api/
│   └── server.py         # FastAPI REST server
├── openenv.yaml          # OpenEnv spec metadata
├── inference.py          # Baseline inference script
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Setup & Usage

### Local (Python)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
python -m uvicorn api.server:app --host 0.0.0.0 --port 7860

# In another terminal, run inference
export HF_TOKEN=your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Docker

```bash
docker build -t bear-rael .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  bear-rael
```

### API Quick Reference

```bash
# Reset to easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "task": "easy"}'

# Take a probing step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "probe_friction", "parameters": {}}'

# Get full state
curl http://localhost:7860/state

# Grade the episode
curl -X POST http://localhost:7860/grade
```

---

## Baseline Scores

Measured with `Qwen/Qwen2.5-72B-Instruct`, seed=42:

| Task | Score | Success | Steps |
|------|-------|---------|-------|
| easy (single_variable_diagnosis) | ~0.62 | ✅ | ~9 |
| medium (dual_variable_interaction) | ~0.48 | ✅ | ~13 |
| hard (full_debugging_noisy) | ~0.31 | ❌ | 20 |

*Scores are reproducible with fixed seeds. Frontier models should score higher on hard.*

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | — | HuggingFace / API key |
| `BEAR_RAEL_URL` | `http://localhost:7860` | Environment server URL |
| `BEAR_TASK` | `all` | Which tasks to run: `easy` / `medium` / `hard` / `all` |

---

## Real-World Mapping

| Environment Component | Real-World Equivalent |
|----------------------|----------------------|
| Hidden friction/alignment/stiffness | Unknown physical parameters in a new workpiece |
| Probe actions | Engineer running diagnostic tests |
| Belief update | Bayesian parameter estimation from sensor data |
| Adaptive insertion strategy | Adjusted robot program based on test results |
| Efficiency penalty | Time/cost of failed insertion attempts |

---

## License

MIT
