"""
inference.py — BEAR-RAEL Baseline Inference Script
===================================================
Runs an LLM agent against all three BEAR-RAEL tasks using the OpenAI client.

Environment variables:
    API_BASE_URL   LLM endpoint  (default: HuggingFace router)
    MODEL_NAME     Model ID      (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key
    BEAR_RAEL_URL  Environment server URL (default: http://localhost:7860)
    BEAR_TASK      Single task to run: easy | medium | hard | all (default: all)

Stdout format (strict):
    [START] task=<task> env=bear-rael model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "hf-placeholder")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL      = os.getenv("BEAR_RAEL_URL", "http://localhost:7860").rstrip("/")
BEAR_TASK    = os.getenv("BEAR_TASK", "all")

MAX_STEPS_PER_TASK = 20
TEMPERATURE        = 0.3
MAX_TOKENS         = 400
BENCHMARK          = "bear-rael"

TASKS = ["easy", "medium", "hard"]
TASK_MAX_STEPS = {"easy": 18, "medium": 20, "hard": 22}

# ---------------------------------------------------------------------------
# Logging helpers (strict format)
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(task: str, seed: int = 42) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/reset", json={"seed": seed, "task": task}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, parameters: Dict[str, float]) -> Dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action_type": action_type, "parameters": parameters},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_state() -> Dict[str, Any]:
    resp = requests.get(f"{ENV_URL}/state", timeout=10)
    resp.raise_for_status()
    return resp.json()


def env_grade() -> Dict[str, Any]:
    resp = requests.post(f"{ENV_URL}/grade", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# LLM prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
You are a robotics debugging agent for a peg-insertion task.

=== STRICT PHASE RULES (FOLLOW EXACTLY) ===

PHASE 1 — DIAGNOSIS (steps 1-3 ONLY):
  - Use probe_friction, probe_alignment, probe_stiffness — ONE each, NEVER repeat.
  - After 3 probes you MUST stop probing. No exceptions.

PHASE 2 — ACTION (steps 4 onward, MANDATORY):
  - Read the belief: [p_high_friction, p_bad_alignment, p_low_stiffness]
  - If p_bad_alignment > 0.6  → use adjust_position with dx=0.01, dy=0.01
  - If p_high_friction > 0.6  → use increase_force with delta=0.5, then insert
  - If p_low_stiffness > 0.6  → use insert with force_magnitude=2.0 (stiffness means MORE force needed)
  - Otherwise                 → use insert with force_magnitude=1.2
  - KEEP inserting every step until task_progress >= 0.95
  - When task_progress >= 0.95 → use commit_solution

=== PARAMETER NAMES (EXACT — do not invent names) ===
  insert:           {"force_magnitude": 1.2}
  adjust_position:  {"dx": 0.01, "dy": 0.01}
  increase_force:   {"delta": 0.5}
  probe_*:          {}
  commit_solution:  {}

=== OUTPUT FORMAT ===
Respond ONLY with valid JSON — no markdown, no explanation, no extra keys:
{"action_type": "insert", "parameters": {"force_magnitude": 1.2}}
""").strip()


def build_user_prompt(obs: Dict[str, Any], step: int, history: List[str]) -> str:
    hist_block = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(f"""
        Step {step} observation:
          position:         {obs.get('position')}
          force:            {obs.get('force')}
          contact_detected: {obs.get('contact_detected')}
          task_progress:    {obs.get('task_progress', 0.0):.3f}
          anomaly_flags:    {obs.get('anomaly_flags')}
          belief:           {[round(b, 3) for b in obs.get('belief', [0.5, 0.5, 0.5])]}
          step_count:       {obs.get('step_count', step)}

        Recent history:
        {hist_block}

        Decide your next action.
    """).strip()


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_agent_action(
    client: OpenAI,
    obs: Dict[str, Any],
    step: int,
    history: List[str],
    probes_used: int = 0,
    adjustments_used: int = 0,
) -> tuple[str, Dict[str, float], str]:
    """Returns (action_type, parameters, raw_response_str)."""
    user_prompt = build_user_prompt(obs, step, history)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "{}").strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        atype  = parsed.get("action_type", "probe_friction")
        raw_params = parsed.get("parameters", {})
        params = {}
        for k, v in raw_params.items():
            try:
                params[k] = float(v)
            except (TypeError, ValueError):
                pass  # skip unparseable values (e.g. strings with units)
        # === HARD ENFORCEMENT: stop probing after 3 total probes ===
        belief = obs.get("belief", [0.5, 0.5, 0.5])
        progress = obs.get("task_progress", 0.0)
        is_probe = atype.startswith("probe_")
        if is_probe and probes_used >= 3:
            # Force insertion based on belief
            if belief[1] > 0.6 and adjustments_used < 2:  # bad alignment, try adjusting once
                return "adjust_position", {"dx": 0.02, "dy": 0.02}, "forced:alignment"
            elif belief[0] > 0.6:  # high friction
                return "increase_force", {"delta": 0.5}, "forced:friction"
            else:
                return "insert", {"force_magnitude": 2.0}, "forced:insert"
        # === HARD ENFORCEMENT: stop adjusting after 2 times, switch to insert ===
        if atype == "adjust_position" and adjustments_used >= 2:
            if belief[0] > 0.6:
                return "increase_force", {"delta": 0.5}, "forced:friction_after_adjust"
            return "insert", {"force_magnitude": 2.0}, "forced:insert_after_adjust"
        # Commit if nearly done
        if progress >= 0.93 and atype not in ("commit_solution", "insert"):
            return "commit_solution", {}, "forced:commit"
        return atype, params, raw

    except Exception as exc:
        print(f"[DEBUG] LLM error: {exc}", flush=True)
        # Smart dynamic fallback: probe sequentially (never repeat), then adjust iteratively!
        progress = obs.get("task_progress", 0.0)
        if progress >= 0.93:
            return "commit_solution", {}, "fallback:commit"
        if probes_used == 0: return "probe_friction", {}, "fallback:probe_friction"
        if probes_used == 1: return "probe_alignment", {}, "fallback:probe_alignment"
        if probes_used == 2: return "probe_stiffness", {}, "fallback:probe_stiffness"
        
        # After 3 probes: select best action from belief, track real bounds constraints
        belief = obs.get("belief", [0.5, 0.5, 0.5])
        if belief[1] > 0.6 and adjustments_used < 4:
            # We must oscillate bounds or approach optimally without blowing out of boundaries!
            pos = obs.get("position", [0.0, 0.0, 0.0])
            dx = max(min(-pos[0], 0.01), -0.01) if abs(pos[0]) > 0.0 else 0.01
            dy = max(min(-pos[1], 0.01), -0.01) if abs(pos[1]) > 0.0 else 0.01
            return "adjust_position", {"dx": dx, "dy": dy}, "fallback:adjust"
            
        force = 4.0 if belief[2] > 0.6 else 2.0
        return "insert", {"force_magnitude": force}, f"fallback:insert (err={exc})"


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def run_episode(client: OpenAI, task: str, seed: int = 42) -> Dict[str, Any]:
    max_steps = TASK_MAX_STEPS.get(task, MAX_STEPS_PER_TASK)

    log_start(task=task, model=MODEL_NAME)

    rewards: List[float]  = []
    history: List[str]    = []
    steps_taken           = 0
    success               = False
    score                 = 0.0
    probes_used      = 0  # tracks total probe calls to enforce phase limit
    adjustments_used = 0  # tracks adjust_position calls to prevent infinite loop

    try:
        obs = env_reset(task=task, seed=seed)

        for step in range(1, max_steps + 1):
            atype, params, raw_resp = get_agent_action(client, obs, step, history, probes_used, adjustments_used)

            try:
                result = env_step(atype, params)
            except Exception as e:
                log_step(step, f"{atype}(ERROR)", 0.0, False, str(e))
                history.append(f"Step {step}: {atype} → ERROR: {e}")
                continue

            obs     = result["observation"]
            reward  = float(result.get("reward", 0.0))
            done    = bool(result.get("done", False))
            info    = result.get("info", {})
            error   = info.get("error") or None
            suc     = bool(info.get("success", False))

            action_str = f"{atype}({json.dumps(params, separators=(',', ':'))})" if params else f"{atype}()"
            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            rewards.append(reward)
            steps_taken = step
            if atype.startswith("probe_"):
                probes_used += 1
            if atype == "adjust_position":
                adjustments_used += 1
            history.append(
                f"Step {step}: {atype}{params} → reward={reward:+.3f} "
                f"progress={obs.get('task_progress', 0.0):.3f} done={done}"
            )

            if suc:
                success = True

            if done:
                break

        # Grade via API
        try:
            grade_result = env_grade()
            scores_dict = grade_result.get("scores", {})
            score = float(scores_dict.get("score", 0.0))
            print(f"[DEBUG] Grader scores: {scores_dict}", flush=True)
        except Exception as ge:
            print(f"[DEBUG] Grade API failed: {ge} — using reward fallback", flush=True)
            total_reward = sum(rewards)
            score = min(max(total_reward / (max_steps * 1.5), 0.0), 1.0)

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task, "success": success, "score": score, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Wait for server to be ready
    for attempt in range(12):
        try:
            resp = requests.get(f"{ENV_URL}/health", timeout=5)
            if resp.status_code == 200:
                break
        except Exception:
            pass
        print(f"[DEBUG] Waiting for server... attempt {attempt + 1}", flush=True)
        time.sleep(5)
    else:
        print("[DEBUG] Server did not become ready. Exiting.", flush=True)
        sys.exit(1)

    tasks_to_run = TASKS if BEAR_TASK == "all" else [BEAR_TASK]

    results = []
    for i, task in enumerate(tasks_to_run):
        seed = 42 + i * 100
        result = run_episode(client, task=task, seed=seed)
        results.append(result)
        time.sleep(1)  # brief pause between episodes

    # Summary
    print("\n[SUMMARY]", flush=True)
    for r in results:
        print(
            f"  task={r['task']} success={r['success']} "
            f"score={r['score']:.3f} steps={r['steps']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
