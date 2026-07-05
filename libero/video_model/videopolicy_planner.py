
"""
videopolicy_planner.py
----------------------

Slim planner for VideoPolicy + RoboCasa that breaks a task instruction
(ep_meta["lang"]) into *small subtasks*, suitable to feed to a video model.

Two APIs:
- plan_steps(...)  -> returns List[str] of short, imperative steps
- plan_task(...)   -> (kept from v1) returns rich JSON plan (optional)

Requirements
- Python 3.9+
- pip install -U google-genai
- (Optional) pip install h5py for read_lang_from_hdf5()

Auth
- Pass api_key="..." OR set GEMINI_API_KEY in your environment.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Optional import guard for h5py; only needed if you use read_lang_from_hdf5
try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None  # type: ignore

# ---------- Minimal planner prompt (steps only) ----------

STEPS_SYSTEM_PROMPT = r"""
You convert ONE RoboCasa task instruction (kitchen domain) into a compact sequence
of small subtasks that a video diffusion model can animate as future frames.

Return a STRICT JSON object:
{ "steps": ["<step 1>", "<step 2>", "..."] }

Rules
- Write imperative, concise steps (<= 12 words each).
- Prefer 3–7 steps for composite tasks; 1–3 for atomic tasks.
- Do NOT number steps; no extra punctuation beyond what's natural.
- Only include objects/actions implied by the instruction; do not invent new ones.
- Preserve any explicit ordering in the instruction (“then … then …”).
- Output ONLY the JSON object above. No commentary.
"""

# (Kept) Rich planner prompt for users who still want detailed chunks
PLANNER_SYSTEM_PROMPT = r"""
You are a Task-to-Plan converter for simulated kitchens (RoboCasa).
Given ONE natural-language task instruction from the RoboCasa dataset,
return a STRICT JSON object that decomposes the task into stepwise chunks
following an Action → World-State-Change (ΔS) pattern.

Rules
- Read the instruction literally; do not invent appliances or objects not implied by the text.
- Prefer 3–7 steps for composite tasks; 1–3 for atomic tasks.
- Each step MUST have:
  step_id (int, 1-based),
  subgoal (<=10 words),
  action (imperative, robot-doing words),
  world_state_delta (what visibly changes after this step),
  objects (array of key nouns),
  preconditions (list; things that must already hold),
  success_criteria (list; unambiguous, observable),
  visuals (object_focus[], camera_hint, start_caption, end_caption),
  duration_seconds (float).
- Keep language concise and concrete; avoid policy/control details.
- If the instruction already enumerates steps (“then … then …”), preserve that order.
- Never include extra commentary outside the JSON.

Output JSON schema
{
  "goal": "<short paraphrase>",
  "assumptions": ["<only what the instruction implies>"],
  "steps": [
    {
      "step_id": 1,
      "subgoal": "",
      "action": "",
      "world_state_delta": "",
      "objects": ["", "..."],
      "preconditions": ["", "..."],
      "success_criteria": ["", "..."],
      "visuals": {
        "object_focus": ["..."],
        "camera_hint": "",
        "start_caption": "",
        "end_caption": ""
      },
      "duration_seconds": 1.8
    }
  ]
}

Return ONLY the JSON.
"""

FEW_SHOT_STEPS = r"""
Examples (steps-only)

Instruction: "Open the microwave, put the bowl inside, close it, then start it."
Expected:
{ "steps": [
  "open the microwave door",
  "place the bowl inside the microwave",
  "close the microwave door",
  "press the start button"
]}

Instruction: "Press the start button on the microwave."
Expected:
{ "steps": ["press the start button on the microwave"] }
"""

def _extract_json(text: str) -> Dict[str, Any]:
    """Extract first top-level JSON object from text (removes fences if present)."""
    if not isinstance(text, str):
        raise ValueError("Model returned non-text response.")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:120]}...")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                return json.loads(candidate)
    raise ValueError("Unbalanced braces; could not extract JSON.")

def _resp_text(resp: Any) -> str:
    """Normalize google-genai responses to text."""
    text = getattr(resp, "text", None)
    if text:
        return text
    text = getattr(resp, "output_text", None)
    if text:
        return text
    return str(resp)

# ---------- Public API (steps only) ----------

def plan_steps(lang: str, api_key: Optional[str] = None, model: str = "gemini-2.5-pro") -> List[str]:
    """
    Convert a RoboCasa instruction string (ep_meta['lang']) into a compact list of subtasks.

    Returns
    -------
    List[str]  # e.g., ["open the microwave door", "place the bowl inside", "close the door", "press start"]
    """
    try:
        from google import genai  # pip install google-genai
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: please `pip install -U google-genai`") from e

    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    user_prompt = (
        STEPS_SYSTEM_PROMPT.strip() + "\n\n" + FEW_SHOT_STEPS.strip()
        + "\n\nInstruction: " + json.dumps(lang) + "\nReturn ONLY the JSON."
    )
    resp = client.models.generate_content(model=model, contents=user_prompt)
    data = _extract_json(_resp_text(resp))
    steps = data.get("steps", [])
    if not isinstance(steps, list) or not all(isinstance(s, str) and s.strip() for s in steps):
        raise ValueError("Model did not return a valid 'steps' list.")
    return [s.strip() for s in steps]

# ---------- Optional: keep rich plan for users who still want it ----------

def plan_task(lang: str, api_key: Optional[str] = None, model: str = "gemini-2.5-pro") -> Dict[str, Any]:
    """Legacy: return detailed plan JSON (action + world_state_delta + visuals, etc.)."""
    try:
        from google import genai  # pip install google-genai
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: please `pip install -U google-genai`") from e

    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    prompt = PLANNER_SYSTEM_PROMPT.strip() + "\n\nInstruction: " + json.dumps(lang) + "\nReturn ONLY the JSON."
    resp = client.models.generate_content(model=model, contents=prompt)
    data = _extract_json(_resp_text(resp))
    if "steps" not in data or not isinstance(data["steps"], list):
        raise ValueError("Detailed planner: missing 'steps' array.")
    return data

# ---------- Convenience helpers ----------

def read_lang_from_hdf5(h5_path: str) -> str:
    """Fetch ep_meta['lang'] from a RoboCasa .hdf5 demo (requires h5py)."""
    if h5py is None:
        raise RuntimeError("h5py is not installed. Please `pip install h5py`.")
    with h5py.File(h5_path, "r") as f:
        ep_meta_raw = f.attrs.get("ep_meta", None)
        if ep_meta_raw is None:
            raise KeyError("Attribute 'ep_meta' not found in HDF5 file.")
        if isinstance(ep_meta_raw, (bytes, bytearray)):
            ep_meta_raw = ep_meta_raw.decode("utf-8")
        ep_meta = json.loads(ep_meta_raw)
        lang = ep_meta.get("lang", None)
        if not lang:
            raise KeyError("ep_meta['lang'] not found in HDF5 attrs.")
        return lang

# ---------- CLI ----------

if __name__ == "__main__":
    import argparse, sys
    parser = argparse.ArgumentParser(description="RoboCasa minimal planner -> small subtasks")
    parser.add_argument("--lang", type=str, help="RoboCasa task instruction string")
    parser.add_argument("--api_key", type=str, default=None, help="Gemini API key (or set GEMINI_API_KEY)")
    parser.add_argument("--model", type=str, default="gemini-2.5-pro")
    parser.add_argument("--detailed", action="store_true", help="Return rich plan JSON instead of small steps")
    args = parser.parse_args()
    if not args.lang:
        print("Please pass --lang 'your instruction'.", file=sys.stderr)
        sys.exit(2)
    if args.detailed:
        out = plan_task(args.lang, api_key=args.api_key, model=args.model)
        print(json.dumps(out, indent=2, ensure_ascii=False))
    else:
        steps = plan_steps(args.lang, api_key=args.api_key, model=args.model)
        print(json.dumps({"steps": steps}, indent=2, ensure_ascii=False))
