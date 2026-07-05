# video_model/planner/online_planner.py
from __future__ import annotations
import os, json, time, hashlib, threading
from typing import Dict, List, Optional
import re, ast
# ---------- prompts (always 2–5 actions) ----------
SYSTEM_PROMPT= """
You are a planner that converts ONE RoboCasa task instruction
into a short, ordered list of ACTIONS for video rollout.

Planning principle
- Follow VLWM-style abstraction: each step is an ACTION that causes a meaningful,
  observable world-state change. Keep state implicit—emit ACTIONS only.

Hard constraints
- Always return between 2 and 5 actions inclusive.
- Output ONLY a strict JSON object with this shape (valid JSON, double quotes):
  {"steps": ["<action 1>", "<action 2>", "..."]}
- Each action is a concise, imperative verb phrase (≤ 12 words).
- Preserve explicit ordering from the instruction (“then … then …”).
- Use only objects/affordances implied by the instruction; don’t invent props.

Granularity rules
- Composite instructions → 3–5 actions capturing the key transitions.
- If more than 5 micro-steps are implied, compress adjacent ones while preserving core transitions.
- If the instruction is atomic (single operation), split naturally into two micro-actions
  (e.g., reach/align → operate; grasp → move).

Style rules
- Use lowercase; no numbering or “Action k:” labels inside strings.
- No extra keys, comments, or trailing text—return ONLY the JSON object."""

INSTRUCTION_PROMPT= """
Task instruction:

{{SYSTEM_PROMPT}}

Return ONLY a strict JSON object of 2–5 concise action strings:
{"steps": ["...", "..."]}

In-context examples:

Instruction:
Open the microwave, put the bowl inside, close it, then start it.
Expected:
{"steps": [
  "open the microwave door",
  "place the bowl inside the microwave",
  "close the microwave door",
  "press the start button"
]}

Instruction:
Press the start button on the microwave.
Expected:
{"steps": [
  "reach to the microwave start button",
  "press the start button on the microwave"
]}

Instruction:
Open the top cabinet and put the plate inside.
Expected:
{"steps": [
  "open the top cabinet door",
  "place the plate on the cabinet shelf",
  "close the top cabinet door"
]}

Instruction:
Cooking Tomato and Eggs
Expected:
{"steps": [
  "preheat the skillet on the stove",
  "crack eggs into a bowl and whisk",
  "season the eggs with salt",
  "sauté the chopped tomatoes in the skillet",
  "pour eggs into skillet and scramble with tomatoes"
]}

Instruction:
Turn on the left stove burner.
Expected:
{"steps": [
  "reach to the left burner knob",
  "turn the left burner knob to on"
]}

Now provide the steps for task instruction:
Instruction:
"""

# ---------- messy-output tolerant parser ----------
_STEP_PREFIX_RE = re.compile(r"^(?:action\s*\d+\s*:\s*|\d+\.\s*|[-*•]\s*)", re.IGNORECASE)



def _strip_wrappers(text: str) -> str:
    """Remove code fences (```json ... ```), stray language tags, and triple quotes."""
    t = text.strip()

    # Remove leading/trailing triple quotes
    t = re.sub(r"^\s*(['\"]) {3}", "", t)  # extremely cautious; keep for rare cases
    t = re.sub(r"^(['\"]) {3}|(['\"]) {3}$", "", t)  # deprecated pattern safeguard
    t = re.sub(r"^\s*(['\"]){3}", "", t)
    t = re.sub(r"(['\"]){3}\s*$", "", t)

    # Remove Markdown code fences like ```json ... ```
    t = re.sub(r"^\s*```[\w-]*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"```\s*$", "", t)

    # If someone wrote: json{ ... }
    t = re.sub(r"^\s*json\s*(?=\{)", "", t, flags=re.IGNORECASE)

    return t.strip()


def _extract_json_obj(text: str):
    """Find the first top-level {...} and parse it as JSON or Python literal."""
    t = _strip_wrappers(text)
    start = t.find("{")
    if start == -1:
        return None

    # Brace matching
    depth = 0
    end = None
    for i, ch in enumerate(t[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end is None:
        return None

    candidate = t[start:end].strip()
    # First try strict JSON
    try:
        return json.loads(candidate)
    except Exception:
        pass
    # Fallback: Python literal (handles single quotes, True/False/None)
    try:
        return ast.literal_eval(candidate)
    except Exception:
        return None

_STEP_PREFIX_RE = re.compile(r"^(?:action\s*\d+\s*:\s*|\d+\.\s*|[-*•]\s*)", re.IGNORECASE)

def _normalize_step(s: str) -> str:
    s = s.strip()
    s = _STEP_PREFIX_RE.sub("", s)               # remove "Action 2:", "1.", "-", etc.
    s = re.sub(r"\s+", " ", s)                   # collapse whitespace
    s = s.strip(" .,:;")                         # trim trailing punctuation
    return s.lower()



def parse_steps_output(raw: str,
                       min_steps: int = 2,
                       max_steps: int = 5) -> Dict[str, List[str]]:
    """
    Post-process LLM output into {"steps": [...]}, always 2–5 actions.
    Handles:
      - ```json{...}``` or ```{...}``` blocks
      - leading 'json' before '{'
      - extra ''' around content
      - single-quoted dicts via ast.literal_eval
      - non-JSON 'Action k: ...' line lists
    """
    steps: List[str] = []

    obj = _extract_json_obj(raw)
    if isinstance(obj, dict) and "steps" in obj and isinstance(obj["steps"], (list, tuple)):
        # Got a steps array; normalize each entry
        steps = [_normalize_step(str(x)) for x in obj["steps"] if str(x).strip()]

    if not steps:
# Fallback: try to parse line-formatted outputs (Action k:, bullets, numbers)
        candidates = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r"Action\s*\d+\s*:\s*(.+)", line, flags=re.IGNORECASE)
            if m:
                candidates.append(m.group(1))
            elif re.match(r"^(\d+\.\s*|[-*•]\s+)", line):
                candidates.append(line)
        if not candidates:
            # Last resort: treat any non-empty lines as candidate steps
            candidates = [l for l in raw.splitlines() if l.strip()]
        steps = [_normalize_step(x) for x in candidates if x.strip()]

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for s in steps:
        if s and s not in seen:
           uniq.append(s); seen.add(s)
    steps = uniq

    # Enforce bounds 2–5
    if len(steps) > max_steps:
        steps = steps[:max_steps]
    elif len(steps) < min_steps:
        # Try to split the first step by 'then/and/,/;' into micro-actions
        if steps:
            parts = re.split(r"\b(?:then|and|,|;)\b", steps[0])
            parts = [_normalize_step(p) for p in parts if p.strip()]
            # Keep unique and non-empty
            extra = []
            ps = set()
            for p in parts:
                if p and p not in ps:
                    extra.append(p); ps.add(p)
            # Replace first step with its split parts if that helps
            if len(extra) >= min_steps:
                steps = extra[:min_steps]
        # If still short, pad by repeating last (best-effort to match contract)
        while len(steps) < min_steps:
            steps.append(steps[-1] if steps else "perform the task")

    return {"steps": steps}

# ---------- tiny disk cache (safe for DDP rank0-only) ----------
def _sha1(s: str) -> str: return hashlib.sha1(s.encode()).hexdigest()

class StepsCache:
    def __init__(self, path: str = "planner_cache.jsonl", ttl_hours: float = 168):
        self.path = path
        self.ttl = ttl_hours * 3600.0
        self._lock = threading.Lock()
        self._mem: dict[str, dict] = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self._mem[rec["key"]] = rec
                    except Exception:
                        pass

    def get(self, key: str):
        now = time.time()
        with self._lock:
            rec = self._mem.get(key)
            if rec and (now - rec["ts"]) < self.ttl:
                return rec["value"]
        return None

    def put(self, key: str, value: dict):
        now = time.time()
        rec = {"key": key, "ts": now, "value": value}
        with self._lock:
            self._mem[key] = rec
            with open(self.path, "a") as f:
                f.write(json.dumps(rec) + "\n")

# ---------- online planner ----------
class OnlinePlanner:
    def __init__(self, model: str = "gemini-2.5-pro",
                 cache_path="planner_cache.jsonl", ttl_hours=168):
        self.model = model
        self.fallback_models = ["gemini-2.5-pro", "gemini-2.0-flash"]
        # Gemini API key from the environment (never hard-code secrets). If unset,
        # the planner serves cached plans and errors clearly on a cache miss.
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.cache = StepsCache(cache_path, ttl_hours)
        try:
            from google import genai  # pip install -U google-genai
            self.client = genai.Client(api_key=self.api_key) if self.api_key else None
        except Exception:
            self.client = None

        # Keep separate maps for train/test lookup, but tolerate missing files.
        # Resolve the train cache relative to this file so it works on any host
        # (the original hardcoded /home/... path is dead on /bigdata machines).
        default_train_cache = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "planner_cache.jsonl",
        )
        self.records = self._load_records_jsonl(default_train_cache)
        self.records_test = self._load_records_jsonl(cache_path)

    def _load_records_jsonl(self, file_path: str) -> Dict[str, Dict[str, List[str]]]:
        out: Dict[str, Dict[str, List[str]]] = {}
        if not file_path or not os.path.exists(file_path):
            return out

        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    key = obj.get("key")
                    value = obj.get("value")
                    if key is not None and value is not None:
                        out[key] = value
                except Exception:
                    continue
        return out

    def _is_valid_steps(self, value) -> bool:
        if not isinstance(value, dict):
            return False
        steps = value.get("steps")
        if not isinstance(steps, list):
            return False
        if len(steps) < 2:
            return False
        return all(isinstance(s, str) and len(s.strip()) > 0 for s in steps)

    def _gemini_call_with_fallback(
        self,
        user_msg: str,
        min_steps: int,
        max_steps: int,
    ) -> Dict[str, List[str]]:
        if self.client is None:
            raise RuntimeError(
                "Planner cache miss and no Gemini client available. Set GEMINI_API_KEY "
                "(and `pip install -U google-genai`) to query the LLM planner, or add the "
                "instruction to the planner cache (planner_cache.jsonl)."
            )
        candidates = []
        for name in [self.model, *self.fallback_models]:
            if isinstance(name, str) and name and name not in candidates:
                candidates.append(name)

        last_err = None
        for model_name in candidates:
            for attempt in range(3):
                try:
                    resp = self.client.models.generate_content(model=model_name, contents=user_msg)
                    text = getattr(resp, "text", None) or getattr(resp, "output_text", None) or str(resp)
                    parsed = parse_steps_output(text, min_steps=min_steps, max_steps=max_steps)
                    if self._is_valid_steps(parsed):
                        return parsed
                    raise ValueError(f"invalid steps payload from model={model_name}: {parsed!r}")
                except Exception as e:
                    last_err = e
                    if attempt < 2:
                        time.sleep(0.3 * (attempt + 1))

        raise RuntimeError(
            f"Planner Gemini API failed for models={candidates}"
        ) from last_err

    def plan_steps(self, instruction: str, min_steps=2, max_steps=5, mode='train') -> Dict[str, List[str]]:
        key = _sha1(f"{self.model}|{instruction}")

        # 1) In-memory cache lookup
        active_records = self.records if mode == "train" else self.records_test
        hit = active_records.get(key)
        if self._is_valid_steps(hit):
            return hit

        # 2) Shared TTL cache lookup
        hit = self.cache.get(key)
        if self._is_valid_steps(hit):
            active_records[key] = hit
            return hit

        # 3) API fallback on miss
        user_msg = (
            INSTRUCTION_PROMPT.replace("{{SYSTEM_PROMPT}}", SYSTEM_PROMPT).strip()
            + " "
            + instruction
        )

        parsed = self._gemini_call_with_fallback(
            user_msg=user_msg, min_steps=min_steps, max_steps=max_steps
        )
        self.cache.put(key, parsed)
        active_records[key] = parsed
        return parsed
