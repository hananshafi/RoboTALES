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
    def __init__(self, path: str = "planner_cache_libero.jsonl", ttl_hours: float = 168):
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
_OPENAI_MODEL_PREFIXES = ("gpt", "o1", "o3", "o4", "chatgpt")


def _infer_provider(model: str) -> str:
    """Pick the LLM provider from the model name: gpt*/o* -> OpenAI, else Gemini."""
    return "openai" if str(model).lower().startswith(_OPENAI_MODEL_PREFIXES) else "gemini"


class OnlinePlanner:
    def __init__(self, model: str = "gemini-2.5-pro",
                 cache_path="planner_cache_libero.jsonl", ttl_hours=168):
        self.model = model
        # Provider inferred from the model name (override by passing a gpt-*/o* model).
        self.provider = _infer_provider(model)
        # API key from the environment (never hard-code secrets). If unset, the
        # planner serves cached plans and errors clearly on a cache miss.
        self.cache = StepsCache(cache_path, ttl_hours)
        if self.provider == "openai":
            self.api_key = os.environ.get("OPENAI_API_KEY")
            try:
                from openai import OpenAI  # pip install -U openai
                self.client = OpenAI(api_key=self.api_key) if self.api_key else None
            except Exception:
                self.client = None
        else:
            self.api_key = os.environ.get("GEMINI_API_KEY")
            try:
                from google import genai  # pip install -U google-genai
                self.client = genai.Client(api_key=self.api_key) if self.api_key else None
            except Exception:
                self.client = None
        self.records = {}
        _cache_file = os.path.join(os.path.dirname(__file__), "planner_cache_libero.jsonl")
        with open(_cache_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                self.records[obj["key"]] = obj["value"]
        # train and test lookups share the shipped cache
        self.records_test = dict(self.records)

    def plan_steps(self, instruction: str, min_steps=2, max_steps=5, mode='train') -> Dict[str, List[str]]:
        #key = _sha1(f"{self.model}|{instruction}")
        key = instruction
        records = self.records if mode == "train" else self.records_test
        hit = records.get(key)
        if hit:
            return hit
        if self.client is None:
            raise RuntimeError(
                f"Planner cache miss for {instruction!r} and no '{self.provider}' client "
                "available. Set GEMINI_API_KEY (Gemini) or OPENAI_API_KEY (OpenAI) and install "
                "the matching SDK, or add this instruction to planner_cache_libero.jsonl."
            )
        user_msg= (INSTRUCTION_PROMPT.replace("{{SYSTEM_PROMPT}}", SYSTEM_PROMPT).strip() + " " + instruction)

        # user_msg = (
        #     SYSTEM_PROMPT.strip()
        #     + "\n\nTask instruction:\n"
        #     + instruction
        #     + "\n\nReturn ONLY the JSON object.\n"
        #     + ICL.strip()
        # )
        max_try = 3
        while max_try > 0:
            try:
                text = self._call_llm(self.model, user_msg)
                parsed = parse_steps_output(text, min_steps=min_steps, max_steps=max_steps)
                self.cache.put(key, parsed)
                return parsed
            except:
                max_try -= 1

    def _call_llm(self, model_name: str, user_msg: str) -> str:
        """Query the active provider (Gemini or OpenAI) and return the raw text."""
        if self.provider == "openai":
            resp = self.client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": user_msg}]
            )
            return resp.choices[0].message.content or ""
        resp = self.client.models.generate_content(model=model_name, contents=user_msg)
        return getattr(resp, "text", None) or getattr(resp, "output_text", None) or str(resp)
