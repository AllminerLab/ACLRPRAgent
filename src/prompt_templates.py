"""English prompt templates transcribed from Supplementary Tables S9 and S10.

Templates retain the reported prediction horizons and output field names.
Runtime record names and clinical values are passed through without translation.
"""
import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
PROMPTS = json.loads((_ROOT / "prompt_templates.json").read_text(encoding="utf-8"))
RETRIEVAL_TASKS = json.loads((_ROOT / "retrieval_tasks.json").read_text(encoding="utf-8"))

def system_prompt(name):
    return PROMPTS[name]["system"]

def render_prompt(name, **values):
    template = PROMPTS[name]
    body = template["user"]
    placeholders = set(re.findall(r"\{([^{}]+)\}", body))
    if placeholders != set(values):
        raise ValueError(f"Prompt {name}: expected inputs {sorted(placeholders)}, got {sorted(values)}")
    # Substitute once: braces inside patient evidence are never interpreted.
    body = re.sub(r"\{([^{}]+)\}", lambda match: str(values[match.group(1)]), body)
    if template.get("example"):
        body += "\n\n" + template["example"]
    return body + "\n\nRequired output: " + template["output"]

def retrieval_goal(index, baseline=None):
    goal = RETRIEVAL_TASKS[index]["goal"]
    if baseline is not None:
        goal += "\nPatient baseline profile: " + str(baseline)
    return goal
