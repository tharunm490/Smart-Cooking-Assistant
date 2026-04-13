from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "recipe"


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        cleaned = cleaned[first_brace : last_brace + 1]

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        repaired = cleaned
        # Remove trailing commas before object/array close.
        repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
        # Quote bare numeric values that include units (for example: 20g, 450kcal).
        repaired = re.sub(
            r'(:\s*)([-+]?\d+(?:\.\d+)?)\s*(kcal|g|mg|ml|minutes?|hours?|hrs?)\b(?=\s*[,}\]])',
            r'\1"\2 \3"',
            repaired,
            flags=re.IGNORECASE,
        )
        return json.loads(repaired)


def normalize_text_list(values: list[str]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        item = value.strip().lower()
        if not item or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def join_recipe_steps(steps: list[str]) -> str:
    return " ".join(step.strip() for step in steps if step.strip())
