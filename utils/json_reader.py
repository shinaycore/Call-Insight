import json
import os
from typing import Any, Dict, List, Union

def load_json(path: str) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Load JSON file and return Python object."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Union[Dict[str, Any], List[Dict[str, Any]]], path: str) -> None:
    """Save Python object as JSON file."""
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
