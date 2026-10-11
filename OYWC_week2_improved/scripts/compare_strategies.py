#!/usr/bin/env python3
"""Compare ACT and RGB-D results without changing the scoring environment.

The script consumes one JSON record per episode.  It intentionally refuses to
compare records with different seed sets or termination definitions, because a
higher score under different conditions is not evidence of an improvement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required = {"strategy", "termination", "episodes"}
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    if not isinstance(data["episodes"], list) or not data["episodes"]:
        raise ValueError(f"{path}: episodes must be a non-empty list")
    return data


def seed_set(data: dict[str, Any]) -> set[Any]:
    return {episode.get("seed") for episode in data["episodes"]}


def metrics(data: dict[str, Any]) -> dict[str, float]:
    episodes = data["episodes"]
    complete = [bool(item.get("complete", False)) for item in episodes]
    successes = sum(complete)
    attempts = len(episodes)
    return {
        "episodes": float(attempts),
        "complete_successes": float(successes),
        "complete_success_rate": successes / attempts,
        "mean_time_s": sum(
            float(item.get("time_s", 0.0)) for item in episodes
        ) / attempts,
        "mean_retries": sum(
            float(item.get("retries", 0.0)) for item in episodes
        ) / attempts,
        "mean_objects": sum(
            float(item.get("objects_in_basket", 0.0)) for item in episodes
        ) / attempts,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("act", type=Path)
    parser.add_argument("rgbd", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    act, rgbd = load(args.act), load(args.rgbd)
    if act["termination"] != rgbd["termination"]:
        raise SystemExit("refusing comparison: termination definitions differ")
    if seed_set(act) != seed_set(rgbd):
        raise SystemExit("refusing comparison: episode seed sets differ")
    report = {
        "termination": act["termination"],
        "strategies": {
            str(act["strategy"]): metrics(act),
            str(rgbd["strategy"]): metrics(rgbd),
        },
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
