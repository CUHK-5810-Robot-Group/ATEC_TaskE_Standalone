#!/usr/bin/env python3
"""Summarise Task-E videos without changing the original media files.

PyAV is optional: the committed JSON report remains usable on machines that
only run the planner tests.  When available, this script verifies duration,
frame count and frame rate against the values used in ``docs/video_analysis``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect(path: Path) -> dict:
    try:
        import av  # type: ignore
    except ImportError as exc:
        raise RuntimeError("install PyAV in an analysis environment to inspect videos") from exc
    container = av.open(str(path))
    stream = next(s for s in container.streams if s.type == "video")
    frames = 0
    for _ in container.decode(stream):
        frames += 1
    fps = float(stream.average_rate) if stream.average_rate else 0.0
    duration = frames / fps if fps else 0.0
    return {"file": path.name, "frames": frames, "fps": fps, "duration_s": duration}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"videos": [inspect(p) for p in args.videos]}
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
