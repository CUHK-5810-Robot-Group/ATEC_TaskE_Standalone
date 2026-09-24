#!/bin/bash
# Usage: ./run_view_task_e.sh --headless --num_envs 1 --enable_cameras
cd "$(dirname "$0")"
exec ./run_env.sh python scripts/view_task_e.py "$@"
