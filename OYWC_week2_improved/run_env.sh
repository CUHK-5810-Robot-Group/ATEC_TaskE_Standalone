#!/bin/bash
# ATEC Task E isolated runtime wrapper
set -e
cd "$(dirname "$0")"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/source/atec_rl_lab:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/miniconda3/envs/atec_task_e/bin:$PATH"
exec "$@"
