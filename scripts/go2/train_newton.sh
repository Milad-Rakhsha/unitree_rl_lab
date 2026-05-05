#!/bin/bash
# Train Go2 bipedal walking policy with Newton (MuJoCo Warp) backend.
# Newton policies transfer directly to MuJoCo sim2sim without dynamics gap.
#
# Usage:
#   ./train_newton.sh [--seed 42] [--max_iterations 10000] [--num_envs 4096]
#
# Requires: conda env 'go2' with isaaclab + isaaclab-newton installed.
# GPU: ~3GB VRAM for 4096 envs.

set -euo pipefail

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate go2

SITE_PKG="$HOME/miniconda3/envs/go2/lib/python3.12/site-packages"
export KIT_DIR="$SITE_PKG/isaacsim/kit"
export TENSORS_BIN="$SITE_PKG/isaacsim/extscache/omni.physics.tensors-110.0.7+110.0.0.lx64.r.cp312.u7f4/bin"
export CONDA_LIB="$HOME/miniconda3/envs/go2/lib"
export LD_LIBRARY_PATH="${KIT_DIR}:${TENSORS_BIN}:${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export DISPLAY=${DISPLAY:-:99}

cd "$(dirname "$0")/../rsl_rl"

python train.py \
    --task Unitree-Go2-Bipedal-Walk-Newton \
    --num_envs "${NUM_ENVS:-4096}" \
    --max_iterations "${MAX_ITER:-10000}" \
    --headless \
    --seed "${SEED:-42}" \
    "$@"
