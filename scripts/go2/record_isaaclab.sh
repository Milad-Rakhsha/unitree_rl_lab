#!/bin/bash
# Record Isaac Lab policy playback using Newton headless (EGL) renderer.
# Produces an MP4 of the policy running in the Isaac Lab environment.
#
# Usage:
#   ./record_isaaclab.sh <checkpoint_path> [output.mp4] [--task TASK] [--num_steps 500]
#
# Examples:
#   ./record_isaaclab.sh logs/.../model_5000.pt isaaclab_bipedal.mp4
#   ./record_isaaclab.sh logs/.../model_5000.pt --task Unitree-Go2-Bipedal-Walk-Newton
#
# NOTE: Cannot run while Newton training is active (GPU/solver conflict).
# Requires: conda env 'go2', isaaclab-newton, isaaclab-visualizers.

set -euo pipefail

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate go2

SITE_PKG="$HOME/miniconda3/envs/go2/lib/python3.12/site-packages"
export KIT_DIR="$SITE_PKG/isaacsim/kit"
export TENSORS_BIN="$SITE_PKG/isaacsim/extscache/omni.physics.tensors-110.0.7+110.0.0.lx64.r.cp312.u7f4/bin"
export CONDA_LIB="$HOME/miniconda3/envs/go2/lib"
export LD_LIBRARY_PATH="${KIT_DIR}:${TENSORS_BIN}:${CONDA_LIB}:${LD_LIBRARY_PATH:-}"
export DISPLAY=${DISPLAY:-:99}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [output.mp4] [--task TASK] [--num_steps 500]"
    exit 1
fi

CHECKPOINT="$1"
shift

OUTPUT="isaaclab_recording.mp4"
if [ $# -gt 0 ] && [[ "$1" == *.mp4 ]]; then
    OUTPUT="$1"
    shift
fi

TASK="${TASK:-Unitree-Go2-Bipedal-Walk-Newton}"
NUM_STEPS="${NUM_STEPS:-500}"

# Parse remaining args
while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --num_steps) NUM_STEPS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$SCRIPT_DIR/../video"

python record_newton.py \
    --task "$TASK" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --num_steps "$NUM_STEPS" \
    --headless
