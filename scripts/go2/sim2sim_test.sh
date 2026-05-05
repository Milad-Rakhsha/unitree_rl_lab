#!/bin/bash
# Run MuJoCo sim2sim transfer test for a trained checkpoint.
# Auto-detects Newton vs PhysX joint ordering from deploy.yaml.
#
# Usage:
#   ./sim2sim_test.sh <checkpoint_path> [options]
#
# Examples:
#   ./sim2sim_test.sh logs/rsl_rl/.../model_3000.pt --cmd-vx 0.3 --render
#   ./sim2sim_test.sh logs/rsl_rl/.../model_3000.pt --duration 30 --csv out.csv
#
# The script wraps scripts/sim2sim/sim2sim_v2.py with proper env setup.
# Use scene_flat.xml (no stairs) for clean sim2sim evaluation.

set -euo pipefail

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate go2
export DISPLAY=${DISPLAY:-:99}
export MUJOCO_GL=${MUJOCO_GL:-egl}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENE="${SCENE:-$(realpath "$SCRIPT_DIR/../../unitree_mujoco/unitree_robots/go2/scene_flat.xml" 2>/dev/null || echo "$HOME/Repos/GO2/unitree_mujoco/unitree_robots/go2/scene_flat.xml")}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [--cmd-vx 0.3] [--duration 30] [--render] [--csv out.csv]"
    exit 1
fi

CHECKPOINT="$1"
shift

cd "$SCRIPT_DIR/../sim2sim"

python sim2sim_v2.py \
    --checkpoint "$CHECKPOINT" \
    --scene "$SCENE" \
    "$@"
