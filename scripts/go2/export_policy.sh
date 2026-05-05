#!/bin/bash
# Export a training checkpoint to JIT torchscript for deployment.
#
# Usage:
#   ./export_policy.sh <checkpoint.pt> [output_policy.pt]
#
# Examples:
#   ./export_policy.sh logs/.../model_3000.pt exported_policy.pt
#
# The exported JIT model can be loaded directly by sim2sim or C++ deploy code
# without needing the rsl_rl training framework.

set -euo pipefail

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate go2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint.pt> [output_policy.pt]"
    exit 1
fi

CHECKPOINT="$1"
OUTPUT="${2:-$(dirname "$CHECKPOINT")/policy_jit.pt}"

cd "$SCRIPT_DIR/../sim2sim"

python export_policy.py \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT"

echo "Exported: $OUTPUT"
