#!/bin/bash
# Test sim2sim transfer for multiple checkpoints
# Usage: ./test_checkpoints.sh <run_dir> [start_iter] [step] [end_iter]

RUN_DIR="${1:-$HOME/Repos/GO2/unitree_rl_lab/logs/rsl_rl/unitree_go2_bipedal_walk/2026-05-04_05-01-35}"
START="${2:-1000}"
STEP="${3:-1000}"
END="${4:-10000}"

export PATH="$HOME/miniconda3/bin:$PATH"
cd ~/Repos/GO2

NOMINAL_YAML="/tmp/nominal_deploy.yaml"
cat > "$NOMINAL_YAML" << 'EOF'
joint_ids_map: [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
step_dt: 0.02
stiffness: [25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0, 25.0]
damping: [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
default_joint_pos: [0.1, -0.1, 0.1, -0.1, 0.8, 0.8, 1.0, 1.0, -1.5, -1.5, -1.5, -1.5]
EOF

echo "Testing checkpoints from $RUN_DIR"
echo "Iter | Survived(s) | Max Height | Final Pitch"
echo "------|-------------|------------|------------"

for iter in $(seq $START $STEP $END); do
    CKPT="$RUN_DIR/model_${iter}.pt"
    if [ ! -f "$CKPT" ]; then
        continue
    fi
    
    OUTPUT=$(conda run -n go2 python sim2sim_bipedal.py \
        --checkpoint "$CKPT" \
        --deploy-yaml "$NOMINAL_YAML" \
        --duration 20.0 \
        2>&1)
    
    # Parse output
    SURVIVED=$(echo "$OUTPUT" | grep "ended at" | grep -oP 't=\K[0-9.]+')
    FELL=$(echo "$OUTPUT" | grep "FELL at" | grep -oP 't=\K[0-9.]+')
    
    if [ -n "$FELL" ]; then
        TIME="$FELL"
        STATUS="FELL"
    else
        TIME="$SURVIVED"
        STATUS="OK"
    fi
    
    # Get max height from progress lines
    MAX_Z=$(echo "$OUTPUT" | grep "BipedalPolicy" | grep -oP 'z=\K[0-9.]+' | sort -rn | head -1)
    
    printf "%5d | %7ss %4s | z=%-8s |\n" "$iter" "$TIME" "$STATUS" "$MAX_Z"
done
