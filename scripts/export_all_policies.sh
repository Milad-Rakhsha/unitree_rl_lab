#!/bin/bash
# Export all trained policies to ONNX format for deployment.
#
# To add a new policy: append one line to the POLICIES array below in the form
#   "Task-Id|experiment_name|run_name|checkpoint_filename"
#
# - Task-Id: gym task id passed to play.py (e.g. Unitree-Go2-Velocity-Rough)
# - experiment_name: log subdirectory (lowercase, underscores)
# - run_name: subdirectory of the run within experiment (e.g. 0_best)
# - checkpoint_filename: .pt file inside the run (e.g. best.pt)

set -u  # error on unset vars; do NOT use -e because the play step is expected to fail

ISAACLAB_ROOT="/home/milad/Documents/Repos/IsaacLab"
UNITREE_RL_LAB_ROOT="/home/milad/Documents/Repos/unitree_rl_lab"
PYTHON_EXE="${CONDA_PREFIX:-/home/milad/.conda/envs/go2}/bin/python"

cd "$UNITREE_RL_LAB_ROOT"

POLICIES=(
    "Unitree-Go2-Velocity-Rough|unitree_go2_velocity_rough|0_best|best.pt"
    "Unitree-Go2-Velocity-Rough-Pace|unitree_go2_velocity_rough_pace|0_best|best.pt"
    "Unitree-Go2-Velocity-Rough-Sport|unitree_go2_velocity_rough_sport|0_best|best.pt"
    "Unitree-Go2-Velocity-Rough-Gallop|unitree_go2_velocity_rough_gallop|0_best|best.pt"
    "Unitree-Go2-Stabilize|unitree_go2_stabilize|0_best|best.pt"
    "Unitree-Go2-Bipedal-Walk-Rough|unitree_go2_bipedal_walk_rough|0_best|best.pt"
    "Unitree-Go2-Bipedal-Standup|unitree_go2_bipedal_standup|0_best|best.pt"
)

TOTAL=${#POLICIES[@]}
SUCCESS=()
FAILED=()

echo "=================================="
echo "Exporting Policies to ONNX ($TOTAL total)"
echo "=================================="
echo ""

export_policy() {
    local idx="$1"
    local task="$2"
    local experiment="$3"
    local run="$4"
    local checkpoint="$5"

    local run_dir="$UNITREE_RL_LAB_ROOT/logs/rsl_rl/$experiment/$run"
    local ckpt_path="$run_dir/$checkpoint"
    local jit_path="$run_dir/exported/policy.pt"
    local onnx_path="$run_dir/exported/policy.onnx"

    echo "[$idx/$TOTAL] $task"
    echo "  run: $run_dir"

    if [ ! -f "$ckpt_path" ]; then
        echo "  SKIP: checkpoint not found: $ckpt_path"
        FAILED+=("$task (no checkpoint)")
        echo ""
        return
    fi

    # Step 1: produce JIT via Isaac Lab play.py.
    # play.py exports JIT before constructing the env, so the PhysX import
    # error from gym.make() is harmless for our purposes. Suppress non-fatal
    # noise but keep the command's exit code from killing the script.
    echo "  [1/2] running play.py to export JIT..."
    "$ISAACLAB_ROOT/isaaclab.sh" -p "$UNITREE_RL_LAB_ROOT/scripts/rsl_rl/play.py" \
        --task "$task" \
        --num_envs 1 \
        --checkpoint "$ckpt_path" \
        --headless > /dev/null 2>&1 || true

    if [ ! -f "$jit_path" ]; then
        echo "  FAIL: JIT not produced at $jit_path"
        FAILED+=("$task (no JIT)")
        echo ""
        return
    fi

    # Step 2: convert JIT to ONNX using standalone script (legacy torch.onnx).
    echo "  [2/2] converting JIT -> ONNX..."
    if "$PYTHON_EXE" "$UNITREE_RL_LAB_ROOT/scripts/export_policy_standalone.py" \
        --jit_path "$jit_path" > /dev/null 2>&1; then
        if [ -f "$onnx_path" ]; then
            local size_kb
            size_kb=$(du -k "$onnx_path" | cut -f1)
            echo "  OK: $onnx_path (${size_kb} KB)"
            SUCCESS+=("$task")
        else
            echo "  FAIL: ONNX not produced at $onnx_path"
            FAILED+=("$task (no ONNX)")
        fi
    else
        echo "  FAIL: ONNX conversion failed"
        FAILED+=("$task (conversion error)")
    fi
    echo ""
}

i=1
for entry in "${POLICIES[@]}"; do
    IFS='|' read -r task experiment run checkpoint <<< "$entry"
    export_policy "$i" "$task" "$experiment" "$run" "$checkpoint"
    i=$((i + 1))
done

echo "=================================="
echo "Post-step: freeze batch dim to 1"
echo "=================================="
# Deploy-side ORT wrapper allocates input tensors from the model's declared
# input shape. Any dynamic axis (e.g. 'batch') comes back as -1 and crashes
# CreateTensor. The standalone exporter above already produces static-batch
# ONNX, but if a policy was exported through Isaac Lab's play.py --export_policy
# (or pulled in from upstream), the resulting file may carry dynamic axes.
# This pass normalizes every ONNX under logs/rsl_rl/ to static batch=1.
"$PYTHON_EXE" "$UNITREE_RL_LAB_ROOT/scripts/freeze_onnx_batch.py" \
    "$UNITREE_RL_LAB_ROOT/logs/rsl_rl" || echo "  WARN: freeze_onnx_batch.py reported issues"
echo ""

echo "=================================="
echo "Post-step: sanitize deploy.yaml"
echo "=================================="
# Some Isaac Lab export paths emit ``joint_ids: None`` (Python repr) instead
# of the YAML null literal. yaml-cpp can't coerce that into vector<int>, so
# JointPositionAction throws TypedBadConversion at state-init time. Rewrite
# any such instance to lowercase ``null`` before the deploy controller sees
# the file.
"$PYTHON_EXE" "$UNITREE_RL_LAB_ROOT/scripts/sanitize_deploy_yaml.py" \
    "$UNITREE_RL_LAB_ROOT/logs/rsl_rl" || echo "  WARN: sanitize_deploy_yaml.py reported issues"
echo ""

echo "=================================="
echo "Summary"
echo "=================================="
echo "  succeeded: ${#SUCCESS[@]} / $TOTAL"
for t in "${SUCCESS[@]}"; do echo "    - $t"; done
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  failed:    ${#FAILED[@]} / $TOTAL"
    for t in "${FAILED[@]}"; do echo "    - $t"; done
    exit 1
fi
