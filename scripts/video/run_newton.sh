#!/bin/bash
export PATH="$HOME/miniconda3/envs/go2/bin:$PATH"
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate go2

SITE_PKG="$HOME/miniconda3/envs/go2/lib/python3.12/site-packages"
export KIT_DIR="$SITE_PKG/isaacsim/kit"
export TENSORS_BIN="$SITE_PKG/isaacsim/extscache/omni.physics.tensors-110.0.7+110.0.0.lx64.r.cp312.u7f4/bin"
export CONDA_LIB="$HOME/miniconda3/envs/go2/lib"
export LD_LIBRARY_PATH="${KIT_DIR}:${TENSORS_BIN}:${CONDA_LIB}:${LD_LIBRARY_PATH}"
export DISPLAY=:99

cd ~/Repos/GO2
python record_newton.py --num_steps 500 \
    --checkpoint /home/horde/Repos/GO2/unitree_rl_lab/logs/rsl_rl/unitree_go2_bipedal_walk/2026-04-21_15-56-29/model_10000.pt \
    --output /home/horde/Repos/GO2/isaaclab_bipedal_policy.mp4 2>&1
