#!/bin/bash
# Setup environment for Go2 training
export PATH="$HOME/miniconda3/bin:$PATH"

SITE_PKG="$HOME/miniconda3/envs/go2/lib/python3.12/site-packages"
KIT_DIR="$SITE_PKG/isaacsim/kit"
TENSORS_BIN="$SITE_PKG/isaacsim/extscache/omni.physics.tensors-110.0.7+110.0.0.lx64.r.cp312.u7f4/bin"
CONDA_LIB="$HOME/miniconda3/envs/go2/lib"

export LD_LIBRARY_PATH="${KIT_DIR}:${TENSORS_BIN}:${CONDA_LIB}:${LD_LIBRARY_PATH}"
