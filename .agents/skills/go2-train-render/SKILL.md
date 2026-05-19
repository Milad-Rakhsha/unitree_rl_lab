---
name: go2-train-render
description: >
  Train, evaluate, render, and record videos for Unitree Go2 locomotion policies
  using Isaac Lab with Newton MjWarp on headless servers. Use when asked to:
  train a walking/standup/stabilize policy, tune reward terms, render policy
  videos on headless machines (no display), run play.py, capture side-by-side
  comparisons, quickly preview policies in standalone MuJoCo, export policies
  to ONNX/JIT for deployment, or do reward engineering for bipedal/quadruped
  locomotion tasks. Also covers the excessive_velocity reward pattern for
  slow-standup training.
---

# Go2 Training & Rendering

Train RL policies in Isaac Lab (Newton MjWarp), record videos on headless
servers, and quickly preview in standalone MuJoCo.

## Repo Layout

```
unitree_rl_lab/
├── scripts/rsl_rl/
│   ├── train.py              # Training entry point
│   ├── play.py               # Play/evaluate + optional video
│   └── play_deploy_fsm.py    # Deploy-style playback
├── scripts/
│   ├── export_all_policies.sh       # Batch export to ONNX
│   └── export_policy_standalone.py  # JIT -> ONNX converter
├── source/unitree_rl_lab/.../go2/
│   ├── standup_env_cfg.py    # Bipedal standup env + rewards
│   ├── standup_mdp.py        # Custom standup MDP functions
│   ├── bipedal_env_cfg.py    # Bipedal walk env + rewards
│   ├── velocity_env_cfg.py   # Quadruped velocity env
│   └── stabilization_env_cfg.py  # Recovery/stabilize env
└── logs/rsl_rl/
    └── <experiment>/
        ├── <run_timestamp>/  # Training run outputs
        │   ├── model_*.pt    # Checkpoints
        │   └── ...
        └── 0_best/           # Best policy for deployment
            ├── best.pt
            ├── exported/policy.onnx
            ├── exported/policy.pt (JIT)
            └── params/deploy.yaml
```

## Environment & Conda

```bash
conda activate go2
cd ~/Repos/GO2/unitree_rl_lab
```

Server: headless Linux with NVIDIA GPU (L40 or similar).
Physics backend: **Newton MjWarp only** (PhysX does not work for these tasks).

## Training

### Launch Training

```bash
# In a tmux session (NOT nohup — sessions die after ~28 min)
tmux new -s train
cd ~/Repos/GO2/unitree_rl_lab

python scripts/rsl_rl/train.py \
  --task Unitree-Go2-Bipedal-Walk \
  --num_envs 4096 \
  --max_iterations 20000 \
  --headless
```

**Always use tmux for training.** Processes launched via `nohup ... &` or
`disown` die silently after ~28 minutes because the parent exec session
terminates and kills orphaned children.

### Available Tasks

| Task ID | Description |
|---------|-------------|
| `Unitree-Go2-Bipedal-Standup` | Bipedal standup (flat) |
| `Unitree-Go2-Bipedal-Standup-Rough` | Bipedal standup (rough terrain) |
| `Unitree-Go2-Bipedal-Walk` | Bipedal walking (flat) |
| `Unitree-Go2-Bipedal-Walk-Rough` | Bipedal walking (rough terrain) |
| `Unitree-Go2-Velocity` | Quadruped velocity tracking |
| `Unitree-Go2-Velocity-Flat` | Quadruped flat terrain |
| `Unitree-Go2-Velocity-Rough` | Quadruped rough terrain |
| `Unitree-Go2-Stabilize` | Recovery/stabilize policy |

### Resume Training

```bash
python scripts/rsl_rl/train.py \
  --task Unitree-Go2-Bipedal-Walk \
  --num_envs 4096 \
  --max_iterations 20000 \
  --headless \
  --resume \
  --load_run <run_dir_name> \
  --checkpoint model_NNNN.pt
```

### Monitor Progress

Training logs to stdout. Key metrics to watch:
- `mean_reward`: total reward (higher = better)
- Per-reward-term breakdowns (vary by task)
- For bipedal standup: `standing_success`, `excessive_velocity_penalty`
- For bipedal walk: `orientation_align`, `base_velocity_tracking`

## Reward Engineering

### General Approach

**One surgical change at a time.** Tuning multiple reward terms simultaneously
almost always fails. Start from a working config and add/modify one term.

### Slow Standup Pattern (excessive_velocity)

To make a policy move slowly (e.g., gradual standup instead of a snap):

```python
# In standup_mdp.py
def excessive_velocity_penalty(env, desired_gravity, vel_threshold=2.0,
                                release_cos=0.85, asset_cfg=...):
    """Returns 1.0 when ||lin_vel||^2 + ||ang_vel||^2 > threshold
    AND cos(gravity, desired) < release_cos. Auto-releases when upright."""
```

Add to env config rewards:
```python
excessive_velocity = RewTerm(
    func=standup_mdp.excessive_velocity_penalty,
    weight=-20.0,
    params={"desired_gravity": DESIRED_GRAVITY, "vel_threshold": 2.0,
            "release_cos": 0.85}
)
```

This forces the policy to move slowly: fast motion is penalized until the
robot is nearly upright (cos >= 0.85), at which point the penalty auto-disables.

### Integrating Rewards Across Tasks

The same reward function can be imported into other env configs:
```python
# In bipedal_env_cfg.py
from . import standup_mdp
excessive_velocity = RewTerm(
    func=standup_mdp.excessive_velocity_penalty,
    weight=-20.0, ...
)
```

### Bipedal Rough Terrain Issues

Bipedal rough terrain causes NaN solver divergence. Root cause: robot falls,
body slams into triangle mesh, extreme contact count overflows solver.
Mitigation: tighter reset terminations (`base_too_low`, `bad_orientation`)
to catch falls early. `euler` integrator degrades gracefully vs `implicitfast`
which NaNs on njmax overflow.

## Recording Videos

### Method 1: Isaac Lab play.py (built-in)

```bash
python scripts/rsl_rl/play.py \
  --task Unitree-Go2-Bipedal-Standup \
  --num_envs 16 \
  --checkpoint logs/rsl_rl/.../model_19999.pt \
  --headless \
  --video \
  --video_length 500
```

Outputs to `logs/rsl_rl/<experiment>/<run>/videos/play/`.

**Caveat**: The built-in video recorder uses a static world-coordinate camera.
It does NOT track the robot. For robot-tracking video, use Method 2.

### Method 2: Custom rgb_array Script (recommended for headless)

Write a script that uses `render_mode="rgb_array"` and captures frames via
`env.unwrapped.render()`. This is the only reliable method for headless
video with Newton MjWarp.

Key pattern:
```python
from isaaclab.app import AppLauncher
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

# ... import gym, tasks, etc. AFTER AppLauncher ...

env_cfg.sim.enable_newton_rendering = True
env = gym.make(task_id, cfg=env_cfg, render_mode="rgb_array")

# ... run policy ...
frame = env.unwrapped.render()  # returns np.ndarray (H, W, 3)
```

**Critical**: `enable_cameras=True` in AppLauncher AND
`env_cfg.sim.enable_newton_rendering = True` are both required.

The `env.unwrapped.render()` call returns a numpy frame. The built-in
`env.render()` may return None — always use `env.unwrapped.render()`.

### Method 3: Standalone MuJoCo Preview (fastest iteration)

For quick policy preview without Isaac Lab overhead, load the `.pt` checkpoint
directly in standalone MuJoCo:

```python
import mujoco
os.environ['MUJOCO_GL'] = 'egl'  # for headless

m = mujoco.MjModel.from_xml_path(scene_xml)
d = mujoco.MjData(m)
renderer = mujoco.Renderer(m, height, width)

# Load policy from .pt checkpoint
ckpt = torch.load(path, map_location=device, weights_only=False)
# Build actor network matching the checkpoint architecture
actor = build_actor(obs_dim=ckpt_obs_dim)
# ... load state dict, run inference loop, render frames ...
```

This skips the full Isaac Lab env setup and is much faster for visual checks.
However, observations must be manually constructed (no env wrapper).

### Side-by-Side Comparison Videos

To compare two policies (e.g., old vs new):
1. Record frames from each policy separately using Method 2
2. Concatenate frames horizontally with labels using OpenCV:
```python
import cv2
cv2.putText(frame, "OLD", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
sep = np.full((h, 4, 3), 255, dtype=np.uint8)
combined = np.concatenate([old_frame, sep, new_frame], axis=1)
```

**Use only ASCII text** — OpenCV `putText` does not render Unicode.

### Video Tips

- Use `imageio.get_writer(path, fps=50, codec="libx264", quality=8)` for
  Isaac Lab videos (sim runs at 50Hz = 0.02s step_dt)
- For MuJoCo standalone: fps matches your render interval
- Compress for sharing: use ffmpeg with libopenh264 or libx264 + scale filter

## Exporting Policies for Deployment

### Batch Export

```bash
bash scripts/export_all_policies.sh
```

Edit the `POLICIES` array in the script to add new policies.

### Manual Export

1. Run play.py to produce JIT:
```bash
python scripts/rsl_rl/play.py --task <task> --num_envs 1 \
  --checkpoint <path/to/best.pt> --headless
```

2. Convert JIT to ONNX:
```bash
python scripts/export_policy_standalone.py --jit_path <path/to/policy.pt>
```

### 0_best Convention

Each experiment has a `0_best/` directory with the best policy:
```
logs/rsl_rl/<experiment>/0_best/
├── best.pt           # Best checkpoint
├── exported/
│   ├── policy.pt     # JIT traced
│   └── policy.onnx   # ONNX export
└── params/
    └── deploy.yaml   # Joint mappings, PD gains, etc.
```

The `logs/` directory is gitignored. To track `0_best/` in git:
```bash
git add -f logs/rsl_rl/<experiment>/0_best/
git lfs push --all origin  # .pt/.jit files are LFS-tracked
git push
```

## Loading a Policy Manually

The actor network architecture (for rsl_rl checkpoints):
```python
def build_actor(obs_dim, act_dim=12, hidden=[512, 256, 128]):
    layers = []
    in_dim = obs_dim
    for h in hidden:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ELU())
        in_dim = h
    layers.append(nn.Linear(in_dim, act_dim))
    return nn.Sequential(*layers)

ckpt = torch.load(path, map_location=device, weights_only=False)
sd = ckpt["actor_state_dict"]
# Keys are prefixed — find obs_dim from first layer weight shape
sample_key = [k for k in sd if "mlp.0.weight" in k][0]
prefix = sample_key.replace("mlp.0.weight", "")
obs_dim = sd[sample_key].shape[1]
actor = build_actor(obs_dim)
new_sd = {k[len(prefix+"mlp."):]: v for k, v in sd.items()
          if k.startswith(prefix+"mlp.")}
actor.load_state_dict(new_sd)
```

## Env Config Structure

Each task has a `*_env_cfg.py` with nested config classes:
- `SceneCfg`: robot asset, terrain, num_envs
- `ActionsCfg`: joint position actions, PD gains
- `ObservationsCfg`: observation terms (policy group)
- `RewardsCfg`: reward terms with weights
- `TerminationsCfg`: episode reset conditions
- `CommandsCfg`: velocity command ranges

Presets are resolved via:
```python
from isaaclab_tasks.utils.hydra import resolve_presets
resolve_presets(env_cfg, {"newton_mjwarp"})
```

This sets the physics backend to Newton MjWarp (required for all Go2 tasks).
