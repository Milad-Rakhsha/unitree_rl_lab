# Go2: velocity policy + Surprise forward model (training and deployment)

This document describes end-to-end steps to train a **base walking** policy, build a **Surprise** forward dynamics model that matches that policy’s observations, export assets for the **C++ deploy** stack, and what to edit in configuration files.

Assumptions:

- You use **Isaac Lab** with this repo (`unitree_rl_lab`) installed as a package.
- You have a separate **`Surprise`** repository for forward-model collection, training, and ONNX export (clone it next to `unitree_rl_lab` if you use the default relative paths below).
- Target robot workflow: **FSM `Velocity`** runs the walking policy; optional **Surprise** gate uses `forward_model.onnx` + `normalization_stats.yaml`.

For **how states hand off**, what **“mixing”** means, the **Surprise NLL** formula, and **every `surprise:` YAML knob**, read **section 2** first.

---

## 1. Repository layout (recommended)

Place the two projects as siblings so the default deploy paths resolve:

```text
<workspace>/
  unitree_rl_lab/          # this repo
  Surprise/                # forward model tooling
  IsaacLab/                # Isaac Lab install (typical)
```

The Go2 deploy config uses paths relative to `unitree_rl_lab/deploy/robots/go2/` that point at `Surprise/data/go2/` (four levels up to the workspace parent, then into `Surprise`). If your layout differs, adjust `model_path` and `stats_path` in `config.yaml` (see section 6).

---

## 2. Runtime behavior: FSM, policy “mixing”, and Surprise

### 2.1 Policies are not blended

At any instant, **one** controller generates joint targets: the **velocity** RL policy, the **stabilize** RL policy, or scripted **Passive** / **FixStand**. Outputs are **not** mixed with learned weights.

**“Mixing”** here means: in **Velocity**, the walking ONNX runs **and** a **forward model** runs **in parallel** only to compute a **surprise score**. That score can request an **FSM transition** (usually to **Stabilize**). After switching states, the previous policy thread **stops**; the new state’s RL or scripted logic runs.

### 2.2 FSM states (typical Go2)

| State | Purpose |
|-------|---------|
| **Passive** | Safe / damped mode; transitions driven by joystick rules in YAML. |
| **FixStand** | Scripted stand / interpolation (`kp`, `kd`, `ts`, `qs`); optional path back to **Velocity**. |
| **Velocity** | Locomotion: `policy_dir/exported/policy.onnx` plus optional **Surprise** ONNX under `FSM.Velocity.surprise`. |
| **Stabilize** | Optional recovery RL: **its own** `policy_dir/exported/policy.onnx`. |

In `config.yaml`, `FSM._` lists which states exist and their numeric `id`. States with `type: RLBase` share `State_RLBase` C++ (`deploy/robots/go2/src/State_RLBase.cpp`) with behavior branches per state name.

### 2.3 Control loop in `Velocity`

1. Each control period (`step_dt` from `params/deploy.yaml` in the velocity run), the deploy stack builds the **policy observation** vector to match training.
2. The **walking** policy ONNX maps `obs → action`.
3. If **`FSM.Velocity.surprise.enabled`**, the Surprise network evaluates **previous** `(obs, action)` against **current** `obs` as the predicted “next” (section 2.4). Surprise **does not** replace the action for this step; it only updates internal gate state.
4. Independently, a **bad-orientation** check can force **Passive** when the base tilt exceeds limits (`bad_orientation` in `State_RLBase`).

### 2.4 Surprise score (Gaussian negative log-likelihood)

Implementation: `State_RLBase::update_surprise_gate`. Observations and actions are normalized with **`normalization_stats.yaml`** (`obs_mean`, `obs_std`, `action_mean`, `action_std`). The Surprise ONNX outputs per-dimension Gaussian **mean** and **log-variance** of the next observation in **normalized** space.

For each dimension \(i\) (with \(\sigma^2_i = e^{\text{logvar}_i}\), log-var clamped for stability):

\[
\text{nll}_i = \tfrac{1}{2}\left( \text{logvar}_i + \frac{(o^{\text{next}}_i - \mu_i)^2}{\sigma^2_i} \right)
\]

**Raw surprise** for the step is the **sum** over \(i\) of \(\text{nll}_i\). This matches Surprise offline metrics such as **NLL (sum)** / percentiles in `test_metrics.npz` from `train_forward_model.py`.

The **first** scored cycle after entering **Velocity** only stores the initial \((obs, action)\); surprise is computed from the **second** step onward (paired transition \((o_t,a_t)\to o_{t+1}\)).

### 2.5 Surprise parameters (`FSM.Velocity.surprise`)

| Parameter | Meaning |
|-----------|---------|
| `enabled` | `false`: do not load Surprise ONNX; no Surprise-driven transition. |
| `model_path` | `forward_model.onnx`. Must expose **two** inputs named like `obs` and `action`, and **one** or **two** outputs (`mean`/`logvar` or concatenated). Relative paths resolve from the deploy project directory. |
| `stats_path` | `normalization_stats.yaml` with `obs_mean`, `obs_std`, `action_mean`, `action_std` — vector sizes must match live obs/action dimensions. |
| `threshold` | Compare to the **gate value** (see below). Same numeric scale as **summed** NLL in normalized space. **Tuning:** start **high**, use `log_every_steps` to read typical raw/EMA values during normal walking, then lower until only meaningful disturbances trip. |
| `use_ema` | `true`: gate compares **EMA-smoothed** surprise to `threshold`. `false`: compare **raw** per-step sum. EMA dampens single-frame spikes. |
| `ema_alpha` | EMA update: \(\text{ema} \leftarrow \alpha\cdot\text{raw} + (1-\alpha)\cdot\text{ema}\). Larger \(\alpha\) tracks faster. |
| `min_steps` | No trip until at least this many surprise evaluations have run (after the initial pairing step). Reduces startup transients. |
| `consecutive_steps` | After `min_steps`, require the gate value to stay **strictly above** `threshold` for this many **consecutive** control ticks to arm `surprise_trip`. If the gate drops at or below threshold, the counter **resets**. Use `1` for “trip as soon as over threshold”; use `3+` to demand sustained high surprise. |
| `log_every_steps` | If `> 0`, logs raw surprise, EMA, threshold, and step index on that interval. Set `0` to disable periodic logs. |

**Gate value:** `gate = use_ema ? ema : raw`. **Trip** when `gate > threshold` **and** the consecutive-over-threshold counter reaches `consecutive_steps` (subject to `min_steps`).

### 2.6 Where Surprise transitions go

- If **`Stabilize`** is listed in `FSM._`, Surprise registers a transition to **Stabilize**.
- Else if **`FixStand`** exists, code **falls back** to **FixStand** (log: “Surprise fallback”).
- If neither exists, Surprise may still load but **automatic transition is disabled** (warning log).

### 2.7 `Stabilize` (recovery policy)

| Parameter | Meaning |
|-----------|---------|
| `policy_dir` | RSL-RL run directory with `exported/policy.onnx` and `params/deploy.yaml` for the **recovery** task (e.g. stabilization training). **Independent** from **Velocity** `policy_dir`. |
| `auto_velocity.enabled` | If `true`, start a **wall-clock** timer when entering **Stabilize**. |
| `auto_velocity.duration_s` | After this many **seconds**, register an automatic transition back to **Velocity** (only if **Velocity** exists in `FSM._`). |

Manual joystick transitions in YAML (e.g. **Start** → Velocity, **LT+B** → Passive) still apply on top of auto rules.

### 2.8 Consistency between walking policy, Surprise, and recovery

- **Velocity** `policy_dir` and the RSL-RL checkpoint used in **Surprise data collection** should be the **same** walking policy (and **same** `--task`) so observation sizes and behavior match.
- **Surprise** must be retrained whenever the walking policy or observation stack changes materially.
- **Stabilize** is a **separate** policy: train, export with `play.py`, point `FSM.Stabilize.policy_dir` at that run.

### 2.9 Practical threshold tuning

1. Deploy with a **high** `threshold` and modest `log_every_steps` (e.g. 200). Read `spdlog` lines for **raw** and **ema** during normal walking to see the typical range.
2. Optionally compare to offline percentiles in Surprise’s `forward_model/test_metrics.npz` (**nll_sum** mean / 95th / 99th) — same scale as on-robot summed NLL if normalization matches.
3. Lower `threshold` until slips, bumps, or pushes you care about trip the gate; if normal walking trips, **raise** `threshold`, increase `consecutive_steps`, or lower `ema_alpha` for more smoothing.
4. If Surprise never fires when expected, confirm **obs/action dimensions** match `stats_path` (dimension mismatch warnings in log) and that the forward model was trained on **this** walking policy.

---

## 3. Train the base walking policy (Go2 velocity)

### 3.1 Tasks

| Gym task | Use case |
|----------|----------|
| `Unitree-Go2-Velocity` | Full training with mixed terrain / friction domain randomization (slower steps, closer to rough real floors). |
| `Unitree-Go2-Velocity-Flat` | Faster iteration on an infinite plane (optional pretrain or debugging). |

Both register the same observation layout for the **walking** policy used on the robot, as long as you keep the same task family when collecting Surprise data (section 4).

### 3.2 Command

From **`unitree_rl_lab`** (with Isaac Sim / Isaac Lab environment active), typical training:

```bash
cd /path/to/unitree_rl_lab

# Example: use the project helper (recommended in README)
./unitree_rl_lab.sh -t --task Unitree-Go2-Velocity --headless

# Equivalent idea (paths vary with your Isaac Lab install):
# python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity --headless
```

Tune `--num_envs`, run name, and training duration as usual. Checkpoints appear under:

```text
logs/rsl_rl/unitree_go2_velocity/<timestamp>/model_<N>.pt
```

(or `unitree_go2_velocity_flat` if you train the flat task—experiment name comes from the RSL-RL agent config).

**Important for Surprise:** whatever checkpoint you later use for rollout collection, train or choose a policy whose **task name matches** the environment you use in `collect_rollout_data.py` (`--task`, section 4). Mismatched tasks usually mean mismatched observation sizes and broken forward models.

---

## 4. Surprise forward model: data collection and training

The forward model predicts the next **policy observation** from `(obs_t, action_t)`. It must be trained on trajectories produced by **the same policy** (and ideally the same task) you deploy for walking.

### Command distribution during collection (linear + angular, spikes)

**`collect_rollout_data.py` does not define a separate “command schedule.”** It uses the task **`env_cfg`** loaded for `--task` (Hydra + `velocity_env_cfg.py`). Command behavior comes from **`CommandsCfg.base_velocity`** (and any **Hydra overrides** you append after the script args).

For **`Unitree-Go2-Velocity`**, Isaac Lab’s **`UniformVelocityCommand`** resamples **`lin_vel_x`**, **`lin_vel_y`**, and **`ang_vel_z` as independent uniforms** over `ranges` (see `IsaacLab/.../velocity_command.py`, `_resample_command`). There is **no built-in option** to forbid “forward + yaw at the same time”; combined commands appear whenever each axis samples a non-zero value.

| Field | Role |
|-------|------|
| **`ranges`** | Active sampling box for each axis. Training can **widen** these over time via **`lin_vel_cmd_levels`** (`tasks/locomotion/mdp/curriculums.py`); collection uses whatever the env is configured with (often full **`limit_ranges`** in play, or your overrides). |
| **`limit_ranges`** | Upper caps for curriculum / play (`RobotPlayEnvCfg` sets `ranges = limit_ranges`). |
| **`resampling_time_range`** | How often new commands are drawn (e.g. every 10 s). **Command jumps** change **`velocity_commands` in the observation**, which alone can spike Surprise until the dynamics settle. |
| **`rel_standing_envs`** | Probability some envs get **zero** velocity command. Adds stand-still data; it does **not** decorrelate lin vs ang when both are active. |

**Why combined lin + ang feels “sensitive”:** the forward model’s NLL is **data-driven**. If simultaneous large linear and angular commands are **underrepresented** in rollouts (or unlike your joystick use on the robot), errors and Surprise scores grow in that region.

**Practical levers:** (1) **Hydra-style overrides** when launching collect to narrow `ranges` (e.g. smaller `ang_vel_z`, or `ang_vel_z=(0,0)` for straight-line-only data). (2) **Temporarily edit** `velocity_env_cfg.py` `CommandsCfg` for dedicated collection runs. (3) **Several collects** with different ranges, then merge datasets before `train_forward_model.py`. (4) Align ranges with **how you actually command** the robot.

Optional Isaac Lab alternative: **`NormalVelocityCommand`** supports **per-axis `zero_prob`** to zero out individual components more often—switching command classes is a bigger change and must stay consistent with how you trained the policy.

### 4.1 One-shot pipeline (recommended)

In the **`Surprise`** repo, the script `src/retrain_forward_model_pipeline.py` runs:

1. `collect_rollout_data.py` — Isaac Sim rollouts with a frozen RSL-RL checkpoint  
2. `train_forward_model.py` — PyTorch training  
3. `export_forward_model_to_onnx.py` — writes `forward_model.onnx` and `normalization_stats.yaml`

**Always launch stage 1 through Isaac Lab’s Python** (same as when you only ran collection), e.g.:

```bash
cd /path/to/IsaacLab

./isaaclab.sh -p /path/to/Surprise/src/retrain_forward_model_pipeline.py \
  --policy /path/to/unitree_rl_lab/logs/rsl_rl/unitree_go2_velocity/<run>/model_1500.pt \
  --task Unitree-Go2-Velocity \
  --num_envs 128 \
  --num_steps 20000 \
  --headless
```

What this does:

- **`--policy`**: RSL-RL `.pt` checkpoint used only for rolling out actions (must match the walking policy you care about).
- **`--task`**: Must be **`Unitree-Go2-Velocity`** (or the exact task you trained, e.g. `Unitree-Go2-Velocity-Flat`) so observations match deploy.
- **`--num_steps` / `--num_envs`**: Total transitions scale with `num_steps * num_envs`. More data usually helps the forward model; training time increases.
- Extra flags such as **`--headless`** are forwarded to `collect_rollout_data.py`.

Outputs:

- Default **`--work_dir`**: `Surprise/runs/surprise_forward_<timestamp>/`  
  - `rollout_data/rollout_data.npz`, `normalization_stats.npz`  
  - `forward_model/forward_model_best.pt`  
- Default **`--export_dir`**: **`Surprise/data/go2/`**  
  - `forward_model.onnx`  
  - `normalization_stats.yaml`  

To pin a directory:

```bash
./isaaclab.sh -p /path/to/Surprise/src/retrain_forward_model_pipeline.py \
  --policy /path/to/model_1500.pt \
  --task Unitree-Go2-Velocity \
  --work_dir /path/to/Surprise/runs/go2_surprise_v2 \
  --num_envs 128 \
  --num_steps 20000 \
  --export_dir /path/to/Surprise/data/go2 \
  --headless
```

**Resume / partial runs:** `--skip_collect`, `--skip_train`, `--skip_export` if you already have `rollout_data.npz` or a trained checkpoint. See the script’s `--help`.

### 4.2 Manual three-step equivalent

If you prefer not to use the pipeline:

**A. Collect** (Isaac Lab Python):

```bash
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/Surprise/src/collect_rollout_data.py \
  --task Unitree-Go2-Velocity \
  --checkpoint /path/to/model_1500.pt \
  --num_envs 128 \
  --num_steps 20000 \
  --output_dir /path/to/output/rollout_data \
  --headless
```

**B. Train** (any machine with PyTorch; GPU recommended):

```bash
cd /path/to/Surprise
python src/train_forward_model.py \
  --data_path /path/to/output/rollout_data/rollout_data.npz \
  --output_dir /path/to/output/forward_model
```

**C. Export ONNX + YAML for C++:**

```bash
cd /path/to/Surprise
python src/export_forward_model_to_onnx.py \
  --checkpoint /path/to/output/forward_model/forward_model_best.pt \
  --output_dir data/go2
```

### 4.3 After you retrain the walking policy

Whenever you replace the **velocity** checkpoint on the robot:

1. Re-run **section 4** with `--policy` set to the **new** `model_*.pt`.
2. Keep **`--task`** aligned with how you trained that policy (`Unitree-Go2-Velocity` vs `Unitree-Go2-Velocity-Flat`).
3. Copy or sync the new `forward_model.onnx` and `normalization_stats.yaml` to the paths referenced in deploy `config.yaml` (section 6).

---

## 5. Export the walking policy ONNX for the robot (`play.py`)

The deploy binary loads the policy from:

```text
<policy_dir>/exported/policy.onnx
```

That folder is produced by **`scripts/rsl_rl/play.py`**, which exports JIT and ONNX next to the loaded checkpoint.

Example:

```bash
cd /path/to/unitree_rl_lab
./unitree_rl_lab.sh -p --task Unitree-Go2-Velocity --checkpoint logs/rsl_rl/unitree_go2_velocity/<run>/model_1500.pt --headless
```

(or `python scripts/rsl_rl/play.py` with the same arguments). You should see `exported/policy.onnx` under the run directory that contains the checkpoint.

**`policy_dir` in deploy config** should point at that **run folder** (the directory that parents `exported/`), not only at the `.pt` file.

---

## 6. Deploy configuration: what to change and where

Primary file:

```text
unitree_rl_lab/deploy/robots/go2/config/config.yaml
```

**Semantics** for Surprise and FSM transitions are covered in **section 2**. Below: file pointers only.

### 6.1 Walking policy (`Velocity`)

- **`FSM.Velocity.policy_dir`**: Path to the **training run directory** that contains **`exported/policy.onnx`**.  
  - Example (relative to `deploy/robots/go2/`):  
    `policy_dir: ../../../logs/rsl_rl/unitree_go2_velocity/<run_name>/`

### 6.2 Surprise forward model

Under **`FSM.Velocity.surprise`**, set paths and numeric knobs (see **section 2.5** for full parameter meanings):

| Key | Role |
|-----|------|
| `enabled` | Turn Surprise gate on/off. |
| `model_path` | `forward_model.onnx` |
| `stats_path` | `normalization_stats.yaml` |
| `threshold` | Trip when gate value exceeds this (summed NLL scale; tune with logs). |
| `use_ema`, `ema_alpha` | Raw vs smoothed surprise vs threshold. |
| `min_steps`, `consecutive_steps` | Burn-in and debouncing. |
| `log_every_steps` | Periodic `spdlog` lines for tuning. |

If `Surprise` is not a sibling of `unitree_rl_lab`, replace `model_path` / `stats_path` with absolute paths or correct relative paths from the deploy project directory.

### 6.3 Stabilization policy (`Stabilize`)

When Surprise fires (or user transitions manually):

- **`FSM.Stabilize.policy_dir`**: Run directory with **`exported/policy.onnx`** for recovery (see **section 2.7** for `auto_velocity`).
- **`FSM.Stabilize.transitions`**: Joystick rules (e.g. back to **Velocity** with **Start**).

### 6.4 Other YAML blocks

- **`FSM.FixStand`**: Scripted stand (`kp`, `kd`, `ts`, `qs`), `auto_velocity` / delays, transitions to **Velocity** or **Passive** — not an RL policy unless used as Surprise fallback.
- **`FSM.Passive`**: Safe mode gains (`mode`, `kd`).

### 6.5 Robot `params/deploy.yaml` (inside each `policy_dir`)

Joint ordering, observation stack, and gains come from **`policy_dir/params/deploy.yaml`** generated for that training run. They must match the policy ONNX. If you change observation definitions in code, retrain and re-export **both** velocity and Surprise.

---

## 7. Build and run `go2_ctrl` on the robot

```bash
cd unitree_rl_lab/deploy/robots/go2/build
cmake .. && cmake --build .
./go2_ctrl -n <network_interface>
```

### 7.1 What to copy to the robot after training

`go2_ctrl` resolves **`policy_dir`** and Surprise paths from `deploy/robots/go2/config/config.yaml`. On the robot you need **matching directory trees** (or edit `config.yaml` to absolute paths you actually use).

**Velocity walking policy** (from `play.py` export — one RSL-RL run folder):

| Path (under the run, e.g. `logs/rsl_rl/unitree_go2_velocity_flat/<date>/`) | Required |
|-----------------------------------------------------------------------------|----------|
| `exported/policy.onnx` | Yes — network loaded at runtime |
| `params/deploy.yaml` | Yes — observation stack, gains, `step_dt`, joint map |

You can sync the whole run directory, or only **`exported/`** and **`params/`** into the **same** relative layout as on your dev machine so `policy_dir` in `config.yaml` still points correctly.

**Surprise forward model** (from `export_forward_model_to_onnx.py`):

| Path under `Surprise/data/go2/` (default) | Required |
|-------------------------------------------|----------|
| `forward_model.onnx` | Yes |
| `normalization_stats.yaml` | Yes |

**Optional — Stabilize RL policy:** same pattern as velocity: that run’s `exported/` + `params/` under `FSM.Stabilize.policy_dir`.

**Example: `scp` from your PC** (replace user, IP, and run names; paths on robot assume `/home/unitree/Repos/` mirrors your workspace):

```bash
# Surprise assets (entire go2 folder)
scp -r /home/milad/Documents/Repos/Surprise/data/go2/ \
  unitree@10.0.0.184:/home/unitree/Repos/Surprise/data/

# Velocity policy: exported ONNX + params (same parent run folder on the robot)
RUN=unitree_go2_velocity_flat/2026-04-08_14-19-59
scp -r /home/milad/Documents/Repos/unitree_rl_lab/logs/rsl_rl/${RUN}/exported \
  unitree@10.0.0.184:/home/unitree/Repos/unitree_rl_lab/logs/rsl_rl/${RUN}/
scp -r /home/milad/Documents/Repos/unitree_rl_lab/logs/rsl_rl/${RUN}/params \
  unitree@10.0.0.184:/home/unitree/Repos/unitree_rl_lab/logs/rsl_rl/${RUN}/
```

Create the parent directory on the robot first if needed (`ssh unitree@10.0.0.184 'mkdir -p ...'`). If your robot’s tree differs, either mirror the **`policy_dir`** structure from `config.yaml` or change **`policy_dir`**, **`model_path`**, and **`stats_path`** on the robot to where you actually placed the files.

**Also copy or rebuild:** the **`go2_ctrl`** binary and updated **`deploy/robots/go2/config/config.yaml`** if you changed paths or thresholds.

Ensure `config.yaml` on the robot matches the paths you deployed.

---

## 8. Quick checklist

| Step | Action |
|------|--------|
| 1 | Train `Unitree-Go2-Velocity` (or Flat) → get `model_*.pt` under `logs/rsl_rl/...`. |
| 2 | Run `play.py` with that checkpoint → get `exported/policy.onnx`. |
| 3 | Run Surprise pipeline (or manual collect → train → export) with **same** `--task` and **same** policy checkpoint → get `forward_model.onnx` + `normalization_stats.yaml`. |
| 4 | Set `policy_dir`, `surprise.model_path`, `surprise.stats_path` in `deploy/robots/go2/config/config.yaml` (see **section 2** for parameter meanings). |
| 5 | Build deploy, copy assets to the robot, tune `threshold` / `consecutive_steps` using logs (**section 2.5**). |

---

## 9. File reference

| Topic | Location |
|-------|----------|
| Go2 velocity env (terrain, rewards) | `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg.py` |
| Gym task registration | `source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/__init__.py` |
| RSL-RL training | `scripts/rsl_rl/train.py` |
| Policy ONNX export | `scripts/rsl_rl/play.py` → `exported/policy.onnx` |
| Rollout collection | `Surprise/src/collect_rollout_data.py` |
| Forward model training | `Surprise/src/train_forward_model.py` |
| ONNX export for Surprise | `Surprise/src/export_forward_model_to_onnx.py` |
| End-to-end Surprise pipeline | `Surprise/src/retrain_forward_model_pipeline.py` |
| Deploy FSM + Surprise paths | `deploy/robots/go2/config/config.yaml` |
| Runtime policy + surprise loading | `deploy/robots/go2/src/State_RLBase.cpp` |
