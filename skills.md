# Go2 workflow skills

Validated, repository-local workflows. `skills.md` is the canonical index; `SKILLS.md` is retained only as a compatibility pointer. Keep reusable scripts in `skill-scripts/`, document every one below, and do not create one-off copies elsewhere.

## Common requirements

> **Figure encoding rule:** exhaust distinct meaningful colors before using line style as a second discriminator. Use color for the primary comparison identity; add solid/dashed/dotted styles only when colors are exhausted, curves overlap, or a secondary factor must be encoded. State the encoding in the legend. When assigning distinct series colors, use this preference order where feasible: **blue, red, black, green, purple, magenta, yellow**, then additional clearly distinguishable colors. Preserve solver identity colors when an established publication palette applies.

> **Reward-breakdown layout rule:** order `Episode_Reward/*` panels by configured numerical weight from highest to lowest, using the immutable run's saved `params/bipedal_env_cfg.py` as the source of truth; do not use alphabetical tag order. Group equal-weight terms only when a panel contains at most two terms; split larger groups into separate panels. Print each weight in its panel title, record the exact panel/term/weight order in the figure summary, and give every subplot its own compact in-axes legend. Never use a figure-level legend or let a legend overlap another subplot; within a panel, colour encodes run and line style encodes term. For behavioral claims (for example bilateral thigh parallelism), distinguish direct reward terms from indirect correlates; reward curves alone do not establish causality.

- Run commands from the repository root.
- Use the checkpoint's matching `params/deploy.yaml`; native DVI replay also requires its saved `params/bipedal_env_cfg.py`.
- Store canonical outputs under `policy_videos/` or the originating run. Workspace artifacts are delivery copies only.
- Validate rendered videos by inspecting frames, dimensions, and telemetry; successful process exit is not sufficient.
- The production MuJoCo transfer path uses deployment PD, 0.8 s settling, and XML joint passive forces. Controller-side DVI passive torque is diagnostic-only.

## Script catalog

### `record_go2_bipedal_native_phases.py`

Native Isaac Lab/Newton DVI recorder. Supports zero or phased velocity commands, archived solver settings, front-view tracking, and rear-leg/contact NPZ telemetry.

```bash
DVI=/home/horde/miniforge3/envs/dvi/bin/python
RUN=logs/rsl_rl/<experiment>/<run>
$DVI skill-scripts/record_go2_bipedal_native_phases.py \
  --checkpoint "$RUN/model_<iteration>.pt" \
  --saved-env-cfg "$RUN/params/bipedal_env_cfg.py" \
  --preset newton_dvi --num-envs 1 --flat-terrain --follow-robot --front-view \
  --zero-command --steps 500 --fps 50 \
  --output policy_videos/<name>/native.mp4 --telemetry policy_videos/<name>/native.npz
```

`--contact-recovery-speed` is a playback-only sensitivity override, not a retrained setting.

### `record_go2_bipedal_sim2sim_phases.py`

MuJoCo replay recorder with the matching observation history, deployment PD, GO2HV torque-speed limit, optional RL/RR contact overrides, and telemetry.

```bash
export MUJOCO_GL=egl
DVI=/home/horde/miniforge3/envs/dvi/bin/python
$DVI skill-scripts/record_go2_bipedal_sim2sim_phases.py \
  --checkpoint "$RUN/model_<iteration>.pt" --deploy-yaml "$RUN/params/deploy.yaml" \
  --zero-command --hide-overlay \
  --output policy_videos/<name>/mujoco_xml_passive.mp4 \
  --telemetry policy_videos/<name>/mujoco_xml_passive.npz
```

- Default: production XML passive joints (`damping=0.1`, `frictionloss=0.2`).
- `--dvi-actuator-passive-forces`: diagnostic-only; disables MuJoCo joint passive forces in memory and applies the DVI Go2HV passive law once in the controller.
- `--no-joint-passive-forces`: controlled no-passive-force diagnostic only.
- Do not edit `scene_flat.xml` for sweep overrides.

### `make_go2_lab_mujoco_telemetry_comparison.py`

Composes native DVI and one MuJoCo replay into a synchronized two-column video. Solid rear-joint traces are native DVI; dashed traces are MuJoCo; RL/RR bars show binary ground contacts.

```bash
$DVI skill-scripts/make_go2_lab_mujoco_telemetry_comparison.py \
  --lab-video "$OUT/native.mp4" --mujoco-video "$OUT/mujoco.mp4" \
  --lab-telemetry "$OUT/native.npz" --mujoco-telemetry "$OUT/mujoco.npz" \
  --provenance 'model_<iteration> | zero command' --output "$OUT/native_vs_mujoco.mp4"
```

### `make_go2_lab_mujoco_passive_ablation_comparison.py`

Composes native DVI, production XML-passive MuJoCo, and actuator-side-passive MuJoCo into a synchronized three-column diagnostic video.

```bash
$DVI skill-scripts/make_go2_lab_mujoco_passive_ablation_comparison.py \
  --lab-video "$OUT/native.mp4" --xml-video "$OUT/mujoco_xml_passive.mp4" \
  --actuator-video "$OUT/mujoco_actuator_passive.mp4" \
  --lab-telemetry "$OUT/native.npz" --xml-telemetry "$OUT/mujoco_xml_passive.npz" \
  --actuator-telemetry "$OUT/mujoco_actuator_passive.npz" \
  --provenance 'model_<iteration> | margin=<m> | gap=<m> | omega=<value>' \
  --output "$OUT/native_vs_mujoco_xml_vs_actuator_passive.mp4"
```

Keep checkpoint, commands, scene, camera, PD, and settling identical across all columns. This is a passive-force placement ablation, not a production transfer setting.

### `make_go2_phased_side_by_side.py`

Composes same-FPS phased MuJoCo recordings with MJWarp on the left and DVI on the right.

```bash
python skill-scripts/make_go2_phased_side_by_side.py \
  --mjwarp /path/to/mjwarp.mp4 --dvi /path/to/dvi.mp4 --output policy_videos/<name>.mp4
```

### `plot_bipedal_rough_overall.py`

Plots overall reward for explicitly supplied completed DVI and MJWarp runs. Uses TensorBoard-style EMA and within-run variability bands; outputs PNG, PDF, and SVG.

```bash
python skill-scripts/plot_bipedal_rough_overall.py \
  --old-dvi-run /path/to/old_dvi --recent-dvi-run /path/to/recent_dvi \
  --mjwarp-run /path/to/mjwarp --output-dir /path/to/figures
```

### `plot_bipedal_rough_reward_terms.py`

Plots common `Episode_Reward/*` terms for explicitly supplied completed DVI and MJWarp runs. Uses TensorBoard-style EMA and outputs PNG, PDF, and SVG.

```bash
python skill-scripts/plot_bipedal_rough_reward_terms.py \
  --old-dvi-run /path/to/old_dvi --recent-dvi-run /path/to/recent_dvi \
  --mjwarp-run /path/to/mjwarp --output-dir /path/to/figures
```


For current-DVI-versus-MJWarp plots, use `plot_dvi_vs_mjwarp_current.py` with
`--reward-config "$RUN/params/bipedal_env_cfg.py"`. It extracts literal `RewTerm`
weights from that immutable config, orders panels by descending absolute weight,
and records the term/weight order in `comparison_summary.txt`.

## Validation checklist

- Confirm the checkpoint and all matching configuration paths.
- For composite videos, validate frame count, dimensions, camera orientation, and RL/RR contact occupancy.
- Exclude failed or incomplete runs from final reward figures.
- Generate PNG, PDF, and SVG for finalized plots. Use the repository publication palette and include steady-state FPS in legend entries.
