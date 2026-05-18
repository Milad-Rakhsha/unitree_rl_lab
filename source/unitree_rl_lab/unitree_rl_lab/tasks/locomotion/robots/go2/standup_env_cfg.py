"""Go2 bipedal **stand-up** environment (quadruped → rear-stance).

Trains a gentle, controlled stand-up motion from a flat four-legged pose
to the bipedal rear-stance. Designed to be the first phase in an FSM:

    Fixed quadruped stance → **Stand-up policy** → Bipedal walk policy

The episode starts with the robot in a calm quadruped pose and terminates
on *success* when the robot reaches the bipedal stance and holds it
calmly for a configurable number of steps. The episode also terminates
on failure (base/hip/head contact) or time-out.

Key design choices for **gentle** stand-up:

* **Heavy smoothness penalties**: ``action_rate``, ``joint_vel``,
  ``joint_acc``, ``joint_torques``, and ``energy`` are all penalized
  more aggressively than in the walk env. The policy is punished for
  jerky, fast motions.
* **No velocity tracking / gait shaping**: there are no walking rewards.
  The only positive signal is orientation alignment + base height +
  success bonus. The policy has no incentive to move fast.
* **Success termination with hold requirement**: the episode only
  terminates (with a large bonus) when the robot reaches the bipedal
  stance AND holds it calmly for ~1 second (50 steps at dt=0.005,
  decimation=4 → 50 actions × 0.02s = 1.0s). This prevents the
  policy from "throwing" itself upright.
* **Positive-only reward clipping**: same ``only_positive_rewards``
  trick as the bipedal walk env — the policy is never punished into
  a net-negative step.
* **Domain randomization**: same DR as the bipedal walk env (friction,
  mass, CoM, PD gains) for sim-to-sim/real robustness. Optionally
  rough terrain for additional robustness.
* **No pushes during stand-up**: unlike the walk env, we don't apply
  random pushes — the stand-up motion should be learned in calm
  conditions. DR handles robustness.

Observation space matches the bipedal walk env (4-frame proprio history)
so the stand-up policy and walk policy see the same observation format,
simplifying the FSM transition.

Physics presets: uses the same :class:`BipedalFlatPhysicsCfg` as the
bipedal walk env. Override with ``presets=newton_mjwarp`` for Newton.
"""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_newton.physics import (
    NewtonCfg,
    MJWarpSolverCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_tasks.utils import PresetCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

from . import bipedal_mdp
from . import standup_mdp

# ---------------------------------------------------------------------------
# Constants (matching bipedal walk env)
# ---------------------------------------------------------------------------

# Projected gravity in the target bipedal rear-stance frame
DESIRED_GRAVITY_REAR = [-1.0, 0.0, 0.0]

# World-frame base z when fully upright on two legs
BIPEDAL_TARGET_BASE_Z = 0.55

# History length for proprioceptive observations (matches walk env)
OBS_HISTORY_LENGTH = 4

# ---------------------------------------------------------------------------
# Terrain configuration (gentle DR — no curriculum, mild roughness)
# ---------------------------------------------------------------------------

STANDUP_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.5),  # only easy terrain
    use_cache=False,
    curriculum=False,  # no curriculum — consistent difficulty
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.005, 0.02),  # gentle roughness
            noise_step=0.005,
            border_width=0.25,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.2,
            slope_range=(0.0, 0.08),  # very gentle slopes
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat-plane Go2 scene with contact sensing."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.2,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = None
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ---------------------------------------------------------------------------
# Events (reset + domain randomization — NO pushes)
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """Quadruped prone spawn + domain randomization (same DR as bipedal walk).

    No random pushes — the stand-up should be learned in calm conditions.
    DR on friction, mass, CoM, and PD gains provides robustness.
    """

    # --- Startup DR (same as bipedal walk env) ---

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.3),
            "dynamic_friction_range": (0.5, 1.3),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.5),
            "operation": "add",
        },
    )

    add_rear_leg_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["R[LR]_thigh", "R[LR]_calf"]
            ),
            "mass_distribution_params": (-0.15, 0.25),
            "operation": "add",
        },
    )

    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {
                "x": (-0.03, 0.03),
                "y": (-0.03, 0.03),
                "z": (-0.02, 0.02),
            },
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.85, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # --- Reset: flat quadruped prone start ---

    reset_prone = EventTerm(
        func=standup_mdp.reset_quadruped_prone,
        mode="reset",
        params={
            "xy_range": 0.3,
            "roll_range": (-0.4, 0.4),
            "pitch_range": (-0.3, 0.3),
            "joint_pos_noise": 0.3,
            "joint_vel_noise": 0.5,
            "base_z_range": (0.20, 0.38),
            "lin_vel_range": 0.3,
            "ang_vel_range": 0.5,
        },
    )

    # NOTE: No push_robot event. Stand-up is learned in calm conditions.


# ---------------------------------------------------------------------------
# Commands (zero — no velocity tracking, pure stand-up)
# ---------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Zero velocity command — the policy only stands up, no locomotion."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1000.0, 1000.0),
        rel_standing_envs=1.0,
        debug_vis=False,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


# ---------------------------------------------------------------------------
# Observations (matches bipedal walk env — 4-frame proprio history)
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Same observation structure as bipedal walk env.

    The policy sees 4-step history of proprio (ang vel, gravity, joint
    pos/vel) plus last action. No velocity command obs since commands are
    zero, but we include a placeholder zero-command obs for structural
    compatibility with the walk policy observation space.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=OBS_HISTORY_LENGTH,
            flatten_history_dim=True,
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            clip=(-100, 100),
            params={"command_name": "base_velocity"},
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Rewards (gentle stand-up shaping)
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Reward shaping for a slow, controlled stand-up (quadruped prone → bipedal rear-stance).

    Overview
    --------
    The reward structure has three layers:

    1. **Goal terms** (positive) — pull the robot toward the upright target pose:
       - ``orientation_align`` (w=3.0)  — monotonic alignment toward rear-stance gravity
       - ``success_bonus``     (w=8.0)  — large per-step reward when holding the target pose
       - ``is_alive``          (w=0.1)  — small survival bonus
       - ``rear_foot_contact`` (w=2.0)  — encourage feet-on-ground final pose

    2. **Smoothness / regularization terms** (negative) — encourage gentleness:
       - ``action_rate``       (w=-0.05)   — penalize jerky action changes
       - ``joint_vel``         (w=-0.001)  — penalize fast joint speeds
       - ``joint_acc``         (w=-1e-7)   — penalize joint-level jerk
       - ``joint_torques``     (w=-1e-4)   — penalize high torques
       - ``energy``            (w=-1e-5)   — penalize power consumption
       - ``velocity_near_upright`` (w=-1.5) — ramp up velocity penalty near upright

    3. **Pose / stability terms** (negative) — keep the stand-up controlled:
       - ``base_height``       (w=-0.5)  — pull toward target height (0.55 m)
       - ``orientation_xy``    (w=-0.05) — penalize lateral roll
       - ``lin_vel_z``         (w=-1.0)  — suppress vertical bouncing
       - ``lin_vel_xy``        (w=-0.5)  — suppress horizontal drift
       - ``ang_vel_xy``        (w=-0.1)  — suppress roll/pitch rates
       - ``ang_vel_z``         (w=-0.3)  — suppress yaw spin
       - ``dof_pos_limits``    (w=-10.0) — hard wall near joint limits
       - ``rear_leg_symmetry`` (w=-0.5)  — force L/R rear-leg symmetry
       - ``front_{hip,thigh,calf}_motion`` (w=-0.25/-0.15/-0.15) — keep front legs tucked
       - ``front_leg_joint_vel`` (w=-0.002) — keep front legs still

    4. **Contact penalties** (negative) — discourage unwanted body contact:
       - ``base_contact``      (w=-1.5)  — trunk should leave the ground
       - ``hip_contact``       (w=-1.0)  — hips off the ground
       - ``thigh_contact``     (w=-1.0)  — thighs off the ground
       - ``front_calf_contact``(w=-0.5)  — front calves off the ground
       - ``front_foot_contact``(w=-1.0)  — front feet off the ground
       - ``head_contact``      (w=-1.0)  — head off the ground
       - ``rear_calf_contact`` (w=-2.0)  — once upright, shift from calves to feet

    Design philosophy
    -----------------
    - No velocity-gate on ``orientation_align``: the robot gets full credit
      for being upright at any speed. Smoothness is enforced entirely by the
      penalty terms, which avoids the failure mode where a velocity-gated
      primary reward creates a local optimum of "barely tilted but calm".
    - ``velocity_near_upright`` is the key adaptive mechanism: it costs nothing
      when the robot is flat (free to push off), but ramps to full strength
      near upright, forcing a slow, controlled final approach and stable hold.
    - ``success_bonus`` is the highest-weighted term (8.0), making the terminal
      pose the single most valuable thing per step. Combined with the success
      termination, this creates strong gradient toward reaching AND holding.
    - ``rear_foot_contact`` + ``rear_calf_contact`` together create a smooth
      gradient to shift support from calves (used during the transition) to
      feet (needed for handoff to the bipedal walk policy).
    - Contact penalties replace contact terminations because the robot starts
      on the ground (contacts unavoidable at episode start).
    - ``only_positive_rewards`` clips the per-step total to >= 0, so penalty
      terms shape the gradient without creating net-negative reward steps.
    """

    # ── 1. orientation_align (w=3.0) ──────────────────────────────────
    # WHAT:  Computes max(0, dot(projected_gravity_body, desired_gravity)),
    #        where desired_gravity = [-1, 0, 0] (the gravity direction in
    #        the target bipedal rear-stance body frame). Returns a scalar
    #        in [0, 1] that increases monotonically as the robot tilts
    #        from prone (dot ≈ 0) toward the upright rear-stance (dot ≈ 1).
    # WHY:   This is the PRIMARY positive shaping signal that pulls the
    #        robot toward the target pose. Without it, the policy has no
    #        gradient to discover that standing up is the goal.
    #        Uses the "raw" (un-gated) variant: the robot gets full credit
    #        for any upright progress regardless of velocity. An earlier
    #        velocity-gated variant was removed because it created a local
    #        optimum where the policy would barely tilt upright and freeze
    #        ("calm but horizontal"). Smoothness is enforced entirely by
    #        the penalty terms below, not by gating the primary reward.
    # WEIGHT: 3.0 — the second-highest positive weight (after success_bonus).
    #        At full alignment (reward=1.0), this contributes 3.0/step,
    #        which dominates the reward landscape during the transition
    #        phase before the success_bonus activates.
    # INTERACTIONS: Works in tandem with velocity_near_upright (w=-1.5),
    #        which penalizes speed near upright. Together they create
    #        "stand up, but slow down as you get there".
    # v4: time-scaled orientation — starts at 20% for the first 75 steps
    # (~1.5s), ramps to 100%. Removes incentive to snap upright instantly.
    orientation_align = RewTerm(
        func=standup_mdp.orientation_align_time_scaled,
        weight=3.0,
        params={
            "desired_gravity": DESIRED_GRAVITY_REAR,
            "ramp_steps": 75,
        },
    )

    # ── 2. success_bonus (w=8.0) ────────────────────────────────────
    # WHAT:  Binary per-step reward (0 or 1) that fires every step the
    #        robot simultaneously satisfies three conditions:
    #          - orientation: cos(angle_to_target) >= 0.90 (~25° from goal)
    #          - height:      base_z >= 0.45 m (near the 0.55 m target)
    #          - calm:        ||ang_vel_world|| < 1.0 rad/s
    # WHY:   Provides strong immediate gradient toward reaching AND
    #        HOLDING the target pose. Unlike the standing_success
    #        termination (which requires 50 consecutive steps), this gives
    #        reward from the very first step the robot enters the target
    #        zone. This is critical for learning: the policy discovers
    #        early that being upright + calm = high reward, long before
    #        it can hold the pose for a full second.
    # WEIGHT: 8.0 — the HIGHEST weight in the entire reward function.
    #        When active, it contributes 8.0/step, dwarfing all other
    #        terms. This makes the terminal pose overwhelmingly valuable
    #        and ensures the policy prioritizes reaching it over any
    #        local optima in the shaping terms. The high weight is
    #        balanced by the fact that the bonus only activates in a
    #        small region of state space (near-upright, calm).
    # INTERACTIONS: The calm condition (max_ang_vel=1.0) aligns with
    #        velocity_near_upright, which also penalizes speed near
    #        upright. Together they make the final approach slow and
    #        stable. The thresholds match standing_success exactly,
    #        so the policy is rewarded continuously for the same
    #        conditions that will eventually trigger episode success.
    success_bonus = RewTerm(
        func=standup_mdp.success_bonus,
        weight=8.0,
        params={
            "desired_gravity": DESIRED_GRAVITY_REAR,
            "min_cos_angle": 0.90,
            "min_base_height": 0.45,
            "max_ang_vel": 1.0,
        },
    )

    # ── 3. velocity_near_upright (w=-1.5) ───────────────────────────
    # WHAT:  Orientation-ramped velocity penalty. Computes:
    #          scale = clamp((cos_angle - 0.3) / (0.85 - 0.3), 0, 1)
    #          penalty = scale * (||lin_vel_w||² + 0.5 * ||ang_vel_w||²)
    #        When the robot is flat (cos < 0.3): scale = 0, no penalty.
    #        As it tilts upright: penalty ramps linearly.
    #        Near upright (cos > 0.85): full penalty on all velocity.
    # WHY:   THE KEY ADAPTIVE MECHANISM for gentle stand-up. The robot
    #        needs to move (push off the ground, tilt upward) during the
    #        early phase, but must slow down and stabilize as it
    #        approaches vertical. A flat velocity penalty would fight the
    #        initial push-off; this ramped version gives the robot
    #        freedom to move when flat but demands stillness near upright.
    # WEIGHT: -1.5 — heavy for a penalty. At full scale (upright) with
    #        moderate velocity (e.g., 1 rad/s angular + 0.5 m/s linear),
    #        the penalty is ~-1.5 * (0.25 + 0.5) = ~-1.1 per step. This
    #        is comparable to the orientation_align reward (3.0), so the
    #        policy can only profit from being upright if it's also calm.
    # INTERACTIONS: Directly complements orientation_align: orientation
    #        pulls the robot upright, velocity_near_upright forces it to
    #        arrive slowly. Also aligns with success_bonus thresholds —
    #        the bonus requires ang_vel < 1.0, and this penalty is
    #        already at full strength by cos=0.85 (where the bonus
    #        activates at cos=0.90), so the policy is already braking
    #        before it enters the success zone.
    velocity_near_upright = RewTerm(
        func=standup_mdp.velocity_near_upright_penalty,
        weight=-1.5,
        params={
            "desired_gravity": DESIRED_GRAVITY_REAR,
            "onset_cos": 0.3,
            "full_cos": 0.85,
        },
    )

    # ── 4. base_height (w=-0.5) ────────────────────────────────────
    # WHAT:  Quadratic penalty: (base_z - 0.55)², where 0.55 m is the
    #        world-frame base height in the target bipedal rear-stance.
    #        Penalizes deviation from the target height in both directions.
    # WHY:   Provides a continuous gradient pulling the base upward from
    #        the prone starting height (~0.20–0.38 m) toward the bipedal
    #        target. Unlike a Gaussian kernel (which plateaus far from
    #        target), the L2 quadratic has nonzero gradient everywhere,
    #        so it provides useful signal even from the initial flat pose.
    # WEIGHT: -0.5 — deliberately moderate. At the starting height
    #        (~0.34 m), the penalty is -0.5 * (0.21)² = -0.022/step,
    #        which is small enough not to overwhelm the orientation signal
    #        early in training. If too heavy, the policy tries to raise
    #        the CoM without properly rotating (e.g., extending legs
    #        while still flat), creating unstable intermediate poses.
    # INTERACTIONS: Complements orientation_align — together they define
    #        the target pose: correct body angle AND correct height.
    #        The height term prevents a pathological solution where the
    #        robot achieves good orientation but at the wrong height
    #        (e.g., crouched on rear legs with body tilted up).
    base_height = RewTerm(
        func=bipedal_mdp.base_height_l2,
        weight=-0.5,
        params={"target_height": BIPEDAL_TARGET_BASE_Z},
    )

    # ── 5. orientation_xy (w=-0.05) ─────────────────────────────────
    # WHAT:  Penalizes proj_grav_b.x² + proj_grav_b.y² (the squared
    #        magnitude of the body-frame gravity vector’s XY components).
    #        This is 0 when the body is perfectly level, and 1 when
    #        fully pitched or rolled 90°.
    # WHY:   Acts as a SMALL lateral roll damper. In the bipedal
    #        rear-stance, the x-component of projected gravity is
    #        intentionally large (~-1.0 because the body is pitched 90°),
    #        so this term weakly penalizes the target pitch. Its primary
    #        role is to penalize the y-component (lateral roll), keeping
    #        the robot from tilting sideways during stand-up. It also
    #        adds slight resistance against over-rotation past vertical.
    # WEIGHT: -0.05 — intentionally tiny. This term conflicts with the
    #        pitch-up we want (orientation_align), so it must be small
    #        enough to be a gentle nudge, not a barrier. At full pitch
    #        (standing upright), it costs only -0.05/step vs the +3.0
    #        from orientation_align, so the net gradient still strongly
    #        favors standing up.
    # INTERACTIONS: Mildly opposes orientation_align on the pitch axis.
    #        The net effect is that the robot prefers to reach the target
    #        orientation along a straight-line pitch trajectory rather
    #        than wobbling through roll.
    orientation_xy = RewTerm(func=bipedal_mdp.orientation_xy_sq, weight=-0.05)

    # ── 6. is_alive (w=0.1) ──────────────────────────────────────
    # WHAT:  Returns 1.0 every step the robot has not been terminated.
    #        This is a constant +0.1/step survival reward.
    # WHY:   Provides a small baseline incentive for the policy to avoid
    #        termination conditions (standing_success counts as time-out,
    #        so the main termination risk is the episode time limit).
    #        More importantly, it ensures the policy always collects
    #        *some* positive reward per step, which helps with
    #        only_positive_rewards clipping and prevents the total reward
    #        from being exactly zero early in training when the policy
    #        hasn’t learned to tilt yet.
    # WEIGHT: 0.1 — very small. Over a 10s episode (500 steps), this
    #        contributes only 50.0 total, compared to 1500+ from
    #        orientation_align and 4000+ from success_bonus. It’s
    #        intentionally a tiebreaker, not a driver.
    is_alive = RewTerm(func=mdp.is_alive, weight=0.1)

    # =====================================================================
    # SMOOTHNESS PENALTIES (terms 7–11)
    # These five terms together define the "gentle" character of the
    # stand-up. They penalize different aspects of motion aggressiveness.
    # The weights are moderate (not aggressive) — the robot needs to
    # actually move to stand up, so over-penalizing motion creates a
    # policy that barely tilts upright and wobbles in place.
    # =====================================================================

    # ── 7. action_rate (w=-0.05) ─────────────────────────────────
    # WHAT:  L2 norm of (action_t - action_{t-1}). Penalizes the change
    #        in joint position targets between consecutive timesteps.
    # WHY:   Suppresses jerky, high-frequency oscillations in the action
    #        signal. A smooth action sequence means smooth joint
    #        trajectories, which translates to a smooth stand-up motion.
    # WEIGHT: -0.05 — heavier than a typical walk env (-0.005–0.01)
    #        because the stand-up doesn’t need rapid action changes
    #        (no gait frequency to track). Tuned to allow gradual
    #        posture changes while penalizing sudden jerks.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)

    # ── 8. joint_vel (w=-0.001) ─────────────────────────────────
    # WHAT:  Sum of squared joint velocities across all 12 joints.
    #        Penalizes fast joint motion regardless of direction.
    # WHY:   Encourages the policy to move joints slowly. Works with
    #        action_rate: action_rate penalizes changes in *commands*,
    #        joint_vel penalizes the *actual speed* of joints. Together
    #        they discourage both commanded and physical jerkiness.
    # WEIGHT: -0.001 — small per-joint, but summed over 12 joints with
    #        typical velocities of 1–5 rad/s during stand-up, this
    #        contributes roughly -0.01 to -0.3 per step.
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)

    # ── 9. joint_acc (w=-1e-7) ─────────────────────────────────
    # WHAT:  Sum of squared joint accelerations (finite difference of
    #        joint velocities). Penalizes joint-level jerk — rapid
    #        changes in joint velocity.
    # WHY:   The highest-order smoothness term. Even if joint velocities
    #        are moderate, sudden *changes* in velocity (accelerations)
    #        create impact forces and mechanical stress. This term
    #        smooths the velocity profile of each joint over time.
    # WEIGHT: -1e-7 — tiny because joint accelerations are large in
    #        magnitude (squared velocities-per-dt). The small weight
    #        ensures this is a gentle regularizer, not a dominant term.
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)

    # ── 10. joint_torques (w=-1e-4) ─────────────────────────────
    # WHAT:  Sum of squared applied torques across all joints.
    #        Penalizes high-torque commands from the PD controller.
    # WHY:   Discourages brute-force actuation. The policy should use
    #        gravity and momentum rather than fighting them with raw
    #        torque. Also improves sim-to-real transfer — real motors
    #        have torque limits and overheat under sustained high torque.
    # WEIGHT: -1e-4 — small, acting as a tiebreaker between trajectories
    #        that achieve the same orientation but with different torque
    #        profiles. The policy prefers the lower-effort path.
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)

    # ── 11. energy (w=-1e-5) ───────────────────────────────────
    # WHAT:  Sum of |joint_vel| * |applied_torque| across all joints.
    #        This is mechanical power consumption (P = τ·ω), the
    #        physical energy expenditure rate.
    # WHY:   Penalizes inefficient motions where the robot is both
    #        moving fast AND applying high torque simultaneously.
    #        A joint moving under gravity (low torque) or a joint
    #        holding position (low velocity) incurs little penalty.
    #        Only wasteful "fighting" motions are expensive.
    # WEIGHT: -1e-5 — the smallest smoothness weight. Energy is more
    #        of a tiebreaker / efficiency nudge than a strong penalty.
    #        It shapes the policy toward energy-efficient trajectories
    #        without interfering with the stand-up motion.
    energy = RewTerm(func=mdp.energy, weight=-1.0e-5)

    # ── 12. dof_pos_limits (w=-10.0) ──────────────────────────────
    # WHAT:  Penalizes joints that are near or exceeding their soft
    #        position limits. The penalty grows as joints approach
    #        the limit boundary, acting as a soft wall.
    # WHY:   Protects the real hardware from being commanded into
    #        mechanical stop positions, which can damage gears and
    #        linkages. Also prevents the policy from relying on joint
    #        limit contact forces as part of the stand-up strategy
    #        (e.g., slamming a leg to full extension and using the
    #        hard stop as a lever).
    # WEIGHT: -10.0 — the second-highest-magnitude penalty (after
    #        rear_calf_contact at -2.0 scaled by orientation).
    #        Intentionally harsh: joint limits are a hard safety
    #        constraint, not a soft preference. The high weight ensures
    #        the policy treats limits as nearly impassable barriers.
    #        This is critical for the stand-up motion, which involves
    #        large joint excursions (especially rear thigh and calf).
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # =====================================================================
    # STABILITY PENALTIES (terms 13–16)
    # World-frame velocity penalties that keep the stand-up controlled.
    # All use world-frame quantities (not body-frame) because at 90°
    # pitch, body axes are rotated relative to world — body-frame Z
    # becomes horizontal, which would penalize the wrong directions.
    # =====================================================================

    # ── 13. lin_vel_z (w=-1.0) ─────────────────────────────────
    # WHAT:  Squared world-Z linear velocity: (v_z_world)².
    #        Penalizes vertical bouncing or upward/downward launching.
    # WHY:   Prevents the policy from learning a "hop-to-vertical"
    #        strategy where the robot throws itself upward and tries
    #        to land on its rear legs. The stand-up should be a smooth
    #        rotation, not a ballistic trajectory. Uses the custom
    #        world-frame version (not stock body-frame) because at
    #        bipedal pitch, body-Z is horizontal.
    # WEIGHT: -1.0 — the heaviest stability penalty. Even moderate
    #        vertical velocity (0.5 m/s) costs -0.25/step, comparable
    #        to orientation_align reward early in the motion. This
    #        strongly discourages any bouncing behavior.
    lin_vel_z = RewTerm(func=bipedal_mdp.lin_vel_z_world_l2, weight=-1.0)

    # ── 14. ang_vel_xy (w=-0.1) ────────────────────────────────
    # WHAT:  Squared world-XY angular velocity: ω_x² + ω_y² (world frame).
    #        Penalizes roll and pitch rotation rates in the world frame.
    # WHY:   Damps the tipping motion itself. During stand-up, the robot
    #        must pitch ~90°, but this penalty ensures it does so at a
    #        controlled rate rather than whipping over. Uses world-frame
    #        (not body-frame) because at 90° pitch, body-X aligns with
    #        world-Z, so body-frame ang_vel_xy would penalize yaw.
    # WEIGHT: -0.1 — moderate. Light enough to allow the necessary
    #        pitch rotation (the robot MUST pitch ~90°), but heavy
    #        enough to penalize fast rotation. At a comfortable 1 rad/s
    #        pitch rate, this costs only -0.1/step.
    # INTERACTIONS: Complements velocity_near_upright, which also
    #        penalizes angular velocity but only near upright. This
    #        term provides a constant (orientation-independent) damping
    #        floor throughout the entire motion.
    ang_vel_xy = RewTerm(func=bipedal_mdp.ang_vel_xy_world_l2, weight=-0.1)

    # ── 15. lin_vel_xy (w=-0.5) ────────────────────────────────
    # WHAT:  Squared world-XY linear velocity: v_x² + v_y² (world frame).
    #        Penalizes horizontal drift (forward/backward/sideways).
    # WHY:   The robot should stand up IN PLACE, not slide around.
    #        This is a standup-specific term (not in the bipedal walk
    #        env, which commands XY velocity). Without it, the policy
    #        might learn to "walk" itself upright by sliding backward.
    # WEIGHT: -0.5 — moderate-to-heavy. At 0.3 m/s horizontal drift,
    #        this costs -0.045/step. Heavy enough to discourage sliding
    #        but not so heavy that it prevents the small lateral
    #        adjustments needed for balance.
    lin_vel_xy = RewTerm(func=standup_mdp.lin_vel_xy_world_l2, weight=-0.5)

    # ── 16. ang_vel_z (w=-0.3) ─────────────────────────────────
    # WHAT:  Squared world-Z angular velocity: ω_z² (yaw rate in world
    #        frame). Penalizes spinning around the vertical axis.
    # WHY:   The stand-up should not involve any yaw rotation. This is
    #        a standup-specific term (the walk env tracks yaw commands).
    #        Without it, the policy might learn asymmetric strategies
    #        that involve spinning during the transition.
    # WEIGHT: -0.3 — heavier than ang_vel_xy (-0.1) because there is
    #        NO reason for yaw motion during stand-up (unlike pitch,
    #        which is the desired rotation axis). A 1 rad/s yaw rate
    #        costs -0.3/step.
    ang_vel_z = RewTerm(func=standup_mdp.ang_vel_z_world_l2, weight=-0.3)

    # ── 17. rear_leg_symmetry (w=-0.5) ────────────────────────────
    # WHAT:  For each of 3 mirrored joint pairs (RR_hip↔RL_hip,
    #        RR_thigh↔RL_thigh, RR_calf↔RL_calf), computes the
    #        squared position difference and averages over pairs.
    #        Penalizes left/right rear-leg asymmetry.
    # WHY:   The bipedal rear-stance is symmetric — both rear legs
    #        should support equal weight. Without this term, the policy
    #        might discover asymmetric strategies (e.g., one leg pushes
    #        harder, creating a "dancing" or listing motion). Symmetry
    #        also improves robustness: a symmetric stance is inherently
    #        more stable than a lopsided one.
    # WEIGHT: -0.5 — moderate. With joints typically within 0.1–0.3 rad
    #        of each other during a good stand-up, this costs
    #        -0.005 to -0.045/step. Heavy enough to enforce rough
    #        symmetry but light enough to allow small left/right
    #        differences needed for balance on uneven terrain.
    # INTERACTIONS: Only applies to REAR legs (the support legs).
    #        Front legs are handled separately by front_{hip,thigh,calf}_motion.
    rear_leg_symmetry = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["RR_hip_joint", "RL_hip_joint"],
                ["RR_thigh_joint", "RL_thigh_joint"],
                ["RR_calf_joint", "RL_calf_joint"],
            ],
        },
    )

    # =====================================================================
    # FRONT LEG REGULARIZATION (terms 18–21)
    # In bipedal rear-stance, the front legs serve no purpose. They
    # should stay tucked near their default (quadruped) positions and
    # not flail or move during stand-up. These four terms enforce that:
    # three position-deviation penalties (one per joint type) plus a
    # velocity penalty. The split by joint type allows tuning hip
    # stiffness independently (hips are the most visible when flailing).
    # =====================================================================

    # ── 18. front_hip_motion (w=-0.25) ────────────────────────────
    # WHAT:  L1 deviation of FL_hip and FR_hip from their default
    #        (quadruped) joint positions: Σ|q - q_default|.
    # WHY:   Front hip abduction/adduction is the most visible flailing
    #        degree of freedom. If the front legs splay out during
    #        stand-up, it looks uncontrolled and wastes energy.
    # WEIGHT: -0.25 — the heaviest of the three front-leg position
    #        penalties. Hips get extra weight because hip flailing is
    #        more disruptive than thigh/calf motion (longer moment arm).
    front_hip_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.25,
        params={"joint_patterns": ["F[LR]_hip_joint"]},
    )

    # ── 19. front_thigh_motion (w=-0.15) ──────────────────────────
    # WHAT:  L1 deviation of FL_thigh and FR_thigh from default positions.
    # WHY:   Keeps the front thighs (upper leg) from swinging during
    #        the stand-up transition. Less critical than hips because
    #        thigh motion is less visible and has less moment arm.
    # WEIGHT: -0.15 — lighter than hips (-0.25) but equal to calves.
    front_thigh_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.15,
        params={"joint_patterns": ["F[LR]_thigh_joint"]},
    )

    # ── 20. front_calf_motion (w=-0.15) ───────────────────────────
    # WHAT:  L1 deviation of FL_calf and FR_calf from default positions.
    # WHY:   Keeps the front lower legs from dangling or extending.
    #        The default calf position (-1.5 rad) has the lower legs
    #        tucked under the body, which is the desired pose.
    # WEIGHT: -0.15 — equal to thighs. The front calves are the least
    #        critical (least visible, shortest moment arm), but still
    #        need regularization to prevent them from flapping freely.
    front_calf_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.15,
        params={"joint_patterns": ["F[LR]_calf_joint"]},
    )

    # ── 21. front_leg_joint_vel (w=-0.002) ────────────────────────
    # WHAT:  Sum of squared joint velocities for all front-leg joints
    #        only (F[LR]_hip, F[LR]_thigh, F[LR]_calf — 6 joints).
    #        Uses the same joint_vel_l2 function as term #8 but
    #        restricted to front legs via asset_cfg joint_names filter.
    # WHY:   Complements the position-deviation terms above. Those
    #        penalize being far from default; this penalizes MOVING
    #        at all. Together, the front legs should be still and
    #        tucked. Without the velocity penalty, the policy might
    #        oscillate the front legs around the default position
    #        (satisfying position but not velocity).
    # WEIGHT: -0.002 — twice the global joint_vel weight (-0.001).
    #        Front legs get extra velocity penalty because they have
    #        zero functional role and should be completely still.
    front_leg_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["F[LR]_.*"]),
        },
    )

    # =====================================================================
    # CONTACT PENALTIES (terms 22–27)
    # These replace contact-based TERMINATIONS, which were removed because
    # the robot starts lying on the ground (contacts unavoidable at episode
    # start — see TerminationsCfg docstring). Instead, undesired contacts
    # are penalized via rewards, providing gradient signal to lift off the
    # ground without killing episodes. The threshold of 1.0 N filters
    # out sensor noise; any force above 1 N is counted as contact.
    # =====================================================================

    # ── 22. base_contact (w=-1.5) ───────────────────────────────
    # WHAT:  Binary penalty (0 or 1) when the base (trunk) body has
    #        contact forces exceeding 1.0 N threshold.
    # WHY:   The trunk should leave the ground as early as possible.
    #        At episode start (prone), this fires every step, creating
    #        immediate incentive to lift the body. The penalty decreases
    #        naturally as the robot transitions to rear-leg support.
    # WEIGHT: -1.5 — the heaviest contact penalty. The trunk is the
    #        largest body, and trunk-on-ground is the furthest state
    #        from the goal. The high weight ensures the policy
    #        prioritizes getting the trunk off the ground early.
    base_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
        },
    )

    # ── 23. hip_contact (w=-1.0) ────────────────────────────────
    # WHAT:  Binary penalty when ANY hip body (FL_hip, FR_hip, RL_hip,
    #        RR_hip) has contact forces > 1.0 N.
    # WHY:   Hip contact means the robot is either still lying on its
    #        side or has fallen. Hips should never touch the ground in
    #        the target bipedal stance (they are mid-body).
    # WEIGHT: -1.0 — standard contact penalty level. Lighter than base
    #        (-1.5) because hip contact is less common and less
    #        indicative of a fully-prone state.
    hip_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip"),
        },
    )

    # ── 24. thigh_contact (w=-1.0) ──────────────────────────────
    # WHAT:  Binary penalty when ANY thigh body (all 4 legs) has
    #        contact forces > 1.0 N.
    # WHY:   Thigh contact indicates the robot is partially collapsed
    #        or lying on its upper legs. In the target rear-stance,
    #        no thighs should touch the ground. Penalizing all 4 thighs
    #        (not just front) is intentional: even rear thighs should
    #        be elevated in the upright pose.
    # WEIGHT: -1.0 — same as hip_contact.
    thigh_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
        },
    )

    # ── 25. front_calf_contact (w=-0.5) ───────────────────────────
    # WHAT:  Binary penalty when FRONT calves (F[LR]_calf only) have
    #        contact forces > 1.0 N.
    # WHY:   Front calves have no role in the bipedal stance and should
    #        be tucked away (along with the rest of the front legs).
    #        Note: REAR calves are handled separately by
    #        rear_calf_contact (term #29) with an orientation-ramped
    #        penalty, because rear calves are valid support structures
    #        during the transition but should give way to rear feet
    #        in the final pose.
    # WEIGHT: -0.5 — lighter than other contact penalties because front
    #        calf contact is less dangerous (the robot isn’t supporting
    #        weight on them) and less common once the front legs are
    #        tucked.
    front_calf_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F[LR]_calf"),
        },
    )

    # ── 26. front_foot_contact (w=-1.0) ───────────────────────────
    # WHAT:  Binary penalty when FRONT feet (F[LR]_foot) have contact
    #        forces > 1.0 N.
    # WHY:   In the bipedal rear-stance, the front feet should be
    #        completely off the ground. Front foot contact means the
    #        robot is either still in a quadruped pose (not yet tilted
    #        up) or has fallen forward. Penalizing this encourages
    #        the robot to commit to the rear-stance rather than
    #        maintaining front foot contact as a safety blanket.
    # WEIGHT: -1.0 — standard contact penalty. Heavier than front_calf
    #        (-0.5) because front foot contact is more indicative of
    #        a fundamentally wrong pose (quadruped instead of bipedal).
    front_foot_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F[LR]_foot"),
        },
    )

    # ── 27. head_contact (w=-1.0) ───────────────────────────────
    # WHAT:  Binary penalty when any Head_* body has contact forces
    #        > 1.0 N. Covers all head-related collision bodies.
    # WHY:   Head contact means the robot has tipped over forward past
    #        vertical (nose hit the ground), which is the worst failure
    #        mode of the stand-up. In the target pose, the head (nose)
    #        points upward. Head contact is also dangerous on real
    #        hardware (LiDAR, cameras on the head).
    # WEIGHT: -1.0 — standard contact penalty. Could be higher to
    #        strongly discourage over-rotation, but the orientation
    #        reward already provides strong gradient away from
    #        over-rotated states.
    head_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*"]),
        },
    )

    # =====================================================================
    # REAR SUPPORT TRANSITION (terms 28–29)
    # These two terms work as a pair to smoothly transition the rear
    # support point from calves (used during the stand-up transition)
    # to feet (needed for handoff to the bipedal walk policy). Both
    # use the same soft orientation ramp (onset_cos=0.3, full_cos=0.85)
    # so the transition gradient is smooth and consistent.
    # =====================================================================

    # ── 28. rear_foot_contact (w=+2.0) ───────────────────────────
    # WHAT:  POSITIVE reward for rear foot (R[LR]_foot) ground contact,
    #        scaled by a soft orientation ramp:
    #          scale = clamp((cos_angle - 0.3) / (0.85 - 0.3), 0, 1)
    #          reward = scale * (fraction of rear feet in contact)
    #        Returns in [0, 1]. When flat: scale ≈ 0, no reward.
    #        When upright with both feet planted: reward ≈ 1.0.
    # WHY:   The bipedal walk policy expects the handoff pose to be
    #        standing on rear FEET, not rear calves. Without this term,
    #        the stand-up policy may learn to balance on calves
    #        (a valid but wrong terminal pose that the walk policy has
    #        never seen). The orientation ramp prevents rewarding random
    #        foot-ground contact during early stand-up (when the robot
    #        is still flat on the ground).
    # WEIGHT: +2.0 — the third-highest positive weight (after
    #        success_bonus at 8.0 and orientation_align at 3.0).
    #        At full activation, each step with both feet planted yields
    #        +2.0, which is a strong incentive competing with the
    #        rear_calf_contact penalty below.
    # INTERACTIONS: Forms a push-pull pair with rear_calf_contact
    #        (w=-2.0). Together: "get your feet on the ground (+2.0)
    #        and get your calves off the ground (-2.0)." Both ramp
    #        identically with orientation, so the combined signal is
    #        zero when flat and full when upright. Also shares the
    #        onset_cos/full_cos ramp with velocity_near_upright.
    rear_foot_contact = RewTerm(
        func=standup_mdp.rear_foot_contact_reward,
        weight=2.0,
        params={
            "desired_gravity": DESIRED_GRAVITY_REAR,
            "onset_cos": 0.3,   # start rewarding early in the tilt-up
            "full_cos": 0.85,   # full reward near upright
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[LR]_foot"),
        },
    )

    # ── 29. rear_calf_contact (w=-2.0) ───────────────────────────
    # WHAT:  Orientation-ramped penalty for rear calf (R[LR]_calf)
    #        ground contact:
    #          scale = clamp((cos_angle - 0.3) / (0.85 - 0.3), 0, 1)
    #          penalty = scale * (any calf in contact ? 1 : 0)
    #        Returns in [0, 1]. Weight is negative, so this is a cost.
    # WHY:   During early stand-up, the rear calves naturally support
    #        the robot’s weight — this is physically correct and should
    #        not be penalized (the ramp is ~0 when flat). As the robot
    #        approaches upright, calf contact becomes undesirable: the
    #        final pose should have weight on rear FEET, not calves.
    #        The smooth ramp gives the policy a continuous gradient to
    #        shift weight from calves to feet, rather than a hard
    #        threshold that might scare the policy away from standing up.
    # WEIGHT: -2.0 — matches rear_foot_contact (+2.0) in magnitude.
    #        The equal-and-opposite weights mean the policy has roughly
    #        zero net incentive for calf contact when flat (both ramps
    #        ≈ 0), but a strong +2/-2 push-pull when upright: plant
    #        feet, lift calves.
    # INTERACTIONS: Push-pull pair with rear_foot_contact. Also
    #        interacts with the global smoothness penalties: the
    #        transition from calf to foot support requires joint motion,
    #        which smoothness penalties resist. The -2.0 weight is
    #        strong enough to overcome the smoothness resistance.
    rear_calf_contact = RewTerm(
        func=standup_mdp.rear_calf_contact_penalty,
        weight=-2.0,
        params={
            "desired_gravity": DESIRED_GRAVITY_REAR,
            "onset_cos": 0.3,
            "full_cos": 0.85,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[LR]_calf"),
        },
    )

    # v4: Penalize reaching upright before ~2s. This is the ONLY
    # additional penalty vs the original config. Everything else
    # (smoothness, contacts, height) stays at original weights.
    early_standup = RewTerm(
        func=standup_mdp.early_standup_penalty,
        weight=-5.0,
        params={
            "desired_gravity": DESIRED_GRAVITY_REAR,
            "min_cos_angle": 0.85,
            "min_steps": 100,
        },
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    """Termination conditions for stand-up.

    * ``time_out`` — episode length exceeded (counted as time-out, not failure)
    * ``standing_success`` — robot reached bipedal stance and held it calmly
      for ~1 second (this is a SUCCESS termination, not a failure)

    NOTE: ALL contact-based terminations are intentionally removed.
    The robot starts in a quadruped pose on the ground — its trunk,
    hips, and head are all close to or touching the floor. During the
    stand-up transition, ground contact is physically unavoidable.
    Contact-based terminations cause 100% early termination and
    prevent any learning. Body contacts are penalized via rewards
    instead, which provides gradient signal without killing episodes.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    standing_success = DoneTerm(
        func=standup_mdp.standing_success,
        time_out=True,  # treated as time-out (not truncation penalty) by RL algo
        params={
            "desired_gravity": DESIRED_GRAVITY_REAR,
            "min_cos_angle": 0.90,       # ~25° from target
            "min_base_height": 0.45,     # near bipedal height
            "max_ang_vel": 1.0,          # must be calm
            "hold_steps": 50,            # hold for ~1s (50 × 0.02s)
        },
    )


# ---------------------------------------------------------------------------
# Physics presets (matching bipedal walk env)
# ---------------------------------------------------------------------------


@configclass
class StandupFlatPhysicsCfg(PresetCfg):
    """Physics backend presets for stand-up on flat terrain.

    Same contact limits as bipedal walk flat preset.
    Use ``presets=newton_mjwarp`` for Newton backend.
    """

    default: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=400,
            nconmax=200,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )
    physx = default


@configclass
class StandupRoughPhysicsCfg(PresetCfg):
    """Physics backend presets for stand-up on rough terrain.

    Same as bipedal walk rough preset — higher njmax/nconmax for mesh contacts.
    """

    default: PhysxCfg = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=600,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        default_shape_cfg=NewtonShapeCfg(margin=0.01),
        num_substeps=1,
        debug_mode=False,
    )
    physx = default


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------


@configclass
class RobotStandupEnvCfg(ManagerBasedRLEnvCfg):
    """Go2 bipedal stand-up on flat terrain.

    Uses :class:`PositiveRewardManagerBasedRLEnv` (per-step total reward
    clipped to >= 0) via the ``only_positive_rewards`` flag.

    Episode length: 10 seconds (generous — a real stand-up takes 2-4s,
    but we give extra time for the policy to figure it out + hold the
    stance for 1s).
    """

    only_positive_rewards: bool = False

    scene: RobotSceneCfg = RobotSceneCfg(
        num_envs=4096, env_spacing=2.5, replicate_physics=True
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics = StandupFlatPhysicsCfg()
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotStandupPlayEnvCfg(RobotStandupEnvCfg):
    """Eval / playback variant: fewer envs, no noise, no DR."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        # Disable DR for eval
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.add_rear_leg_mass = None
        self.events.randomize_base_com = None
        self.events.randomize_actuator_gains = None
        # Camera tracking
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.4)
        self.viewer.origin_type = "asset_root"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"


@configclass
class RobotStandupRoughEnvCfg(RobotStandupEnvCfg):
    """Stand-up on gentle rough terrain for robustness.

    Uses mild terrain DR (50% flat, 30% gentle rough, 20% gentle slopes)
    to train a policy that works on imperfect ground without making the
    task too hard.
    """

    def __post_init__(self):
        super().__post_init__()

        # Physics: rough-terrain preset (higher contact limits)
        self.sim.physics = StandupRoughPhysicsCfg()

        # Terrain: swap plane for procedural gentle terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = STANDUP_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = 0

        # Slightly wider initial tilt to handle uneven ground
        # Rough terrain: even wider orientation DR
        self.events.reset_prone.params["roll_range"] = (-0.5, 0.5)
        self.events.reset_prone.params["pitch_range"] = (-0.4, 0.4)


@configclass
class RobotStandupRoughPlayEnvCfg(RobotStandupRoughEnvCfg):
    """Rough terrain play/eval variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        # Disable DR for eval
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.add_rear_leg_mass = None
        self.events.randomize_base_com = None
        self.events.randomize_actuator_gains = None
        # Camera tracking
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.4)
        self.viewer.origin_type = "asset_root"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"
