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
    """Rewards shaped for a slow, controlled stand-up motion.

    Positive terms:
    * ``orientation_align`` — monotonic pull toward bipedal stance
    * ``base_height`` — pull toward bipedal target height (negative weight
      on L2 deviation)
    * ``success_bonus`` — large one-time bonus when the robot reaches the
      standing pose (via is_alive gated on the hold counter)

    Negative / regularization terms (HEAVY — encourages gentleness):
    * ``action_rate`` — penalizes change in actions (jerky motions)
    * ``joint_vel`` — penalizes fast joint velocities
    * ``joint_acc`` — penalizes joint accelerations (jerk)
    * ``joint_torques`` — penalizes large torques
    * ``energy`` — penalizes power consumption
    * ``ang_vel_xy`` — penalizes world-frame roll/pitch rates
    * ``lin_vel_z`` — penalizes vertical bouncing
    * Contact penalties (thigh, calf, head, front foot)

    All weights are tuned so the smoothness penalties dominate the gradient
    during the transition — the policy learns to stand up slowly because
    fast motions are expensive.
    """

    # -- Orientation: velocity-gated alignment toward rear-stance --
    # Primary positive signal. The robot only gets credit for being more
    # upright when it is moving gently (ang_vel < 1.5 rad/s, lin_vel
    # < 0.8 m/s). Fast "throw-to-vertical" trajectories earn almost no
    # orientation reward because the gate suppresses it during the
    # high-velocity phase. This is the core mechanism that encourages
    # the policy to take its time standing up.
    # -- Orientation: raw alignment toward rear-stance --
    # No velocity gate — the robot gets full credit for being upright
    # regardless of how fast it got there. Smoothness is enforced by
    # the penalty terms below, not by gating the primary reward.
    orientation_align = RewTerm(
        func=standup_mdp.orientation_align_raw,
        weight=3.0,
        params={
            "desired_gravity": DESIRED_GRAVITY_REAR,
        },
    )

    # -- Success bonus: per-step reward when holding the target pose --
    # Gives strong immediate signal for reaching AND holding the upright
    # pose (orientation > 0.90, height > 0.45m, ang_vel < 1.0 rad/s).
    # High weight makes holding the pose the most valuable thing.
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

    # -- Velocity near upright: slow down as you approach the target --
    # Free to move when flat, increasingly penalized for speed as the
    # robot gets more upright. Forces a slow, controlled final approach
    # and stable hold once upright.
    velocity_near_upright = RewTerm(
        func=standup_mdp.velocity_near_upright_penalty,
        weight=-1.5,
        params={
            "desired_gravity": DESIRED_GRAVITY_REAR,
            "onset_cos": 0.3,
            "full_cos": 0.85,
        },
    )

    # -- Height: pull toward bipedal target --
    # Weight kept moderate — the L2 penalty at quadruped height (0.34m)
    # vs bipedal target (0.55m) is already (0.21)^2 = 0.044 per step.
    # Too heavy a weight overwhelms the orientation signal early on.
    base_height = RewTerm(
        func=bipedal_mdp.base_height_l2,
        weight=-0.5,
        params={"target_height": BIPEDAL_TARGET_BASE_Z},
    )

    # -- Roll penalty (keep lateral tilt small) --
    orientation_xy = RewTerm(func=bipedal_mdp.orientation_xy_sq, weight=-0.05)

    # -- Alive bonus: small per-step reward for not terminating --
    is_alive = RewTerm(func=mdp.is_alive, weight=0.1)

    # -- SMOOTHNESS PENALTIES (the core of "gentle" stand-up) --

    # Action rate: penalize changes between consecutive actions
    # Much heavier than walk env (-0.005) to suppress jerky motions
    # Smoothness penalties — moderate, not aggressive. The robot needs
    # to actually move to stand up; over-penalizing motion creates a
    # policy that barely tilts upright and wobbles.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    energy = RewTerm(func=mdp.energy, weight=-1.0e-5)

    # Joint limits: hard penalty near limits
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # -- Stability penalties (world-frame) --

    # Vertical velocity: penalize bouncing / throwing
    lin_vel_z = RewTerm(func=bipedal_mdp.lin_vel_z_world_l2, weight=-1.0)

    # Roll/pitch rates: penalize fast rotations (world frame)
    ang_vel_xy = RewTerm(func=bipedal_mdp.ang_vel_xy_world_l2, weight=-0.1)

    # Horizontal drift: penalize sliding backward/forward/sideways
    # The robot should stand up in place, not drift around
    lin_vel_xy = RewTerm(func=standup_mdp.lin_vel_xy_world_l2, weight=-0.5)

    # Yaw spin: penalize rotation around vertical axis
    ang_vel_z = RewTerm(func=standup_mdp.ang_vel_z_world_l2, weight=-0.3)

    # Rear leg symmetry: penalize left/right leg asymmetry
    # Forces both rear legs to move similarly → no "dancing"
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

    # -- Front leg regularization --
    # Keep front legs tucked — they shouldn't flail during stand-up
    front_hip_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.25,
        params={"joint_patterns": ["F[LR]_hip_joint"]},
    )
    front_thigh_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.15,
        params={"joint_patterns": ["F[LR]_thigh_joint"]},
    )
    front_calf_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.15,
        params={"joint_patterns": ["F[LR]_calf_joint"]},
    )
    front_leg_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["F[LR]_.*"]),
        },
    )

    # -- Contact penalties --
    # Base (trunk) contact: not terminal (see TerminationsCfg note) but
    # heavily penalized. The robot should learn to get its body off the
    # ground as quickly as possible.
    base_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
        },
    )
    hip_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip"),
        },
    )
    thigh_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
        },
    )
    # Only penalize FRONT calf contact. Rear calves are the support
    # structure in bipedal stance — they SHOULD touch the ground.
    front_calf_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F[LR]_calf"),
        },
    )
    front_foot_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F[LR]_foot"),
        },
    )
    head_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*"]),
        },
    )

    # -- Rear foot contact reward (Option 1: encourage feet-on-ground) --
    # The bipedal walk policy expects the handoff pose to be standing on
    # rear FEET (R[LR]_foot), not rear calves. Without this, the standup
    # policy may learn to balance on calves, creating a pose the walk
    # policy has never seen. This reward incentivizes the robot to get
    # its rear feet firmly planted on the ground in the final upright pose.
    # Gated by orientation so it only activates once the robot is mostly
    # upright (avoids rewarding random foot contact during early standup).
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

    # -- Rear calf contact penalty --
    # Soft ramp: calf contact costs nothing when flat, ramps to full
    # penalty as the robot approaches upright. Gives a smooth gradient
    # to shift weight from calves to feet without scaring it away from
    # standing up.
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
