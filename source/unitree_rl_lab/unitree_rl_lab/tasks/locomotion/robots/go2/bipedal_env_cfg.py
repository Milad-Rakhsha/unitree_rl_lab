"""Go2 bipedal **walking** environment (rear-legs stance).

Single configuration class :class:`RobotBipedalWalkEnvCfg` (task id
``Unitree-Go2-Bipedal-Walk``) that trains the Go2 to stand on its hind legs
*and* walk to velocity commands. Flat ground, flat quadruped spawn.

The command space is restricted to **forward/backward linear velocity
and yaw rate** (``lin_vel_y`` pinned to zero). The task is simpler than
an omnidirectional bipedal gait because the rear legs only need to
produce a sagittal-plane walking pattern plus in-place turning — not
lateral side-stepping — which matches the Go2 rear-leg kinematic range.

The reward shape follows the arclab-hku Go1 baseline / TumblerNet recipe
(npj Robotics 2025) with Go2-specific weights tuned across several
iterations to escape successive local optima:

* **Monotonic** orientation reward (``max(0, dot(proj_grav_b, target))``)
  with weight ``2.0`` — dominant positive term, so the marginal reward
  of going from a partial tilt to a full bipedal stance outweighs the
  motion penalties that grow with rear-leg extension.
* **L1 motion penalties** split front / rear against ``default_joint_pos``.
  Front legs keep the arclab-hku Go1 weights (``hip=-0.15``,
  ``thigh=-0.05``, ``calf=-0.05``) so they stay tucked and cannot carry
  weight. Rear legs use ``-0.05 / -0.015 / -0.015`` — small enough that
  the marginal walking reward exceeds the marginal regulariser cost, but
  non-zero so the rear legs cannot freeze into a splayed V-pose.
* **Quadratic** base-height penalty with weight ``-1.0`` — pulls toward
  the bipedal target without punishing the COM bob of a normal walking
  gait. The earlier ``-3.0`` was used to escape the initial crouch
  local optimum; once the stance is solved a lighter weight stops the
  policy from freezing in place to avoid height oscillation.
* **Softened motion-penalty budget** on the world-frame vertical and
  roll/pitch velocities (``lin_vel_z`` ``-0.5`` / ``ang_vel_xy``
  ``-0.05``) and on ``action_rate`` (``-0.005``). The previous
  combination (``-2.0 / -0.05 / -0.02``) added ~``-0.35`` per step,
  which exceeded the marginal tracking reward of an early walking
  attempt; the policy therefore preferred a stationary wobble to any
  vigorous swing motion.
* **Front-foot contact penalty** (``weight=-1.5`` per foot) — blocks the
  "4-foot inverted dog" cheat where the policy tilts the body ~90° and
  stands on all four feet (rear splayed behind, front reaching to the
  ground), observed in the 2026-04-18_12-28-55 run.
* **Reward-gated velocity command curriculum** (``lin_vel_cmd_levels`` /
  ``ang_vel_cmd_levels``): ranges start at ``+/- 0.1`` and widen toward
  ``+/- 1.0`` only when the policy tracks the current ranges well.
  Heading command is disabled so the ang-vel curriculum is not bypassed.
* **Gravity-aligned velocity tracking and penalties.** Linear tracking
  uses ``track_lin_vel_xy_yaw_frame_exp`` (weight ``2.0``,
  ``std=sqrt(0.5)`` — wide enough that a ~0.5 m/s tracking error during
  early walking still yields a meaningful positive gradient), and
  yaw-rate tracking uses ``track_ang_vel_z_world_exp`` (weight
  ``0.75``, ``std=sqrt(4)``). The yaw-tracking parameters were widened
  after observing that the intrinsic balance-yaw wobble of a bipedal
  robot (~0.6–0.8 rad/s) permanently gated a narrower ang-vel reward
  and stalled the curriculum at its initial range. Vertical-velocity / roll-pitch-rate penalties use the
  world-frame ``lin_vel_z_world_l2`` / ``ang_vel_xy_world_l2`` helpers.
  At the bipedal tilt (body pitched ~+pi/2) body-X points along world
  +Z, so body-frame rewards would ask the policy to move *upward* for a
  "forward" command and penalise the yaw rate it is supposed to track.
* **Level (yaw-frame) debug arrows.** The command uses
  :class:`~unitree_rl_lab.tasks.locomotion.mdp.commands.UniformLevelVelocityCommand`,
  which rotates the goal arrow only by the yaw component of the base
  and draws the current arrow from the world-frame linear velocity, so
  the viewer visualisation stays on the horizontal plane at any body tilt.
* **``only_positive_rewards``** clip on the per-step total — penalties
  still shape the gradient via term-level differences, but the agent is
  never pushed into a net-negative step.
* **Flat quadruped spawn** + ~1 m/s random pushes.
* **Rear-foot air-time shaping** (``feet_air_time_positive_biped`` on
  ``R[LR]_foot``). Rewards single-stance air time on the rear feet
  whenever a non-trivial linear command is issued. Velocity tracking
  alone produced a shuffling stance that barely lifted either rear
  foot; the air-time reward actively pushes the gait toward a clean
  alternating single-stance pattern. Front feet are excluded — they
  stay tucked under ``front_foot_contact``.
* **No ``bad_orientation`` / ``base_too_low`` terminations.**
  Terminations are limited to base-body and hip-link contact (the latter
  matching the arclab-hku ``["base", "trunk", "hip"]`` list) — falling is
  cheap, the policy can try again within the same episode.

Stage-2 observation: the policy group stacks ``num_history = 4`` frames of
proprioception (``base_ang_vel``, ``projected_gravity``, ``joint_pos_rel``,
``joint_vel_rel``). This gives an MLP actor enough temporal context to
implicitly estimate body linear velocity / CoM motion without the full
TumblerNet Estimator Net + auxiliary regression head.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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

# ---------------------------------------------------------------------------
# Stance constants (rear-legs stance only)
# ---------------------------------------------------------------------------

# projected_gravity_b of the body in the target stance pose (unit vector):
# rear stance (nose up, hind legs on ground) -> body pitched ~+pi/2 about
# body-y so world-down maps to -body_x.
DESIRED_GRAVITY_REAR = [-1.0, 0.0, 0.0]

# World-frame base z target when fully upright on two legs (Go2 specific).
BIPEDAL_TARGET_BASE_Z = 0.55

# History length for proprioceptive observations fed to the actor (Stage 2).
OBS_HISTORY_LENGTH = 4


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Infinite-plane Go2 scene with contact sensing (no terrain generator)."""

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
# Events (reset + domain randomization)
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """Flat quadruped spawn + mild domain randomization.

    ``randomize_rigid_body_material`` is PhysX-only and is not supported by
    the Newton solver; upstream ``velocity_env_cfg.py`` comments it out for
    the Newton port and we follow the same pattern here.
    """

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.8, 2.0),
            "operation": "add",
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}},
    )


# ---------------------------------------------------------------------------
# Commands (reward-gated curriculum on lin/ang vel ranges)
# ---------------------------------------------------------------------------


@configclass
class CommandsCfg:
    """Forward/backward + yaw velocity command with a reward-gated curriculum.

    Lateral (``lin_vel_y``) is pinned to zero in both ``ranges`` and
    ``limit_ranges`` — the task is forward/backward walking plus
    in-place yaw rotation only. Sagittal-plane walking is achievable with
    the Go2 rear-leg kinematics, whereas lateral side-stepping on two
    legs requires a much wider hip abduction range than the robot has.

    The policy starts with a small linear range and a moderate yaw range
    and the :func:`~unitree_rl_lab.tasks.locomotion.mdp.lin_vel_cmd_levels`
    / :func:`~unitree_rl_lab.tasks.locomotion.mdp.ang_vel_cmd_levels`
    curricula widen them by ``+/- 0.1`` per episode once the
    corresponding tracking reward exceeds ``0.8 * weight``. The ang-vel
    range starts at ``+/- 0.3`` rather than ``+/- 0.1`` because the
    intrinsic balance-yaw wobble of the bipedal stance dominates the
    error signal when the command is smaller than the wobble magnitude,
    which kept earlier runs stuck at the initial curriculum level.

    Heading command is disabled so the ``ang_vel_z`` curriculum actually
    gates the yaw commands the policy sees, rather than those commands
    being overwritten by heading control.
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-0.3, 0.3),
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
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
# Observations (Stage 2: proprio history on the actor)
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Actor sees a 4-step history of proprioception; critic sees the full
    single-frame state (including ground-truth base linear velocity).

    History is applied **per-term** rather than at the group level so
    ``velocity_commands`` (stateless, user-specified) and ``last_action``
    (already a lagged copy of the policy output) are not duplicated across
    timesteps — that would waste input dimension with no information gain.
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
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
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
# Rewards (TumblerNet / arclab-hku port)
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Rear-stance bipedal walking rewards.

    Positive terms: velocity tracking, monotonic orientation alignment,
    rear-foot single-stance air time.
    Negative terms: base-height deviation, vertical velocity, roll/pitch
    rate, per-joint-group motion L1, action rate, joint-limit
    violation, non-foot contacts (thighs / calves / head / front feet).

    No torque / acceleration / energy penalties — they scale with the
    holding torque the bipedal pose requires and deter exploration of
    fast-recovery motions.
    """

    # -- tracking (ungated) --
    #
    # Yaw-frame (gravity-aligned) linear tracking and world-frame yaw-rate
    # tracking. The body-frame variants used for flat quadruped locomotion
    # are wrong here: at the ~90° bipedal tilt, body-X points along world-Z,
    # so ``track_lin_vel_xy_exp`` asks for vertical motion and
    # ``track_ang_vel_z_exp`` asks for a roll rate instead of a yaw rate.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(4.0)},
    )

    # -- orientation (monotonic pull toward rear-stance, + small xy penalty) --
    #
    # Weight bumped from 0.8 -> 2.0 so the marginal reward of going from a
    # partial crouch (dot~=0.5) to a full bipedal stance (dot=1.0) dominates
    # the joint-motion penalties that grow with rear-leg extension.
    orientation_align = RewTerm(
        func=bipedal_mdp.orientation_align,
        weight=2.0,
        params={"desired_gravity": DESIRED_GRAVITY_REAR},
    )
    # Penalises roll (proj_grav_b.y^2) — the large x-component in the
    # target stance is tolerated because the weight is tiny.
    orientation_xy = RewTerm(func=bipedal_mdp.orientation_xy_sq, weight=-0.03)

    # -- height & stability --
    #
    # ``base_height`` weight softened from -3.0 -> -1.0. The strong -3.0
    # was needed to pull the robot out of the quadruped crouch, but once
    # the bipedal pose is solved (base height within a few cm of target)
    # the penalty is just punishing the natural COM bob of a walking
    # gait. ``lin_vel_z`` softened from -2.0 -> -0.5 for the same reason:
    # lifting a leg to step inherently produces world-Z velocity in the
    # body, and the strong -2.0 turns "stand perfectly still" into the
    # only affordable strategy.
    base_height = RewTerm(
        func=bipedal_mdp.base_height_l2,
        weight=-1.0,
        params={"target_height": BIPEDAL_TARGET_BASE_Z},
    )
    # World-frame (gravity-aligned) versions so these penalties do not
    # collide with the tilted body at the bipedal stance — the body-frame
    # ``lin_vel_z_l2`` would penalise *horizontal* motion, and the body-frame
    # ``ang_vel_xy_l2`` would penalise the yaw rate the command is tracking.
    lin_vel_z = RewTerm(func=bipedal_mdp.lin_vel_z_world_l2, weight=-0.5)
    ang_vel_xy = RewTerm(func=bipedal_mdp.ang_vel_xy_world_l2, weight=-0.05)

    # -- motion regularizers (L1 against default joint pos) --
    #
    # Split front / rear: the rear legs carry the entire body weight and
    # must sweep through a large range to actually walk, so their L1
    # penalties are scaled down by 3x relative to the front legs. Front
    # legs are regularised with the original arclab-hku Go1 weights to
    # keep them tucked close to their default positions (so the body
    # cannot pivot forward onto them). Removing the rear L1 entirely
    # lets the rear legs splay behind the body in a static V-pose, so
    # keep a small non-zero weight.
    front_hip_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.15,
        params={"joint_patterns": ["F[LR]_hip_joint"]},
    )
    front_thigh_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.05,
        params={"joint_patterns": ["F[LR]_thigh_joint"]},
    )
    front_calf_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.05,
        params={"joint_patterns": ["F[LR]_calf_joint"]},
    )
    rear_hip_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.05,
        params={"joint_patterns": ["R[LR]_hip_joint"]},
    )
    rear_thigh_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.015,
        params={"joint_patterns": ["R[LR]_thigh_joint"]},
    )
    rear_calf_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.015,
        params={"joint_patterns": ["R[LR]_calf_joint"]},
    )

    # -- action / joint-limit regularizers --
    #
    # ``action_rate`` softened -0.02 -> -0.005. The policy must make
    # fast swing-leg action changes to walk at all; at -0.02 the per-step
    # penalty reached ~0.25 (14% of the positive reward sum), which
    # strictly disincentivises vigorous motion. -0.005 keeps regularisation
    # without making stillness the cheapest option.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)

    # -- gait shaping --
    #
    # Rewards the rear feet for long single-stance air-times (one foot in
    # the air at a time), gated on non-zero linear command magnitude.
    # Encourages a proper walking gait instead of the shuffling / static
    # stance observed in earlier runs. Front feet are excluded — they
    # stay tucked and are penalised separately if they touch the ground.
    rear_feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.4,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[LR]_foot"),
        },
    )

    # -- contact penalties --
    #
    # ``thigh_contact`` / ``calf_contact`` / ``head_contact`` penalise
    # non-foot body parts touching the ground. ``calf_contact`` is kept
    # at a larger magnitude than ``thigh_contact`` because calf contact
    # is the common failure mode when the rear legs partially collapse
    # during walking (the knee folds, the calf scrapes the ground).
    # ``front_foot_contact`` forbids the front feet from supporting body
    # weight — this is the direct anti-cheat for the "4-foot splay" pose
    # where the policy tilts the body ~90° and stands on all four feet.
    thigh_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"),
        },
    )
    calf_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-3.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
        },
    )
    front_foot_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.5,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="F[LR]_foot"),
        },
    )
    head_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*"]),
        },
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    """Terminations: time-out + hard base/hip contact.

    ``bad_orientation`` and ``base_too_low`` are intentionally absent —
    those kill exploration at the flat spawn (the policy fails once, dies,
    and never gets to practice recovery).

    ``base_contact`` and ``hip_contact`` mirror the arclab-hku Go1 reference
    (``terminate_after_contacts_on = ["base", "trunk", "hip"]``). The hip
    termination is a safety barrier against "sitting-on-hip" poses where
    the policy rotates the body backward so the hip brackets rest on the
    ground — that geometry only occurs for non-bipedal behaviour.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )
    hip_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_hip"),
            "threshold": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# Curriculum (reward-gated command widening)
# ---------------------------------------------------------------------------


@configclass
class CurriculumCfg:
    """Widens the velocity command ranges when the policy tracks them well.

    Both terms read the mean episodic tracking reward at each episode
    boundary and step ``ranges.*`` outwards by ``+/- 0.1`` up to
    ``limit_ranges.*`` whenever the reward exceeds ``0.8 * weight``.
    """

    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)


# ---------------------------------------------------------------------------
# Env cfg
# ---------------------------------------------------------------------------


@configclass
class RobotBipedalWalkEnvCfg(ManagerBasedRLEnvCfg):
    """Single configuration for Go2 rear-stance bipedal walking.

    The env wrapper (see ``__init__.py`` registration) is
    :class:`~unitree_rl_lab.tasks.locomotion.robots.go2.bipedal_env.PositiveRewardManagerBasedRLEnv`,
    which clips the per-step **total** reward to ``>= 0`` when
    :attr:`only_positive_rewards` is ``True``. Per-term values and their
    episodic sums used for TensorBoard logging are not clipped.
    """

    only_positive_rewards: bool = True

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # IsaacLab 3.0: ``SimulationCfg.physics`` defaults to ``None``.
        # Assign a new :class:`PhysxCfg` instance instead of mutating
        # ``self.sim.physx.<attr>`` (that API no longer exists).
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics = PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotBipedalWalkPlayEnvCfg(RobotBipedalWalkEnvCfg):
    """Eval / playback variant: fewer envs, no observation noise, no pushes.

    Sets ``ranges = limit_ranges`` so the user can issue full-range
    commands to the trained policy immediately — the curriculum is left
    in place but becomes a no-op once the ranges already equal the
    limits.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
