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
  Front legs use ``hip=-0.25``, ``thigh=-0.10``, ``calf=-0.10`` (roughly
  2x the original arclab-hku Go1 weights) to keep the arms firmly tucked
  and damp the arm-waving that otherwise appears during rear-leg walking.
  Rear legs use ``-0.05 / -0.015 / -0.015`` — small enough that the
  marginal walking reward exceeds the marginal regulariser cost, but
  non-zero so the rear legs cannot freeze into a splayed V-pose.
* **Front-leg joint-velocity L2 penalty** (``-0.001`` on ``F[LR]_.*``).
  The L1 position penalty above pulls the arms toward default but does
  not punish high-frequency oscillation about default; this term is a
  direct dynamic damper on the arms.
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
  ``ang_vel_cmd_levels``): linear ranges start at ``+/- 0.1`` and yaw
  ranges at ``+/- 0.5``, both widen toward ``+/- 1.0`` only when the
  policy tracks the current ranges well. The yaw curriculum uses a
  lower reward-threshold fraction (``0.6`` of weight) than the linear
  curriculum (``0.8`` of weight) because the tightened yaw kernel caps
  the achievable mean reward below ``weight`` at the bipedal wobble
  floor. Heading command is disabled so the ang-vel curriculum is not
  bypassed.
* **Gravity-aligned velocity tracking and penalties.** Linear tracking
  uses the task-local
  :func:`~unitree_rl_lab.tasks.locomotion.robots.go2.bipedal_mdp.track_lin_vel_xy_bipedal_frame_exp`
  (weight ``2.0``, ``std=sqrt(0.5)`` — wide enough that a ~0.5 m/s
  tracking error during early walking still yields a meaningful positive
  gradient), and yaw-rate tracking uses ``track_ang_vel_z_world_exp``
  (weight ``1.0``, ``std=sqrt(1.0)``). An earlier iteration used
  ``std=sqrt(4)``, which flattened the reward so much that the policy
  could ignore every yaw command and still collect ~88% of the maximum
  reward at a 1 rad/s error. The tightened kernel is paired with a
  0.6 * weight curriculum threshold and a widened initial yaw range
  (``+/- 0.5`` rad/s instead of ``+/- 0.3`` rad/s) so the balance-yaw
  wobble (~0.5-0.8 rad/s) doesn't stall the curriculum. Vertical-velocity
  / roll-pitch-rate penalties use the world-frame ``lin_vel_z_world_l2``
  / ``ang_vel_xy_world_l2`` helpers. At the bipedal tilt (body pitched
  ~+pi/2) body-X points along world +Z, so body-frame rewards would
  ask the policy to move *upward* for a "forward" command and penalise
  the yaw rate it is supposed to track.
* **Pitch-invariant yaw-frame tracking.** EXPERIMENTAL (2026-04-18).
  The upstream ``track_lin_vel_xy_yaw_frame_exp`` uses
  :func:`~isaaclab.utils.math.yaw_quat`, whose ZYX Euler extraction is
  singular at ``±pi/2`` body pitch — exactly the bipedal target. At the
  singularity the returned yaw collapses to floating-point noise; any
  tiny crossing of the singularity (from a foot-strike impulse or an
  inverted-pendulum wobble) flips the extracted yaw by ``pi``, silently
  inverting the meaning of the ``lin_vel_x`` command. The policy
  reconciles the opposing gradients on either side of vertical by
  ignoring the linear command and defaulting to a constant body-frame
  drift, which manifests at play time as "every robot walks backward".
  The local ``track_lin_vel_xy_bipedal_frame_exp`` uses
  :func:`~unitree_rl_lab.tasks.locomotion.robots.go2.bipedal_mdp.bipedal_yaw_quat`,
  which extracts yaw from the horizontal projection of body-Y (the roll
  axis) and stays continuous through the bipedal singularity while
  agreeing with :func:`~isaaclab.utils.math.yaw_quat` at flat poses.
* **Level (yaw-frame) debug arrows.** The command uses
  :class:`~unitree_rl_lab.tasks.locomotion.mdp.commands.UniformLevelVelocityCommand`
  with ``pitch_invariant_yaw=True``, which rotates the goal arrow only
  by the pitch-invariant yaw of the base and draws the current arrow
  from the world-frame linear velocity, so the viewer visualisation
  stays on the horizontal plane at any body tilt and does not flip
  across the bipedal singularity.
* **``only_positive_rewards``** clip on the per-step total — penalties
  still shape the gradient via term-level differences, but the agent is
  never pushed into a net-negative step.
* **Flat quadruped spawn** + every-4-7 s ~1 m/s random pushes.
* **Domain randomization for sim-to-sim / sim-to-real transfer.** The
  training env now randomizes per-body friction + restitution, per-link
  mass (base + rear-leg thigh/calf), base CoM offset (+/- 3 cm in x/y,
  +/- 2 cm in z), and actuator PD-gain scale (+/- 15% on kp/kd). An
  earlier training run with only base-mass randomization trained policies
  that flipped backward on handover in Mujoco; the added terms break
  that over-fit to PhysX-specific dynamics.
* **Rear-foot air-time shaping** (``feet_air_time_positive_biped`` on
  ``R[LR]_foot``, ``threshold=0.25`` s, ``weight=1.5``). Rewards
  single-stance air time on the rear feet whenever a non-trivial linear
  command is issued. Velocity tracking alone produced a shuffling stance
  that barely lifted either rear foot; the air-time reward actively
  pushes the gait toward a clean alternating single-stance pattern.
  Front feet are excluded — they stay tucked under ``front_foot_contact``.
* **Flight-phase penalty** (``feet_both_airborne`` on ``R[LR]_foot``,
  ``weight=-0.75``). The air-time reward above is zero (not negative)
  when both rear feet leave the ground simultaneously, so the policy
  could still collect linear-tracking reward by pronking forward through
  the world — a local optimum that does not require learning alternating
  leg coordination. This penalty makes the two-feet-airborne state
  strictly negative whenever a non-trivial linear command is active.
* **Swing-foot clearance reward** (``feet_clearance_reward`` on
  ``R[LR]_foot``, ``target_height=0.10`` m, ``std=0.05``,
  ``tanh_mult=2.0``, ``weight=0.5``). Single-stance air time alone does
  not require the swing foot to meaningfully lift — the policy found a
  dragging local optimum where the swing foot scrapes just above the
  floor. This Spot-ported term pays off for bringing any moving foot to
  the target height; stationary stance feet are masked out by the
  ``tanh(|v_horiz|)`` gate and do not contribute.
* **In-contact foot-slide penalty** (``feet_slide`` on ``R[LR]_foot``,
  ``weight=-0.25``). Punishes horizontal foot velocity while the foot is
  loaded, which is the other half of dragging (foot in contact AND
  moving). Together with the clearance reward it pushes the gait from
  a shuffle toward a clean step-through pattern.
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
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_tasks.utils import PresetCfg
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
# Terrain configuration (rough terrain for robust bipedal walking)
# ---------------------------------------------------------------------------

GO2_BIPEDAL_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.3),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.35, noise_range=(0.01, 0.04), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.25, slope_range=(0.0, 0.15), platform_width=1.5, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.025, 0.08), platform_width=1.5
        ),
    },
)

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
    """Flat quadruped spawn + domain randomization tuned for sim-to-sim transfer.

    EXPERIMENTAL (2026-04-18): the previous config only randomized base mass
    and issued mild random pushes; the resulting policies overfit to the
    PhysX friction / contact / PD-gain defaults and collapsed immediately
    on the Mujoco side (the robot flipped backward on handover). The set
    of randomizations added here is the standard sim-to-sim / sim-to-real
    recipe for quadruped locomotion ported to the bipedal stance:

    * per-shape ground + robot friction + restitution (hardware floors are
      never the 1.2 static / 1.0 dynamic Isaac Lab defaults);
    * per-link mass on every body (not only the base) so the policy does
      not memorise the exact inertia tensor of the USD file;
    * base CoM offset (a ~1 cm CoM shift changes the bipedal balance
      lever-arm substantially and is well within USD-vs-URDF drift);
    * actuator PD-gain scaling (real motor firmware is calibrated to
      +/- a few percent of the nominal kp/kd);
    * more frequent and stronger random pushes so the policy learns to
      recover instead of riding through flat-ground open-loop trajectories.
    """

    # EXPERIMENTAL: static/dynamic friction + restitution randomization across
    # all robot bodies. The range brackets the Go2 rubber-foot friction on
    # wet/dry laminate, concrete, and Mujoco's default plane friction
    # (~0.6-1.0). ``num_buckets=64`` keeps the PhysX material count well
    # under the 64k cap. This is the single most important DR term for
    # sim-to-sim transfer because the two simulators model friction
    # differently and the policy must not encode PhysX-specific friction.
    #
    # EXPERIMENTAL (2026-04-18): shifted range from (0.4, 1.2) to
    # (0.5, 1.3) so the effective foot-ground friction (mu_robot * 1.0
    # ground) is centred around mu=0.9 instead of mu=0.8. The previous
    # lower edge of 0.4 is lower than any credible real-world or Mujoco
    # setting for a Go2 rubber foot; burning rollouts on that corner
    # only hurts the bipedal policy, which is friction-margin-sensitive.
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

    # EXPERIMENTAL: per-link mass add. The base gets a wider window than the
    # limbs because payload / battery variance dominates on hardware.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.5),
            "operation": "add",
        },
    )
    # EXPERIMENTAL: rear-leg mass add. The rear limbs carry the robot weight
    # in the bipedal stance; a +/- small mass offset on thigh/calf changes
    # the required swing torque and is what differs most between PhysX and
    # Mujoco USD inertia bakeouts.
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

    # EXPERIMENTAL: base CoM offset. +/- 3 cm in x/y and +/- 2 cm in z brackets
    # the realistic hardware mounting tolerance plus any battery-placement
    # shift. For a bipedal stance this is a big deal: a 3 cm x-shift of the
    # base CoM changes the zero-moment balance pole by about the same amount.
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

    # EXPERIMENTAL: PD-gain scaling on startup. Real motor firmware has +/-
    # a few % calibration variance per motor and per robot; Mujoco's
    # actuator model reacts slightly differently to the same nominal kp/kd
    # than PhysX. ``scale`` operation is multiplicative around the nominal
    # (25, 0.5) gains, which preserves the relative stiffness/damping ratio
    # each policy rollout assumes.
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

    # EXPERIMENTAL (2026-04-18): switched from ``reset_joints_by_scale``
    # (which multiplies the default by a random factor and so gives
    # negligible randomization on near-zero defaults like the front hips)
    # to ``reset_joints_by_offset`` with +/- 0.1 rad per joint. The old
    # ``position_range=(1.0, 1.0)`` was a no-op: every rollout spawned at
    # exactly ``default_joint_pos``, so the policy never saw the small
    # pose mismatch it gets on the Mujoco side at the end of the
    # ``BipedalRear`` intro ramp (joints track the intro target under
    # kp=25 with a few-degree steady-state error). +/- 0.1 rad is
    # isotropic across the 12 joints and brackets that tracking error
    # while staying well inside the quadruped-standable envelope.
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.1, 0.1), "velocity_range": (-1.0, 1.0)},
    )

    # EXPERIMENTAL: pushes made more frequent (every 4-7 s instead of 10-15 s)
    # and extended to all three horizontal + vertical components. A bipedal
    # stance is an inverted pendulum — without frequent pushes the policy
    # over-fits to the open-loop trajectories of the training rollouts and
    # has no margin for the contact-timing differences between PhysX and
    # Mujoco. Pushes are kept small (1 m/s horizontal, 0.3 m/s vertical)
    # to stay within a recoverable perturbation.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 7.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-0.3, 0.3),
            }
        },
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
    corresponding tracking reward exceeds a fraction of ``weight``
    (``0.8`` for linear, ``0.6`` for yaw — see :class:`CurriculumCfg`).
    The ang-vel range starts at ``+/- 0.5`` rather than ``+/- 0.1``
    because the intrinsic balance-yaw wobble of the bipedal stance
    dominates the error signal when the command is smaller than the
    wobble magnitude, which kept earlier runs stuck at the initial
    curriculum level.

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
        # EXPERIMENTAL (2026-04-18): bipedal stance target is ~90 deg body
        # pitch, which is the ZYX-Euler gimbal-lock singularity. The debug
        # goal-velocity arrow must use a pitch-invariant yaw extraction
        # (body-Y horizontal projection) or it flips by pi each time the
        # body crosses the singularity from either side.
        pitch_invariant_yaw=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1),
            lin_vel_y=(0.0, 0.0),
            # EXPERIMENTAL (2026-04-18): yaw start range widened 0.3 -> 0.5.
            # Paired with the tightened ``track_ang_vel_z`` kernel (``std=1``),
            # a 0.5 rad/s commanded yaw gives the policy a clear gradient
            # away from the ~0.5 rad/s intrinsic wobble, whereas a 0.3 rad/s
            # command is indistinguishable from the wobble in reward space.
            ang_vel_z=(-0.5, 0.5),
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
    rear-foot single-stance air time, rear-foot swing clearance.
    Negative terms: base-height deviation, vertical velocity, roll/pitch
    rate, per-joint-group motion L1, front-leg joint-velocity L2, action
    rate, joint-limit violation, non-foot contacts (thighs / calves /
    head / front feet), two-rear-feet-airborne flight, in-contact rear
    foot slide.

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
    #
    # EXPERIMENTAL (2026-04-18): swapped from the upstream
    # ``track_lin_vel_xy_yaw_frame_exp`` to a local pitch-invariant variant
    # ``bipedal_mdp.track_lin_vel_xy_bipedal_frame_exp``. The upstream
    # helper uses ``yaw_quat`` (ZYX Euler extraction), which is singular
    # at a ``±pi/2`` body pitch — exactly the bipedal stance target. At
    # the singularity the extracted yaw collapses to rounding noise; a
    # tiny crossing of the singularity (from foot-strike impulses or the
    # inverted-pendulum wobble) flips the extracted yaw by ``pi``,
    # silently inverting the meaning of the ``lin_vel_x`` command. During
    # training this produces opposing gradients on either side of vertical
    # that the policy reconciles by ignoring the linear command entirely
    # and defaulting to a constant backward body-frame drift. The new
    # helper extracts yaw from the horizontal projection of body-Y (roll
    # axis), which is continuous through the bipedal singularity.
    track_lin_vel_xy = RewTerm(
        func=bipedal_mdp.track_lin_vel_xy_bipedal_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    )
    # EXPERIMENTAL (2026-04-18): tightened yaw-rate tracking from std=sqrt(4.0)
    # to std=sqrt(1.0) and bumped weight 0.75 -> 1.0. The prior kernel was
    # effectively flat over the entire command range (``exp(-1/4) ~= 0.78`` at
    # 1 rad/s error), so the policy was receiving ~88% of max reward while
    # completely ignoring yaw commands. The new kernel puts the inflection at
    # the command magnitudes the policy is actually asked to track. The
    # curriculum threshold for this term is lowered to 0.6 * weight in the
    # corresponding ``ang_vel_cmd_levels`` term (see ``CurriculumCfg``) to
    # account for the intrinsic wobble of the bipedal stance; 0.8 * weight
    # would stall the curriculum at the initial range.
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(1.0)},
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
        weight=-2.0,
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
    # penalties are scaled down relative to the front legs. Front legs
    # are regularised harder to keep them tucked close to their default
    # positions (so the body cannot pivot forward onto them *and* the
    # arms don't swing around as visual noise). Removing the rear L1
    # entirely lets the rear legs splay behind the body in a static
    # V-pose, so keep a small non-zero weight there.
    #
    # EXPERIMENTAL (2026-04-18): front-leg L1 weights roughly doubled
    # (hip -0.15 -> -0.25, thigh -0.05 -> -0.10, calf -0.05 -> -0.10)
    # to calm the arm-waving observed during rear-leg walking. The front
    # legs have no functional role in a bipedal gait and each joint was
    # drifting +/- 0.5 rad from default in the earlier run; the policy
    # was effectively using them as free inertia-shaping actuators.
    front_hip_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.25,
        params={"joint_patterns": ["F[LR]_hip_joint"]},
    )
    front_thigh_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.10,
        params={"joint_patterns": ["F[LR]_thigh_joint"]},
    )
    front_calf_motion = RewTerm(
        func=bipedal_mdp.joint_deviation_from_default_l1,
        weight=-0.10,
        params={"joint_patterns": ["F[LR]_calf_joint"]},
    )
    # EXPERIMENTAL (2026-04-18): direct L2 penalty on front-leg joint
    # velocities. The position-deviation penalties above pull the arms
    # back toward default but do not punish high-frequency oscillation
    # about default — the policy was seen rapidly flicking the arms to
    # shift CoM with near-zero mean deviation. This term is a dynamic
    # damper on the arms with a small weight, tuned so the per-step
    # penalty for an idle-arm policy is ~0 while a fast-waving arm
    # (|q_dot| ~ 5 rad/s per joint x 6 joints) pays ~0.15/step.
    front_leg_joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["F[LR]_.*"]),
        },
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
    # Rewards the rear feet for single-stance air-times (one foot in the
    # air at a time), gated on non-zero linear command magnitude. Encourages
    # an alternating walking gait instead of the shuffling / static stance
    # observed in earlier runs. Front feet are excluded — they stay tucked
    # and are penalised separately if they touch the ground.
    #
    # EXPERIMENTAL (2026-04-18): threshold lowered 0.4 -> 0.25 and weight
    # raised 1.0 -> 1.5. The 0.4 s target required the rear-leg swing to be
    # longer than one half-stride at ~1 m/s; the policy found it cheaper to
    # pronk (both feet airborne together) than to time a 0.4 s single-stance
    # swing. 0.25 s matches a realistic single-step air-time at walking
    # pace, and the higher weight makes the single-stance reward large
    # enough to dominate the local-optimum rewards from shuffling.
    rear_feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "threshold": 0.25,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[LR]_foot"),
        },
    )
    # EXPERIMENTAL (2026-04-18): explicit pronk / flight-phase penalty.
    # ``feet_air_time_positive_biped`` is zero when both rear feet are
    # airborne simultaneously but does not *punish* that state, so pronking
    # forward through the world was a free (zero-cost) way to accumulate
    # linear-velocity-tracking reward without the complexity of learning
    # alternating leg coordination. This term pushes the policy away from
    # that local optimum by making two-feet-airborne strictly negative
    # whenever a non-trivial linear command is active.
    rear_feet_flight = RewTerm(
        func=bipedal_mdp.feet_both_airborne,
        weight=-0.75,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[LR]_foot"),
        },
    )
    # EXPERIMENTAL (2026-04-18): swing-foot clearance reward. The
    # single-stance air-time reward only requires that one foot be in the
    # air, not that it meaningfully *lift* — the policy found a dragging
    # local optimum where the swing foot scrapes along just above the
    # ground, satisfying the contact sensor threshold while not clearing
    # the floor. This term (Spot-style) pays off for bringing any moving
    # foot to ``target_height=0.10`` m; stance feet do not contribute
    # because the ``tanh(|v_horiz|)`` gate zeroes them out. The exponential
    # kernel with ``std=0.05`` caps the per-step reward at ~1 when every
    # swinging foot is at the target and decays smoothly for lower
    # clearances, which gives a clean gradient out of the dragging regime.
    rear_feet_clearance = RewTerm(
        func=bipedal_mdp.feet_clearance_reward,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="R[LR]_foot"),
            "target_height": 0.10,
            "std": 0.05,
            "tanh_mult": 2.0,
            "command_name": "base_velocity",
        },
    )
    # EXPERIMENTAL (2026-04-18): in-contact foot-slide penalty. Directly
    # punishes horizontal foot velocity while the foot is loaded, which is
    # the other half of the dragging behaviour the clearance reward cannot
    # see (dragging == foot is in contact AND moving). A small negative
    # weight is enough because the reward is summed across feet and grows
    # linearly with slide speed; this should not discourage the small,
    # short-duration slip that happens at every heel-strike.
    rear_feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="R[LR]_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names="R[LR]_foot"),
        },
    )

    # -- contact penalties --
    #
    # ``thigh_contact`` / ``calf_contact`` / ``head_contact`` penalise
    # non-foot body parts touching the ground. ``front_foot_contact``
    # forbids the front feet from supporting body weight — the direct
    # anti-cheat for the "4-foot splay" pose where the policy tilts the
    # body ~90 deg and stands on all four feet.
    #
    # EXPERIMENTAL (2026-04-18): softened ``calf_contact`` (-3.0 -> -1.0)
    # and ``front_foot_contact`` (-1.5 -> -0.5). ``undesired_contacts``
    # returns the *number* of bodies in contact above threshold, so at a
    # flat quadruped spawn the policy starts each episode at
    # ``2 * -1.5 = -3`` front-foot penalty plus any calf brush, against a
    # max orientation-align reward of ``+2``. That reward landscape pulls
    # the policy toward the fastest possible front-foot lift-off and
    # learns a "rock-backward-hard" stand-up that barely damps out in
    # PhysX and diverges into a backward flip under Mujoco's contact
    # model. The softer weights keep the anti-cheat direction but stop
    # making it the dominant gradient at spawn, so the policy can afford
    # a controlled stand-up instead of an overshoot.
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
        weight=-1.0,
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
    ``limit_ranges.*`` whenever the reward exceeds a configurable
    fraction of the term's weight. Linear tracking uses the default
    ``0.8 * weight`` fraction; yaw tracking uses ``0.6 * weight`` to
    account for the intrinsic wobble floor of the bipedal stance.
    """

    lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    # EXPERIMENTAL (2026-04-18): yaw curriculum threshold dropped from the
    # default 0.8 * weight to 0.6 * weight. With the tighter ``std=sqrt(1.0)``
    # kernel the intrinsic bipedal yaw-wobble (~0.5-0.8 rad/s) limits the
    # achievable mean yaw-tracking reward to roughly 0.6-0.7 even for a
    # policy that correctly tracks the commanded component; leaving the
    # threshold at 0.8 would permanently stall the curriculum at its initial
    # range. 0.6 lets the curriculum advance once the policy is tracking
    # meaningfully above the wobble floor.
    ang_vel_cmd_levels = CurrTerm(
        func=mdp.ang_vel_cmd_levels,
        params={"reward_threshold_fraction": 0.6},
    )


# ---------------------------------------------------------------------------
# Physics presets
# ---------------------------------------------------------------------------


@configclass
class BipedalFlatPhysicsCfg(PresetCfg):
    """Physics backend presets for bipedal walking on flat terrain.

    Use ``presets=newton_mjwarp`` (or ``presets=newton``) CLI override to
    select Newton (MuJoCo Warp) backend for sim2sim transfer.

    Bipedal Go2 needs much higher njmax than the four-legged default (65)
    because the standing posture creates many more contacts.
    ``njmax=400``, ``nconmax=200`` handles peaks up to ~366 seen in practice.
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
class BipedalRoughPhysicsCfg(PresetCfg):
    """Physics backend presets for bipedal walking on rough terrain.

    Use ``presets=newton_mjwarp`` (or ``presets=newton``) CLI override to
    select Newton (MuJoCo Warp) backend for sim2sim transfer.

    Matches the stock IsaacLab rough-terrain locomotion setup: Newton's
    own collision pipeline (``use_mujoco_contacts=False``) with a 1 cm
    shape margin for stable mesh contact.  ``njmax`` / ``nconmax`` are
    raised relative to the flat preset because the bipedal stance creates
    more contacts on triangle-mesh terrain.
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
        self.sim.physics = BipedalFlatPhysicsCfg()
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotBipedalWalkPlayEnvCfg(RobotBipedalWalkEnvCfg):
    """Eval / playback variant: fewer envs, no observation noise, no DR, no pushes.

    Sets ``ranges = limit_ranges`` so the user can issue full-range
    commands to the trained policy immediately — the curriculum is left
    in place but becomes a no-op once the ranges already equal the
    limits.

    All domain-randomization events enabled in training are disabled here
    so the evaluation runs see deterministic nominal dynamics, which is
    the correct reference point for inspecting policy behaviour and for
    eyeballing sim-to-sim transfer before exporting to Mujoco.
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        # EXPERIMENTAL (2026-04-18): turn off every DR term for playback.
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.add_rear_leg_mass = None
        self.events.randomize_base_com = None
        self.events.randomize_actuator_gains = None
        self.events.push_robot = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotBipedalWalkRoughEnvCfg(RobotBipedalWalkEnvCfg):
    """Bipedal walking on rough terrain variant — adds terrain robustness.
    
    Enables procedurally-generated rough terrain (random rough, pyramid slopes,
    boxes, flat) and expanded initial spawn tilt. Everything else inherits from
    the base flat-terrain bipedal config.
    """

    def __post_init__(self):
        super().__post_init__()
        
        # -- physics: use rough-terrain preset (higher njmax, shape margin)
        self.sim.physics = BipedalRoughPhysicsCfg()
        
        # -- terrain: swap flat plane for procedural rough terrain
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = GO2_BIPEDAL_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = 0
        
        # -- initial spawn tilt: expanded from ±0.05 to ±0.14 rad
        self.events.reset_base.params["pose_range"]["roll"] = (-0.14, 0.14)
        self.events.reset_base.params["pose_range"]["pitch"] = (-0.14, 0.14)
        
        # -- tighter reset tolerances for rough terrain
        # Reset before the robot reaches extreme contact states that
        # cause solver divergence (NaN).  On flat terrain these are
        # intentionally absent to allow exploration, but on rough
        # terrain the triangle-mesh contacts explode when the body
        # slams into the ground.
        self.terminations.base_too_low = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.18},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=bipedal_mdp.bad_bipedal_orientation,
            params={
                "desired_gravity": DESIRED_GRAVITY_REAR,
                "max_angle_to_target": 1.6,
                "falling_vel_threshold": 0.5,
            },
        )
        
        # -- terrain curriculum: disabled for consistent training difficulty
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotBipedalWalkRoughPlayEnvCfg(RobotBipedalWalkRoughEnvCfg):
    """Rough terrain play/eval variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.observations.policy.enable_corruption = False
        # Turn off all DR for playback
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.add_rear_leg_mass = None
        self.events.randomize_base_com = None
        self.events.randomize_actuator_gains = None
        self.events.push_robot = None
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        # Camera tracking the robot for video recording
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.55)
        self.viewer.origin_type = "asset_root"
        self.viewer.env_index = 0
        self.viewer.asset_name = "robot"



