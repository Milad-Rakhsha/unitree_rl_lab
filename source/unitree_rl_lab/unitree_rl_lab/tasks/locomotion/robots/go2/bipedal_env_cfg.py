"""Go2 bipedal (two-leg) walking environment.

Trains the Go2 to stand and locomote on **two** legs. Two variants are
provided so the user can compare which is easier to learn:

* :class:`RobotBipedalRearEnvCfg`  — stands on the hind legs (body pitched
  nose-up, "begging" pose). Task ID: ``Unitree-Go2-Bipedal-Rear``.
* :class:`RobotBipedalFrontEnvCfg` — stands on the front legs (body pitched
  nose-down, "handstand" pose). Task ID: ``Unitree-Go2-Bipedal-Front``.

Infinite-plane variants (``*-Flat``) are also provided for faster iteration.

Key differences from the four-legged velocity env:

1. The body is expected to be rotated ~pi/2 about the pitch axis; the
   orientation target is expressed via ``desired_gravity`` in the body
   frame instead of the usual "flat ground" assumption.
2. The two "swing" legs (non-stance) are penalized for ground contact and
   encouraged to stay near a tucked pose; the two "stance" legs drive
   locomotion.
3. The base is initialized with the target pitch and an elevated z so the
   policy starts near the bipedal manifold and can focus on balance /
   walking rather than first having to rise from a crouch.
"""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
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
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

from . import bipedal_mdp

# -- Stance presets --------------------------------------------------------
#
# projected_gravity_b of the body in the target stance pose (unit vector).
#   rear stance  : body pitched ~+pi/2 about body-y → world-down maps to -body_x
#   front stance : body pitched ~-pi/2 about body-y → world-down maps to +body_x
DESIRED_GRAVITY_REAR = [-1.0, 0.0, 0.0]
DESIRED_GRAVITY_FRONT = [1.0, 0.0, 0.0]

# Pitch applied to the default (flat) base orientation at reset.
#
# IsaacLab's ``quat_from_euler_xyz(roll, pitch, yaw)`` uses intrinsic XYZ
# rotations. For pitch = +pi/2 (rotation about +y, right-hand rule), the
# body-x axis (nose/forward) rotates toward -world_z (i.e. points down) —
# that's the **nose-down** / handstand pose. For pitch = -pi/2, nose points
# +world_z (up) — that's the **nose-up** / begging pose.
#
# So the sign is the opposite of what the stance name might suggest:
#   rear stance  (nose up, hind legs on ground)  = pitch -pi/2
#   front stance (nose down, front legs on ground) = pitch +pi/2
INIT_PITCH_REAR = -math.pi / 2
INIT_PITCH_FRONT = math.pi / 2

# World-frame base z target when fully upright on two legs (Go2 specific).
BIPEDAL_TARGET_BASE_Z = 0.50

# Joint regex groups on the Go2.
FRONT_FOOT_NAMES = ["F[LR]_foot"]
REAR_FOOT_NAMES = ["R[LR]_foot"]
FRONT_LEG_JOINTS = ["F[LR]_hip_joint", "F[LR]_thigh_joint", "F[LR]_calf_joint"]
REAR_LEG_JOINTS = ["R[LR]_hip_joint", "R[LR]_thigh_joint", "R[LR]_calf_joint"]

# Rough bipedal joint targets used at reset: {pattern: (center, halfwidth)} in rad.
# These place the robot close to the target bipedal pose so that learning
# focuses on walking rather than the hard "how do I stand up?" problem.
#
# Notes on Go2 joint ranges (soft limits):
#   thigh: ~[-0.686, 4.501]  — positive swings the knee forward.
#   calf:  ~[-2.818, -0.888] — calf is *always* bent (never below -0.888).
# Targets outside these ranges are silently clamped at reset, so we keep them
# strictly inside the limits.
REAR_STANCE_JOINT_TARGETS: dict[str, tuple[float, float]] = {
    # hind legs (stance): nearly straight — as close to -0.888 as the soft
    # limit allows, with the thigh swung forward to support the pitched-up body.
    "R[LR]_hip_joint": (0.0, 0.1),
    "R[LR]_thigh_joint": (1.2, 0.15),
    "R[LR]_calf_joint": (-1.0, 0.1),
    # front legs (swing): curled toward the chest.
    "F[LR]_hip_joint": (0.0, 0.1),
    "F[LR]_thigh_joint": (1.3, 0.2),
    "F[LR]_calf_joint": (-2.4, 0.15),
}
FRONT_STANCE_JOINT_TARGETS: dict[str, tuple[float, float]] = {
    # front legs (stance): nearly straight to support the pitched-down body.
    "F[LR]_hip_joint": (0.0, 0.1),
    "F[LR]_thigh_joint": (1.2, 0.15),
    "F[LR]_calf_joint": (-1.0, 0.1),
    # hind legs (swing): tucked toward the belly.
    "R[LR]_hip_joint": (0.0, 0.1),
    "R[LR]_thigh_joint": (1.4, 0.2),
    "R[LR]_calf_joint": (-2.4, 0.15),
}

# Per-joint targets that the "swing" (non-stance) legs are softly pulled
# toward by the ``swing_leg_deviation`` reward. These are the *center* values
# from ``*_STANCE_JOINT_TARGETS`` above — we penalize deviation from the tuck
# pose, not from the flat-stance ``default_joint_pos`` (which would actively
# fight our desired bipedal posture).
REAR_SWING_JOINT_TARGETS: dict[str, float] = {
    "F[LR]_hip_joint": 0.0,
    "F[LR]_thigh_joint": 1.3,
    "F[LR]_calf_joint": -2.4,
}
FRONT_SWING_JOINT_TARGETS: dict[str, float] = {
    "R[LR]_hip_joint": 0.0,
    "R[LR]_thigh_joint": 1.4,
    "R[LR]_calf_joint": -2.4,
}


# Flat-only bipedal terrain: learning to balance on two legs is hard enough
# without simultaneously dealing with slopes and random rough ground.
BIPEDAL_FLAT_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=5,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 0.0),
    use_cache=False,
    sub_terrains={"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)},
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat ground + Go2 + contact sensing (no curriculum terrains)."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=BIPEDAL_FLAT_TERRAIN_CFG,
        max_init_terrain_level=0,
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
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Domain randomization + stance-aware reset.

    Defaults to rear-stance; the ``pose_range.pitch`` and
    ``reset_robot_joints.params.joint_targets`` fields are overridden in
    :meth:`RobotBipedalEnvCfg.__post_init__` when the front stance is
    selected.
    """

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.7, 1.8),
            "dynamic_friction_range": (0.6, 1.6),
            "restitution_range": (0.0, 0.15),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-0.8, 2.0),
            "operation": "add",
        },
    )

    # Initialize the base already pitched to the bipedal pose. The +z offset
    # elevates the body so it doesn't clip into the ground while tilted.
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "z": (0.10, 0.15),
                "yaw": (-3.14, 3.14),
                "roll": (-0.08, 0.08),
                "pitch": (INIT_PITCH_REAR - 0.15, INIT_PITCH_REAR + 0.15),
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
        func=bipedal_mdp.reset_joints_to_target,
        mode="reset",
        params={
            "joint_targets": REAR_STANCE_JOINT_TARGETS,
            "velocity_range": (-0.5, 0.5),
        },
    )

    # Small periodic shoves help with balance robustness.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 12.0),
        params={"velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
    )


@configclass
class CommandsCfg:
    """Forward / angular velocity command.

    Defaults are narrow (slow walk); the ``lin_vel_cmd_levels`` curriculum
    expands them as the agent improves, up to ``limit_ranges``.
    """

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.3,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.05, 0.05), lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.3, 0.3)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 0.5), lin_vel_y=(-0.15, 0.15), ang_vel_z=(-0.8, 0.8)
        ),
    )


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


@configclass
class ObservationsCfg:
    """Observations for the bipedal task.

    No height-scanner / terrain-height observations are used; the policy
    runs on a flat plane. The target gravity is implicit per-environment
    (fixed by stance choice) so it does not need to appear in the observation.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100), noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100), noise=Unoise(n_min=-1.5, n_max=1.5)
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

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Bipedal reward shaping (rear-stance defaults).

    Stance-dependent fields are patched in :meth:`RobotBipedalEnvCfg.__post_init__`
    so that the same cfg class serves both stance variants:

    * ``target_orientation.params["desired_gravity"]``
    * ``feet_air_time.params["sensor_cfg"].body_names``
    * ``air_time_variance.params["sensor_cfg"].body_names``
    * ``feet_slide.params["asset_cfg"|"sensor_cfg"].body_names``
    * ``swing_feet_contact.params["sensor_cfg"].body_names``
    * ``swing_leg_deviation.params["joint_targets"]``

    Positive shaping: stay alive, match the target orientation, maintain a
    target base height, track the commanded velocity, and get air-time on
    the two stance feet.

    Negative shaping: keep the swing (non-stance) feet off the ground, keep
    the swing legs near their tucked reset pose, and apply the usual
    joint / action regularizers.
    """

    is_alive = RewTerm(func=mdp.is_alive, weight=0.25)
    # Tracking rewards are GATED on "is upright": they only pay out when the
    # body is close to the bipedal target stance (within ~45° of target, via
    # ``cos_threshold``). Without this gate the policy can collect tracking
    # reward in a partially-collapsed pose — a competing local optimum that
    # blocks bipedal emergence. Idea from Zhang et al. 2025, arXiv:2507.20382.
    track_lin_vel_xy = RewTerm(
        func=bipedal_mdp.track_lin_vel_xy_upright_exp,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "desired_gravity": DESIRED_GRAVITY_REAR,
            "cos_threshold": 0.7,
        },
    )
    track_ang_vel_z = RewTerm(
        func=bipedal_mdp.track_ang_vel_z_upright_exp,
        weight=0.75,
        params={
            "command_name": "base_velocity",
            "std": math.sqrt(0.25),
            "desired_gravity": DESIRED_GRAVITY_REAR,
            "cos_threshold": 0.7,
        },
    )

    target_orientation = RewTerm(
        func=bipedal_mdp.target_orientation_exp,
        weight=2.5,
        params={"std": 0.12, "desired_gravity": DESIRED_GRAVITY_REAR},
    )
    # std widened from 0.03 → 0.08: natural CoM oscillation during walking
    # easily exceeds 5 cm, and at 0.03 the reward falls off too fast to
    # tolerate stepping. Weight dropped slightly to avoid dominating the mix.
    base_height = RewTerm(
        func=bipedal_mdp.base_height_exp,
        weight=1.0,
        params={"target_height": BIPEDAL_TARGET_BASE_Z, "std": 0.08},
    )

    # Biped-specific single-stance reward: pays out the current contact/air
    # time (clipped at `threshold`) whenever exactly one of the two stance
    # feet is in contact and the commanded velocity is non-zero. This is the
    # key "go walk" signal — the quadruped ``feet_air_time`` (which only
    # fires on foot landing with air_time ≥ threshold) is too sparse for a
    # policy that hasn't learned a gait yet.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=REAR_FOOT_NAMES),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )
    # Softer than before (was -0.5): variance in per-foot air/contact times
    # is naturally high while the policy is *learning* to step, and a strong
    # penalty there preempts walking altogether.
    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-0.2,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=REAR_FOOT_NAMES)},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=REAR_FOOT_NAMES),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=REAR_FOOT_NAMES),
        },
    )

    # Discourage the swing (non-stance) feet from touching the floor.
    swing_feet_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FRONT_FOOT_NAMES),
        },
    )

    # Keep the swing legs near their *tucked* target pose (not the flat
    # quadruped default). See :func:`bipedal_mdp.joint_deviation_from_target_l1`
    # for why this matters.
    swing_leg_deviation = RewTerm(
        func=bipedal_mdp.joint_deviation_from_target_l1,
        weight=-0.1,
        params={"joint_targets": REAR_SWING_JOINT_TARGETS},
    )

    # -- regularizers --
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # Torque/energy penalties relaxed from the quadruped defaults: holding
    # the robot balanced on two legs with the mass over a small support
    # polygon naturally requires more torque than flat-stance walking.
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1e-4)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0)
    energy = RewTerm(func=mdp.energy, weight=-5e-6)

    # Only the head is truly "bad" contact in bipedal — thighs/hips are
    # allowed to graze the body (e.g. the swing-leg thighs curled toward
    # the chest sit very close to the base).
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.5,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Bipedal terminations (rear-stance defaults).

    ``bad_orientation.params["desired_gravity"]`` is patched for front stance.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # Terminate if the body orientation drifts far from the target stance.
    # limit_angle ~1.4 rad ≈ 80°: generous enough for the policy to explore
    # dynamic balancing without getting killed on every minor wobble.
    bad_orientation = DoneTerm(
        func=bipedal_mdp.bad_target_orientation,
        params={"desired_gravity": DESIRED_GRAVITY_REAR, "limit_angle": 1.4},
    )
    # Base z is measured in world frame. On a pitched body the CoM can dip
    # fairly low during the push-off phase; keep this lenient.
    base_too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.15},
    )


@configclass
class CurriculumCfg:
    """Auto-expand the lin-vel command range as the agent learns to track it."""

    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


# ---------------------------------------------------------------------------
# Base env
# ---------------------------------------------------------------------------


@configclass
class RobotBipedalEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for two-legged Go2 locomotion.

    Subclasses select the stance by setting :attr:`stance` (``"rear"`` or
    ``"front"``). The base class defaults to rear stance; :meth:`__post_init__`
    patches the stance-dependent reward / termination / event parameters.
    """

    stance: str = "rear"

    # -- Recommendation-B knobs (optional) -------------------------------
    # ``flat_init`` spawns the robot on four legs in the default quadruped
    # pose and scales pushes up so the policy has to learn the full
    # "stand up, then walk" trajectory. Use together with
    # ``only_positive_rewards`` — on its own, tracking/height/orientation
    # rewards are all zero at spawn so the agent has nothing to climb.
    #
    # ``only_positive_rewards`` clips the per-step *total* reward to ``>=0``
    # (Legged-Gym trick). Requires the env class to be
    # :class:`PositiveRewardManagerBasedRLEnv` (enforced via the
    # ``*FlatInit*`` gym registrations).
    flat_init: bool = False
    only_positive_rewards: bool = False

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        if self.stance not in ("rear", "front"):
            raise ValueError(f"stance must be 'rear' or 'front', got {self.stance!r}")
        if self.stance == "front":
            self._apply_front_stance_overrides()
        if self.flat_init:
            self._apply_flat_init_overrides()

        # general settings
        self.decimation = 4
        # Flat-init gives the policy more time to stand up *and* walk before
        # the time-out fires; the pre-pitched base variants finish sooner.
        self.episode_length_s = 22.0 if self.flat_init else 15.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # update sensor update periods
        self.scene.contact_forces.update_period = self.sim.dt

        # flat terrain → no terrain curriculum
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False

    def _apply_front_stance_overrides(self) -> None:
        """Flip all stance-dependent parameters from rear → front."""
        # Events: init pitch + reset joint targets
        self.events.reset_base.params["pose_range"]["pitch"] = (
            INIT_PITCH_FRONT - 0.15,
            INIT_PITCH_FRONT + 0.15,
        )
        self.events.reset_robot_joints.params["joint_targets"] = FRONT_STANCE_JOINT_TARGETS

        # Rewards: desired gravity + stance/swing foot groups
        self.rewards.target_orientation.params["desired_gravity"] = DESIRED_GRAVITY_FRONT
        self.rewards.track_lin_vel_xy.params["desired_gravity"] = DESIRED_GRAVITY_FRONT
        self.rewards.track_ang_vel_z.params["desired_gravity"] = DESIRED_GRAVITY_FRONT
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = FRONT_FOOT_NAMES
        self.rewards.air_time_variance.params["sensor_cfg"].body_names = FRONT_FOOT_NAMES
        self.rewards.feet_slide.params["asset_cfg"].body_names = FRONT_FOOT_NAMES
        self.rewards.feet_slide.params["sensor_cfg"].body_names = FRONT_FOOT_NAMES
        self.rewards.swing_feet_contact.params["sensor_cfg"].body_names = REAR_FOOT_NAMES
        self.rewards.swing_leg_deviation.params["joint_targets"] = FRONT_SWING_JOINT_TARGETS

        # Terminations: desired gravity for the bad-orientation check
        self.terminations.bad_orientation.params["desired_gravity"] = DESIRED_GRAVITY_FRONT

    def _apply_flat_init_overrides(self) -> None:
        """Port arclab-hku's Go1 bipedal recipe to our Go2 env (Rec B).

        Near-literal port of
        ``legged_gym/envs/bipedal_dog/bipedal_dog_config_baseline.py`` from
        ``arclab-hku/bipedal_locomotion_for_quadrupedal_robots``. Key
        differences vs. the default (Rec-A) bipedal cfg:

        1. **Monotonic orientation reward** — ``max(0, proj_grav · target)``
           with *positive* weight (0.8). Replaces the narrow Gaussian
           ``target_orientation_exp`` which is numerically zero at the flat
           spawn and provides no gradient. Any tilt toward the target
           stance now pays out.
        2. **Quadratic base-height penalty** (weight -0.5) instead of the
           Gaussian reward — quadratic has nonzero gradient everywhere,
           the Gaussian plateaus far from target.
        3. **No gait shaping.** ``feet_air_time``, ``swing_feet_contact``,
           ``swing_leg_deviation``, ``feet_slide``, ``air_time_variance``
           all zeroed. Velocity tracking alone drives locomotion.
        4. **Strong vertical-velocity damping** (``lin_vel_z``, weight -2.0)
           — new term, prevents bouncing/oscillation modes.
        5. **Motion penalties against ``default_joint_pos``** (hip/thigh/
           calf groups) — soft L1 pull toward the flat quadruped pose.
        6. **Full command range from iteration 0** (no curriculum).
           ``lin_vel_x ∈ [-1, 1]``, ``lin_vel_y ∈ [-0.5, 0.5]``,
           ``ang_vel_z ∈ [-1, 1]``.
        7. **Plain (un-gated) tracking rewards.** With the monotonic
           orientation reward providing the gradient, Rec-A's upright gate
           is counterproductive (masks tracking gradient exactly when the
           agent needs it to commit to walking).
        8. **Other regularizers zeroed** to match the reference.
        9. **Flat quadruped reset**, 1 m/s pushes, ``bad_orientation``
           loosened to 2.8 rad (only full flip-overs terminate).

        Used together with :attr:`only_positive_rewards` (set True by the
        ``*FlatInit*`` variants) — this matches the reference recipe.
        """
        # -- Reset: flat quadruped spawn --------------------------------
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.3, 0.3),
            "y": (-0.3, 0.3),
            "yaw": (-3.14, 3.14),
            "roll": (-0.05, 0.05),
            "pitch": (-0.05, 0.05),
        }
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={"position_range": (1.0, 1.0), "velocity_range": (-1.0, 1.0)},
        )
        if self.events.push_robot is not None:
            self.events.push_robot.params["velocity_range"] = {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
            }
            self.events.push_robot.interval_range_s = (10.0, 15.0)

        # -- Terminations: only terminate on true flip-over / base contact --
        #
        # The default bipedal terminations were tuned for a pre-pitched spawn;
        # from a flat quadruped spawn they fire *instantly*:
        #
        # * ``base_too_low`` (0.15 m): the untrained policy flails, legs
        #   collapse, base drops below 0.15 m → episode dies in ~1 s. This
        #   is the main reason Rec-B was plateauing at ~30 total reward
        #   (very short episodes × tiny per-step reward).
        # * ``bad_orientation`` at 1.4 rad: flat spawn is ~1.57 rad off the
        #   bipedal target by design, so this would fire on every env at
        #   step 0. Loosened to 2.8 rad.
        #
        # The arclab-hku Go1 recipe only terminates on ``base/trunk/hip``
        # contact (``base_contact`` below) plus a generous orientation gate.
        # We match that: drop ``base_too_low`` entirely for FlatInit.
        self.terminations.bad_orientation.params["limit_angle"] = 2.8
        self.terminations.base_too_low = None

        # Also drop ``base_contact``. Previous run showed the policy
        # retreating from dot>0.78 specifically because 9% of its
        # pitch-up attempts (around iter 150) caused a brush of the
        # base on the ground → instant termination → loss of the
        # remaining ~1000 steps of orientation reward. That "fall =
        # game over" signal is exactly what was preventing further
        # exploration. Without early termination the policy can fall,
        # get up, and learn to recover — while still being strongly
        # incentivized to stay upright (lying on the back gives 0
        # target_orientation, vs ~3.0/step when standing).
        self.terminations.base_contact = None

        # -- Commands: moderate start ------------------------------------
        #
        # Previous run narrowed to ±0.3 m/s and left the
        # ``lin_vel_cmd_levels`` curriculum active, expecting it to
        # widen the range automatically. It didn't — curriculum level
        # stayed at 0.3 for all 398 iters while ``error_vel_xy``
        # dropped to 0.37 (well below the expand threshold). Setting
        # initial ranges directly to a level the robot can track is
        # simpler and more reliable; the curriculum still sits on top
        # and can push it up further if the tracking reward exceeds
        # ``weight * 0.8``.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.limit_ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.limit_ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.limit_ranges.ang_vel_z = (-1.0, 1.0)

        # -- Rewards: arclab-hku port ---------------------------------
        # Drop is_alive — balance/height terms already reward surviving.
        self.rewards.is_alive.weight = 0.0

        # Orientation-**gated** tracking rewards. The gate is a soft
        # multiplicative factor ``max(0, dot(proj_grav_b, target))`` that
        # varies smoothly from 0 (flat quadruped) to 1 (target stance).
        # Without this, our earlier FlatInit runs converged to a leaned-
        # quadruped local optimum (dot ≈ 0.4, near-max pose-independent
        # tracking) — see bipedal_mdp.track_lin_vel_xy_orient_gated_exp
        # docstring for details. With the gate, tracking reward is
        # *strictly* unavailable until the body pitches up.
        target_gravity = DESIRED_GRAVITY_FRONT if self.stance == "front" else DESIRED_GRAVITY_REAR
        self.rewards.track_lin_vel_xy = RewTerm(
            func=bipedal_mdp.track_lin_vel_xy_orient_gated_exp,
            weight=1.5,
            params={
                "command_name": "base_velocity",
                "std": math.sqrt(0.25),
                "desired_gravity": target_gravity,
            },
        )
        self.rewards.track_ang_vel_z = RewTerm(
            func=bipedal_mdp.track_ang_vel_z_orient_gated_exp,
            weight=0.75,
            params={
                "command_name": "base_velocity",
                "std": math.sqrt(0.25),
                "desired_gravity": target_gravity,
            },
        )
        # Monotonic "tilt toward target" reward, provides gradient
        # everywhere on the tilt manifold (including at flat, where the
        # gated tracking rewards are zero and would otherwise leave the
        # agent blind).
        #
        # Weight bumped from 2.0 → 3.0 after the previous run plateaued
        # at dot≈0.775 (~50° pitch). Motion-penalty growth per unit of
        # extra pitch was eating nearly all of the marginal orientation
        # gain; a stronger orientation weight gives the policy a
        # clearer reason to push past the half-bipedal local optimum.
        self.rewards.target_orientation = RewTerm(
            func=bipedal_mdp.orientation_align,
            weight=3.0,
            params={"desired_gravity": target_gravity},
        )
        # Sharp commitment bonus for the final approach. Kept in place
        # as a nudge near the target, but the main driver for standing
        # upright now is the ``swing_feet_contact`` penalty, which
        # physically forces front-foot clearance (see gait-shaping
        # section below).
        self.rewards.target_orientation_commit = RewTerm(
            func=bipedal_mdp.orientation_commit,
            weight=5.0,
            params={"desired_gravity": target_gravity, "threshold": 0.7},
        )
        # Quadratic base-height penalty.
        self.rewards.base_height = RewTerm(
            func=bipedal_mdp.base_height_l2,
            weight=-0.5,
            params={"target_height": BIPEDAL_TARGET_BASE_Z},
        )

        # Stability / regularization
        self.rewards.lin_vel_z = RewTerm(func=bipedal_mdp.lin_vel_z_l2, weight=-2.0)
        self.rewards.base_angular_velocity.weight = -0.05
        # Action-rate penalty reduced 4× after the previous run
        # plateaued at dot=0.80. The policy had learned very smooth
        # control (action_rate down from -0.27 at iter 50 to -0.12 at
        # iter 775), but bipedal balance on point feet needs fast
        # corrective control. Lowering the weight from -0.02 to -0.005
        # lets the policy use quicker actions to stabilize near
        # vertical without paying a crushing penalty.
        self.rewards.action_rate.weight = -0.005
        # Keep ``undesired_contacts`` on the head only. Broadening to
        # thighs/calves would fire every time a swing leg grazes the
        # ground during the stand-up transient; with
        # ``only_positive_rewards`` clipping the total, that wipes out
        # the orientation reward earned on the same step → the agent
        # sees zero gradient on exactly the behavior we want to
        # reinforce. Base/hip collision is already caught by the
        # ``base_contact`` termination.
        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces", body_names=["Head_.*"]
        )

        # Gait-shaping terms — re-enabled here to PHYSICALLY FORCE the
        # policy onto two legs. Previous runs plateaued at dot=0.80
        # because the policy found a pitched-quadruped "cheat": body
        # tilted 37° forward with all four feet still on the ground
        # (base_z ≈ 0.26 m, i.e. quadruped height, not standing
        # height). Orientation reward paid the tilt, tracking reward
        # paid the quadruped walk; no shaping term demanded that the
        # swing (front-for-rear, rear-for-front) feet actually leave
        # the ground.
        #
        # ``swing_feet_contact`` (weight -3.0) fires for every swing
        # foot in contact with the ground above 1 N. With two front
        # feet planted this is -6/step, easily enough to make
        # pitched-quadruped walking a net-negative strategy. Requires
        # ``only_positive_rewards = False`` (turned off in the
        # RobotBipedalRearFlatInitEnvCfg below) — otherwise the
        # negative total gets clipped to 0 and the gradient
        # distinguishing "cheat" from "idle" disappears.
        #
        # ``feet_air_time`` (weight +2.0) rewards stance feet swinging
        # through the air — once bipedal, this pays for an actual
        # walking gait.
        self.rewards.feet_air_time.weight = 2.0
        # Air-time threshold lowered 0.3 → 0.15 s. At 0.3 s (quadruped
        # trot calibration) the reward only pays for long, deliberate
        # swings; a bipedal Go2 has to balance and step rapidly, with
        # per-foot air times on the order of 0.1–0.2 s, so almost none
        # of those swings cleared the threshold — the reward
        # flatlined at +0.33/step regardless of how well the robot
        # stepped. At 0.15 s the quick balance-corrective steps now
        # register as walking, and the policy has a reason to shorten
        # and clarify them further.
        self.rewards.feet_air_time.params["threshold"] = 0.15
        self.rewards.swing_feet_contact.weight = -3.0
        self.rewards.air_time_variance.weight = 0.0
        self.rewards.feet_slide.weight = 0.0
        self.rewards.swing_leg_deviation.weight = 0.0
        # Turn off torque/acc/energy/joint-vel regularizers.
        self.rewards.joint_vel.weight = 0.0
        self.rewards.joint_acc.weight = 0.0
        self.rewards.joint_torques.weight = 0.0
        self.rewards.energy.weight = 0.0

        # Per-joint-category L1 pull toward the BIPEDAL TARGET pose.
        #
        # Previous version used ``joint_deviation_from_default_l1`` which
        # pulls all joints back to the quadruped default. That's the
        # right target for arclab-hku's aliengo/go1 (their default joint
        # angles already encode a deep crouch that's close to the
        # bipedal pose), but for Go2 the default is the flat quadruped
        # stance: using it actively pulls the *swing* legs back down
        # into the quadruped pose, which directly fights the
        # orientation reward. The previous run plateaued at
        # ``dot=0.76`` (~50° pitch) exactly because these penalties
        # grew as the body pitched up further.
        #
        # Swap to ``joint_deviation_from_target_l1`` with the same
        # target pose the reset event uses (``REAR_STANCE_JOINT_TARGETS``
        # / ``FRONT_STANCE_JOINT_TARGETS``). Now the penalty is minimum
        # at the desired bipedal pose: the regularizer agrees with the
        # orientation reward instead of fighting it.
        stance_targets = (
            REAR_STANCE_JOINT_TARGETS if self.stance == "rear" else FRONT_STANCE_JOINT_TARGETS
        )
        hip_targets = {k: v[0] for k, v in stance_targets.items() if "hip" in k}
        thigh_targets = {k: v[0] for k, v in stance_targets.items() if "thigh" in k}
        calf_targets = {k: v[0] for k, v in stance_targets.items() if "calf" in k}

        # Weights zeroed after the previous run. The prescribed stance
        # target (rear thigh=1.2, calf=-1.0) encodes a *crouched*
        # bipedal pose, not the vertical standing pose we actually
        # want. Pulling the legs toward that target is an additional
        # force keeping the body at ~50° pitch: straightening the rear
        # legs to go fully vertical means moving thigh→~0.3 and
        # calf→~-0.9, which *increases* ``|q − q_target|`` and grows
        # the penalty.
        #
        # For now, let the policy find its own bipedal leg pose.
        # ``dof_pos_limits`` (-10) keeps joints off the hard stops,
        # ``action_rate`` (-0.02) prevents flailing, and
        # ``undesired_contacts`` penalizes head contact. Once we see
        # a clean bipedal gait we can reintroduce a tighter motion
        # regularizer around *that* pose.
        self.rewards.hip_motion = RewTerm(
            func=bipedal_mdp.joint_deviation_from_target_l1,
            weight=0.0,
            params={"joint_targets": hip_targets},
        )
        self.rewards.thigh_motion = RewTerm(
            func=bipedal_mdp.joint_deviation_from_target_l1,
            weight=0.0,
            params={"joint_targets": thigh_targets},
        )
        self.rewards.calf_motion = RewTerm(
            func=bipedal_mdp.joint_deviation_from_target_l1,
            weight=0.0,
            params={"joint_targets": calf_targets},
        )


# ---------------------------------------------------------------------------
# Concrete variants
# ---------------------------------------------------------------------------


@configclass
class RobotBipedalRearEnvCfg(RobotBipedalEnvCfg):
    """Two-legged Go2 standing / walking on the hind legs (body pitched nose-up)."""

    stance: str = "rear"


@configclass
class RobotBipedalFrontEnvCfg(RobotBipedalEnvCfg):
    """Two-legged Go2 standing / walking on the front legs (body pitched nose-down)."""

    stance: str = "front"


# --- Infinite-plane variants (no terrain generator) -- fast iteration. -----


@configclass
class RobotBipedalRearFlatEnvCfg(RobotBipedalRearEnvCfg):
    """Rear-stance variant on an infinite plane (no terrain generator)."""

    def __post_init__(self):
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        super().__post_init__()


@configclass
class RobotBipedalFrontFlatEnvCfg(RobotBipedalFrontEnvCfg):
    """Front-stance variant on an infinite plane (no terrain generator)."""

    def __post_init__(self):
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        super().__post_init__()


# --- Play / eval configs (fewer envs, no corruption, no pushes). -----------


def _play_overrides(self: RobotBipedalEnvCfg) -> None:
    self.scene.num_envs = 32
    self.observations.policy.enable_corruption = False
    self.events.push_robot = None
    self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges


@configclass
class RobotBipedalRearPlayEnvCfg(RobotBipedalRearEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play_overrides(self)


@configclass
class RobotBipedalFrontPlayEnvCfg(RobotBipedalFrontEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play_overrides(self)


@configclass
class RobotBipedalRearFlatPlayEnvCfg(RobotBipedalRearFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play_overrides(self)


@configclass
class RobotBipedalFrontFlatPlayEnvCfg(RobotBipedalFrontFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play_overrides(self)


# --- Recommendation-B variants: flat spawn + positive-only rewards. --------
#
# These share the same reward shaping / upright gating as the base variants,
# but (a) spawn the robot on four legs flat, (b) apply 1 m/s pushes, and
# (c) require the ``PositiveRewardManagerBasedRLEnv`` entry point so the
# per-step total reward is clipped to >= 0. Use when the base variants
# plateau because the policy has already found a "cheat" local optimum.


@configclass
class RobotBipedalRearFlatInitEnvCfg(RobotBipedalRearFlatEnvCfg):
    flat_init: bool = True
    # ``only_positive_rewards`` turned OFF. Previous runs with this clip
    # enabled produced a policy that walks as a 37°-pitched quadruped
    # (``base_z ≈ 0.26 m``, i.e. still quadruped height) with all four
    # feet on the ground — a "cheat" that collects orientation and
    # tracking reward without actually going bipedal. To break that
    # cheat we need to punish front-foot ground contact; with
    # ``only_positive_rewards`` active the total gets clipped to 0 and
    # the policy can't distinguish "cheating quadruped" from "nothing
    # happening". Letting rewards go negative restores the gradient
    # that disambiguates the two.
    only_positive_rewards: bool = False


@configclass
class RobotBipedalFrontFlatInitEnvCfg(RobotBipedalFrontFlatEnvCfg):
    flat_init: bool = True
    only_positive_rewards: bool = False


@configclass
class RobotBipedalRearFlatInitPlayEnvCfg(RobotBipedalRearFlatInitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play_overrides(self)


@configclass
class RobotBipedalFrontFlatInitPlayEnvCfg(RobotBipedalFrontFlatInitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play_overrides(self)
