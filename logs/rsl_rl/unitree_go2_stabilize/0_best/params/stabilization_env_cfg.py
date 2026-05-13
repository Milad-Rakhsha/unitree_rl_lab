"""Go2 stabilization / fall-recovery environment.

Trains a policy to recover to a calm four-legged standing pose from
structured falling scenarios:

* Bipedal forward / backward / sideways falls (from rear-stance)
* Quadruped stumbles (large perturbation from four-legged stance)
* Uniform random (broad coverage for edge cases)

The reward is simple: be upright, reach default pose, stay calm.
The complexity lives in the *reset distribution*, not the reward.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_newton.physics import NewtonCfg, MJWarpSolverCfg
from isaaclab_tasks.utils.hydra import PresetCfg
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
from unitree_rl_lab.tasks.locomotion.robots.go2 import stabilization_mdp

# Flat terrain only — stabilization target is nominal plane contact.
# NOTE: terrain_generator is NOT used for Newton — the MuJoCo solver
# rejects the generated mesh patches ("mesh volume is too small").
# Use terrain_type="plane" instead (see RobotSceneCfg below).


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Flat ground + Go2 + contact sensing (no curriculum terrains)."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
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

    height_scanner = None  # Not needed for flat-terrain stabilization
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
# Events (domain randomization + structured fall resets)
# ---------------------------------------------------------------------------


@configclass
class EventCfg:
    """Scenario-based fall resets + startup DR.

    The single :attr:`reset_fall` event replaces both ``reset_base`` and
    ``reset_robot_joints`` from the original config — it sets the full
    (root_pose, root_vel, joint_pos, joint_vel) state from structured
    falling scenarios.
    """

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.5, 3.5),
            "operation": "add",
        },
    )

    # Occasional impulse at reset (on top of scenario velocities).
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (-30.0, 30.0),
            "torque_range": (-6.0, 6.0),
        },
    )

    # Structured fall scenario reset (replaces reset_base + reset_robot_joints)
    reset_fall = EventTerm(
        func=stabilization_mdp.reset_fall_scenarios,
        mode="reset",
        params={
            "bipedal_forward_frac": 0.25,
            "bipedal_backward_frac": 0.25,
            "bipedal_sideways_frac": 0.20,
            "quadruped_stumble_frac": 0.20,
            # remaining 0.10 → uniform random
            "joint_pos_noise": 0.15,
            "joint_vel_noise": 3.0,
        },
    )


@configclass
class CommandsCfg:
    """Zero nominal velocity command (policy only stabilizes, does not track motion)."""

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


@configclass
class ActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True, clip={".*": (-100.0, 100.0)}
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


@configclass
class ObservationsCfg:
    """Proprioceptive observations — no velocity command (pure stabilization)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
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
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, clip=(-100, 100))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, clip=(-100, 100))
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01, clip=(-100, 100))
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Rewards
# ---------------------------------------------------------------------------


@configclass
class RewardsCfg:
    """Encourage upright calm standing: orientation, default pose, low velocities, smooth actions.

    The reward is intentionally simple — the learning signal comes from
    the structured reset distribution, not a complex reward shape.
    """

    is_alive = RewTerm(func=mdp.is_alive, weight=0.2)

    upright = RewTerm(func=mdp.upright_gravity_exp, weight=2.0, params={"std": 0.08})
    default_pose = RewTerm(func=mdp.default_joint_pose_exp, weight=1.25, params={"std": 1.2})

    root_vel_penalty = RewTerm(func=mdp.root_body_velocity_l2, weight=-0.35, params={"ang_weight": 0.25})
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.0025)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.12)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.5e-4)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-8.0)
    energy = RewTerm(func=mdp.energy, weight=-1.5e-5)

    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)

    # Penalize body/head/hip/thigh/calf contacts — these should NOT touch the ground.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip", ".*_thigh", ".*_calf"]),
        },
    )
    head_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*"]),
        },
    )


# ---------------------------------------------------------------------------
# Terminations
# ---------------------------------------------------------------------------


@configclass
class TerminationsCfg:
    """Terminate on base or head ground contact.

    ``bad_orientation`` is intentionally removed — the robot spawns in
    heavily tilted states and must be allowed to experience large angles
    to learn recovery. The base/head contact termination is the safety
    boundary: if the body or head hits the ground, the episode is over.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Base (trunk) touches the ground → terminal
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )

    # Head touches the ground → terminal
    head_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_.*"]),
            "threshold": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# Physics presets
# ---------------------------------------------------------------------------


@configclass
class StabilizePhysicsCfg(PresetCfg):
    """Physics backend presets for stabilization training.

    Use ``presets=newton`` or ``presets=newton_mjwarp`` CLI override for Newton.
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


# ---------------------------------------------------------------------------
# Env configs
# ---------------------------------------------------------------------------


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics = StabilizePhysicsCfg()

        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
