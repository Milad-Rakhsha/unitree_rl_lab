"""Go2 *gallop / bound* velocity-tracking environment.

Encourages an asymmetric bounding gait where front and rear leg pairs
move in near-unison, with a flight phase between each pair's ground
contact — similar to a cheetah's transverse gallop or a greyhound's
double-suspension gallop.

Key differences from the sport (trot) env:
* **Gait reward**: ``feet_gait`` with bound offsets (front pair in phase,
  rear pair in phase, 180° between front/rear).
* **Flight bonus**: explicit reward for all-feet-airborne instants.
* **Higher velocity ceiling**: commands up to ±2.5 m/s forward.
* **Relaxed vertical & orientation penalties**: bounding naturally
  produces more pitch oscillation and vertical CoM motion.
* **Energy penalty reduced**: galloping is inherently higher-energy.

Foot body order (``.*_foot`` glob expansion):
  0 = FL_foot, 1 = FR_foot, 2 = RL_foot, 3 = RR_foot
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.terrains.height_field import HfRandomUniformTerrainCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import (
    RobotEnvCfg,
    RobotPlayEnvCfg,
    VelocityRoughPhysicsCfg,
)


@configclass
class RobotGallopEnvCfg(RobotEnvCfg):
    """Go2 gallop/bound env on flat terrain — fast asymmetric gait."""

    def __post_init__(self):
        super().__post_init__()

        # -- commands: high-speed forward, moderate lateral/yaw.
        #    Curriculum starts conservative and ramps up to ±2.5 m/s.
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.5), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.3, 0.3)
        )
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 2.5), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.8, 0.8)
        )

        # -- rewards: reshape for bounding gait --

        # 1) Gait pattern: BOUND
        #    Front pair (FL=0, FR=1) in phase at offset 0.0
        #    Rear pair  (RL=2, RR=3) in phase at offset 0.5
        #    Period 0.3s → ~3.3 strides/sec (typical quadruped bound)
        #    threshold=0.5 means 50% duty factor (stance vs swing)
        self.rewards.gallop_gait = RewTerm(
            func=mdp.feet_gait,
            weight=1.5,
            params={
                "period": 0.3,
                "offset": [0.0, 0.0, 0.5, 0.5],  # FL, FR, RL, RR
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 0.5,
                "command_name": "base_velocity",
            },
        )

        # 2) Disable the trot-biased air_time_variance penalty — bounding
        #    has inherently different air times between front/rear pairs.
        self.rewards.air_time_variance = RewTerm(
            func=mdp.air_time_variance_penalty,
            weight=0.0,  # disabled
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )

        # 3) Feet air time: boost to encourage flight phases.
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=0.4,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )

        # 4) Flat orientation: very relaxed — bounding produces significant
        #    pitch oscillation that's physically correct.
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.flat_orientation_l2, weight=-0.2
        )

        # 5) Vertical velocity: substantially relaxed — the CoM bounces
        #    vertically during each stride cycle.
        self.rewards.base_linear_velocity = RewTerm(
            func=mdp.lin_vel_z_l2, weight=-0.2
        )

        # 6) Action rate: fast gait transitions need quick action changes.
        self.rewards.action_rate = RewTerm(
            func=mdp.action_rate_l2, weight=-0.01
        )

        # 7) Joint position: wide excursions expected during gallop.
        self.rewards.joint_pos = RewTerm(
            func=mdp.joint_position_penalty,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stand_still_scale": 2.0,
                "velocity_threshold": 0.3,
            },
        )

        # 8) Foot clearance: encourage higher foot lifts for dynamic gait.
        self.rewards.feet_height_body = RewTerm(
            func=mdp.feet_height_body,
            weight=-0.3,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "target_height": -0.20,  # relative to body frame
                "tanh_mult": 2.0,
            },
        )


# ---------------------------------------------------------------------------
# Gentle-rough gallop variant (same terrain as sport rough)
# ---------------------------------------------------------------------------

GALLOP_GENTLE_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(10.0, 10.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.005, 0.03),   # gentle: 5-30 mm bumps
            noise_step=0.005,
            border_width=0.25,
        ),
    },
)


@configclass
class RobotGallopRoughEnvCfg(RobotGallopEnvCfg):
    """Go2 gallop/bound on gentle rough terrain — bumpy ground, no obstacles.

    Inherits all gallop reward shaping from :class:`RobotGallopEnvCfg`
    and adds:
    * Gentle random-uniform terrain (5-30 mm bumps)
    * Terrain difficulty curriculum
    * Slightly relaxed bad_orientation termination (1.0 rad vs 0.8)
    * Richer reset randomisation (velocity + joint perturbation)

    Observation dims match the flat gallop env exactly, enabling direct
    checkpoint resume.  The bumps are gentle enough that proprioception
    alone is sufficient.
    """

    def __post_init__(self):
        super().__post_init__()

        # -- physics: rough-terrain solver (higher njmax for mesh contacts)
        self.sim.physics = VelocityRoughPhysicsCfg()

        # -- terrain: gentle bumps only
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = GALLOP_GENTLE_ROUGH_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = 5

        # -- curriculum: terrain difficulty alongside command magnitude
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        # -- terminations: slightly relaxed for uneven ground
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": 1.0}
        )

        # -- events: richer resets for robustness
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (0.8, 1.2)


@configclass
class RobotGallopRoughPlayEnvCfg(RobotGallopRoughEnvCfg):
    """Play variant for the gentle-rough gallop env."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # spawn across full terrain grid
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # full-range commands immediately
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        # disable noise and pushes during evaluation
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


@configclass
class RobotGallopPlayEnvCfg(RobotGallopEnvCfg):
    """Play variant for the gallop env."""

    def __post_init__(self):
        super().__post_init__()

        # smaller scene for evaluation
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # full-range commands immediately
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        # disable noise and pushes during evaluation
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
