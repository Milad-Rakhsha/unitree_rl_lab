"""Go2 *sport* velocity-tracking environments (flat + gentle-rough).

Two env configs live here:

* **RobotSportEnvCfg** – flat terrain, fast commands, relaxed penalties.
* **RobotSportRoughEnvCfg** – *gentle* rough terrain (random-uniform bumps
  only, no stairs/boxes/slopes) for robustness fine-tuning on top of the
  flat sport gait.

Both share the same reward relaxations that encourage agile locomotion:
higher velocity command ranges (±1.5 m/s), reduced orientation/action-rate/
joint-position penalties, and boosted feet-air-time reward.
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
class RobotSportEnvCfg(RobotEnvCfg):
    """Go2 sport env on flat terrain – faster commands, relaxed orientation."""

    def __post_init__(self):
        super().__post_init__()

        # -- commands: widen the velocity command ranges for agile running.
        #    Curriculum starts at ±0.2/±0.1/±0.5 and ramps toward
        #    ±1.5/±0.6/±1.5.  This is faster than the base quadruped env
        #    (±1.0/±0.4/±1.0) but achievable on flat ground.
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.2, 0.2), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.5, 0.5)
        )
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.6, 0.6), ang_vel_z=(-1.5, 1.5)
        )

        # -- rewards: relax penalties that constrain dynamic gaits.

        # Flat-orientation: allow body tilt during sprinting/turning
        # (original weight = -2.5, too strict for running gaits).
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.flat_orientation_l2, weight=-0.5
        )

        # Action rate: running gaits need faster action changes than
        # conservative walking.  Halved from -0.1 to -0.02.
        self.rewards.action_rate = RewTerm(
            func=mdp.action_rate_l2, weight=-0.02
        )

        # Joint position: wide joint excursions are natural during
        # running.  Reduced from -0.7 to -0.2, and stand_still_scale
        # lowered so the policy doesn't get over-penalized when moving.
        self.rewards.joint_pos = RewTerm(
            func=mdp.joint_position_penalty,
            weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stand_still_scale": 2.0,
                "velocity_threshold": 0.3,
            },
        )

        # Vertical velocity: some bounce is expected during trot/gallop.
        # Reduced from -2.0 to -0.5.
        self.rewards.base_linear_velocity = RewTerm(
            func=mdp.lin_vel_z_l2, weight=-0.5
        )

        # Feet air time: boost to encourage more dynamic, lifted gaits
        # rather than shuffling.  Increased from 0.1 to 0.3.
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=0.3,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )


@configclass
class RobotSportPlayEnvCfg(RobotSportEnvCfg):
    """Play variant for the flat sport env."""

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


# ---------------------------------------------------------------------------
# Gentle-rough sport variant
# ---------------------------------------------------------------------------

# Terrain generator with ONLY random-uniform bumps — no stairs, boxes, or
# slopes.  Noise range is reduced compared to the full rough env so the
# terrain is "bumpy ground" rather than an obstacle course.
GENTLE_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
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
class RobotSportRoughEnvCfg(RobotSportEnvCfg):
    """Go2 sport env on gentle rough terrain — bumpy ground, no obstacles.

    Inherits all sport reward relaxations from :class:`RobotSportEnvCfg`
    and adds:
    * Gentle random-uniform terrain (5-30 mm bumps)
    * Terrain difficulty curriculum
    * Slightly relaxed bad_orientation termination (1.0 rad vs 0.8)
    * Richer reset randomisation (velocity + joint perturbation)

    Note: no privileged height-scan is added so the observation dims
    match the flat sport env exactly, enabling direct checkpoint resume.
    The bumps are gentle enough that proprioception alone is sufficient.
    """

    def __post_init__(self):
        super().__post_init__()

        # -- physics: rough-terrain solver (higher njmax for mesh contacts)
        self.sim.physics = VelocityRoughPhysicsCfg()

        # -- terrain: gentle bumps only
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = GENTLE_ROUGH_TERRAIN_CFG
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
class RobotSportRoughPlayEnvCfg(RobotSportRoughEnvCfg):
    """Play variant for the gentle-rough sport env."""

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
