"""Go2 *pace* velocity-tracking environment.

Encourages a lateral pacing gait where same-side legs move in unison:
FL+RL swing together, then FR+RR swing together.  This produces a
characteristic lateral rocking motion seen in camels, giraffes, some
large dogs, and some fast-walking quadrupeds.

Key differences from the trot (sport) env:
* **Gait reward**: ``feet_gait`` with pace offsets — left pair in phase,
  right pair in phase, 180° between left/right sides.
* **Moderate velocity**: pacing is typically a medium-speed gait, faster
  than walk but not sprint-territory.
* **Roll tolerance**: pacing naturally produces lateral body rocking
  (roll oscillation), so roll penalties are relaxed.
* **Air time variance disabled**: same-side pairing inherently has
  different front/rear air times.

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
class RobotPaceEnvCfg(RobotEnvCfg):
    """Go2 pacing env on flat terrain — lateral same-side gait."""

    def __post_init__(self):
        super().__post_init__()

        # -- commands: moderate speed range for pacing.
        #    Pacing is a medium-speed gait; curriculum ramps to 2.0 m/s.
        self.commands.base_velocity.ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.5), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.3, 0.3)
        )
        self.commands.base_velocity.limit_ranges = mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 2.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.8, 0.8)
        )

        # -- rewards: reshape for pacing gait --

        # 1) Gait pattern: PACE
        #    Left pair  (FL=0, RL=2) in phase at offset 0.0
        #    Right pair (FR=1, RR=3) in phase at offset 0.5
        #    Period 0.4s → 2.5 strides/sec (pacing is slightly slower than gallop)
        self.rewards.pace_gait = RewTerm(
            func=mdp.feet_gait,
            weight=1.5,
            params={
                "period": 0.4,
                "offset": [0.0, 0.5, 0.0, 0.5],  # FL, FR, RL, RR — same-side pairs
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "threshold": 0.5,
                "command_name": "base_velocity",
            },
        )

        # 2) Disable air_time_variance — pacing has inherently different
        #    front/rear air times within each side pair.
        self.rewards.air_time_variance = RewTerm(
            func=mdp.air_time_variance_penalty,
            weight=0.0,  # disabled
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
        )

        # 3) Feet air time: moderate bonus for clean swing phases.
        self.rewards.feet_air_time = RewTerm(
            func=mdp.feet_air_time,
            weight=0.4,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "command_name": "base_velocity",
                "threshold": 0.5,
            },
        )

        # 4) Flat orientation: relaxed — pacing produces lateral roll
        #    oscillation which is physically correct for this gait.
        self.rewards.flat_orientation_l2 = RewTerm(
            func=mdp.flat_orientation_l2, weight=-0.15
        )

        # 5) Vertical velocity: moderately relaxed — pacing has some
        #    vertical bounce but less than bounding.
        self.rewards.base_linear_velocity = RewTerm(
            func=mdp.lin_vel_z_l2, weight=-0.3
        )

        # 6) Action rate: moderate smoothness.
        self.rewards.action_rate = RewTerm(
            func=mdp.action_rate_l2, weight=-0.01
        )

        # 7) Joint position: moderate excursions during pace.
        self.rewards.joint_pos = RewTerm(
            func=mdp.joint_position_penalty,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "stand_still_scale": 2.0,
                "velocity_threshold": 0.3,
            },
        )

        # 8) Foot clearance: moderate lift target.
        self.rewards.feet_height_body = RewTerm(
            func=mdp.feet_height_body,
            weight=-0.3,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
                "target_height": -0.22,  # slightly lower than gallop
                "tanh_mult": 2.0,
            },
        )


@configclass
class RobotPacePlayEnvCfg(RobotPaceEnvCfg):
    """Play variant for the pace env."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # full-range commands immediately
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        # disable noise and pushes during evaluation
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None


# ---------------------------------------------------------------------------
# Gentle-rough pace variant
# ---------------------------------------------------------------------------

PACE_GENTLE_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
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
            noise_range=(0.005, 0.03),
            noise_step=0.005,
            border_width=0.25,
        ),
    },
)


@configclass
class RobotPaceRoughEnvCfg(RobotPaceEnvCfg):
    """Go2 pacing on gentle rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.sim.physics = VelocityRoughPhysicsCfg()

        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = PACE_GENTLE_ROUGH_TERRAIN_CFG
        self.scene.terrain.max_init_terrain_level = 5

        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": 1.0}
        )

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
class RobotPaceRoughPlayEnvCfg(RobotPaceRoughEnvCfg):
    """Play variant for the gentle-rough pace env."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
