"""Go2 velocity-tracking environment on procedurally-generated rough terrain.

This is the ``unitree_rl_lab`` equivalent of IsaacLab's
``isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py``,
but it keeps the ``unitree_rl_lab`` rewards / observations / command
curriculum / action clipping / noise model, which the authors have tuned for
sim-to-real transfer on Go2 hardware.

Differences relative to the flat :mod:`velocity_env_cfg`:

* Terrain is switched from a plane to ``ROUGH_TERRAINS_CFG`` (random_rough,
  boxes, stairs, slopes, ...), scaled down for the Go2 form factor.
* A ``RayCasterCfg`` height scanner is added and exposed **only as a critic
  observation** (privileged). The actor still sees the same proprioceptive
  stack, so policies trained here are directly deployable with the existing
  hardware deploy code (no raycast/LiDAR required on the real robot).
* ``terrain_levels_vel`` curriculum is added on top of the existing
  ``lin_vel_cmd_levels``: difficulty expands along two independent axes
  (terrain complexity and command magnitude).
* ``bad_orientation`` is relaxed to 1.2 rad so the robot can tilt on slopes
  and stairs without early-terminating.
* Base reset gets non-zero initial linear/angular velocity and joint angles
  are randomized in ``[0.5, 1.5]`` around the default pose for richer
  initial conditions (matching IsaacLab's upstream rough env).

All rewards, action/observation scales, noise levels, deploy-relevant
clipping and ``CriticCfg``/``PolicyCfg`` asymmetry come from
:class:`RobotEnvCfg` via subclassing.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import RobotEnvCfg, RobotSceneCfg


@configclass
class RobotRoughSceneCfg(RobotSceneCfg):
    """Scene with procedural rough terrain and a privileged height scanner."""

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class RobotRoughEnvCfg(RobotEnvCfg):
    """Go2 rough-terrain velocity tracking env (policy-deploy-compatible)."""

    scene: RobotRoughSceneCfg = RobotRoughSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    def __post_init__(self):
        super().__post_init__()

        # -- terrain: swap the flat plane for procedural rough terrain.
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 5
        # Scale Go2-sized terrain features (matches IsaacLab upstream).
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # -- sensors: tick the height scanner at the policy decimation rate.
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # -- observations: add a privileged height-scan to the critic group.
        #    Actor keeps the exact proprioceptive observation stack from
        #    :mod:`velocity_env_cfg`, so the deployed policy is unchanged.
        self.observations.critic.height_scanner = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        # -- curriculum: progress along terrain difficulty alongside the
        #    existing command-magnitude curriculum.
        self.curriculum.terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
        # terrain_levels_vel requires the terrain generator to opt in to
        # curriculum-based grid layout.
        self.scene.terrain.terrain_generator.curriculum = True

        # -- terminations: relax tip-over to tolerate slopes/stairs.
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation, params={"limit_angle": 1.2}
        )

        # -- events: richer resets so policies see diverse initial conditions
        #    (same shape as IsaacLab's upstream rough env).
        self.events.reset_base.params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (0.5, 1.5)


@configclass
class RobotRoughPlayEnvCfg(RobotRoughEnvCfg):
    """Play variant: small scene, max-difficulty commands, no domain randomization."""

    def __post_init__(self):
        super().__post_init__()

        # smaller, faster-to-load scene
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # spawn randomly across the terrain grid instead of following the curriculum
        self.scene.terrain.max_init_terrain_level = None
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # issue full-range commands immediately (no curriculum ramp)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges

        # disable observation noise and push events during evaluation
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
