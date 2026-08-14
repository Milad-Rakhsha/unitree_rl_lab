"""One-variable DVI ablations from the stock Isaac Lab Go2 flat baseline.

Each class inherits the stock task configuration directly.  Keep every change
local and explicit so that results identify the first Unitree RL Lab setting
that alters the reproduced stock DVI behavior.
"""

import copy

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.utils import preset

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import RobotEnvCfg


@configclass
class UnitreeGo2FlatStockSelfCollisionEnvCfg(UnitreeGo2FlatEnvCfg):
    """Stock Go2 DVI task with only robot self-collision enabled."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True


@configclass
class UnitreeGo2FlatStockMechanicalGroupEnvCfg(UnitreeGo2FlatEnvCfg):
    """Stock task/PPO with the remaining Unitree RL Lab mechanical DVI group.

    This intentionally groups the known coupled mechanics: custom Go2 asset
    definition, current custom solver values, and the all-joint/frictional DVI
    actuator.  It is split only if this grouped trial regresses.
    """

    def __post_init__(self):
        super().__post_init__()

        # Unitree RL Lab asset properties (including self-collision and the
        # 100 rad/s linear/angular velocity caps) instead of the stock asset.
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Current Unitree RL Lab flat-velocity DVI solver deviations from stock.
        solver = self.sim.physics.newton_dvi.solver_cfg
        solver.joint_alpha = 0.0
        solver.contact_max_iterations = 10

        # Current custom DVI actuator semantics: all joints and 0.01 friction.
        self.scene.robot.actuators["GO2HV"] = preset(
            default=self.scene.robot.actuators["GO2HV"],
            newton_dvi=ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                effort_limit_sim=23.5,
                velocity_limit_sim=30.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.01,
            ),
        )


@configclass
class UnitreeGo2FlatStockDynamicsGroupEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Mechanical group plus Unitree command/reset/disturbance dynamics only."""

    def __post_init__(self):
        super().__post_init__()
        unitree = RobotEnvCfg()
        self.commands = unitree.commands
        self.events = unitree.events


@configclass
class UnitreeGo2FlatStockObservationsGroupEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Mechanical group plus Unitree policy/critic observation definitions only."""

    def __post_init__(self):
        super().__post_init__()
        self.observations = RobotEnvCfg().observations


@configclass
class UnitreeGo2FlatStockCoreRewardsGroupEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Mechanical group plus Unitree tracking/base-orientation reward subset."""

    def __post_init__(self):
        super().__post_init__()
        rewards = RobotEnvCfg().rewards
        for name in (
            "joint_vel", "joint_acc", "joint_torques", "action_rate", "dof_pos_limits",
            "energy", "joint_pos", "feet_air_time", "air_time_variance", "feet_slide",
            "undesired_contacts",
        ):
            setattr(rewards, name, None)
        self.rewards = rewards


@configclass
class UnitreeGo2FlatStockJointFeetRewardsGroupEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Mechanical group plus Unitree joint/feet/contact reward subset."""

    def __post_init__(self):
        super().__post_init__()
        rewards = RobotEnvCfg().rewards
        for name in (
            "track_lin_vel_xy", "track_ang_vel_z", "base_linear_velocity",
            "base_angular_velocity", "flat_orientation_l2",
        ):
            setattr(rewards, name, None)
        self.rewards = rewards


@configclass
class UnitreeGo2FlatStockJointPenaltiesRewardsGroupEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Stock task plus Unitree non-contact joint/action reward penalties only."""

    def __post_init__(self):
        super().__post_init__()
        rewards = RobotEnvCfg().rewards
        for name in (
            "track_lin_vel_xy", "track_ang_vel_z", "base_linear_velocity",
            "base_angular_velocity", "flat_orientation_l2", "feet_air_time",
            "air_time_variance", "feet_slide", "undesired_contacts",
        ):
            setattr(rewards, name, None)
        self.rewards = rewards


@configclass
class UnitreeGo2FlatStockFootContactRewardsGroupEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Stock task plus Unitree foot/contact reward terms only."""

    def __post_init__(self):
        super().__post_init__()
        rewards = RobotEnvCfg().rewards
        for name in (
            "track_lin_vel_xy", "track_ang_vel_z", "base_linear_velocity",
            "base_angular_velocity", "flat_orientation_l2", "joint_vel",
            "joint_acc", "joint_torques", "action_rate", "dof_pos_limits",
            "energy", "joint_pos",
        ):
            setattr(rewards, name, None)
        self.rewards = rewards


@configclass
class _UnitreeGo2FlatStockSingleRewardDifferenceEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Apply exactly one Unitree reward-term difference to the stock reward set."""

    unitree_term: str = ""
    stock_term: str | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.stock_term is not None:
            setattr(self.rewards, self.stock_term, None)
        unitree_term = getattr(RobotEnvCfg().rewards, self.unitree_term)
        setattr(self.rewards, self.unitree_term, copy.deepcopy(unitree_term))


@configclass
class UnitreeGo2FlatStockJointVelRewardEnvCfg(_UnitreeGo2FlatStockSingleRewardDifferenceEnvCfg):
    unitree_term = "joint_vel"


@configclass
class UnitreeGo2FlatStockJointTorquesRewardEnvCfg(_UnitreeGo2FlatStockSingleRewardDifferenceEnvCfg):
    unitree_term = "joint_torques"
    stock_term = "dof_torques_l2"


@configclass
class UnitreeGo2FlatStockActionRateRewardEnvCfg(_UnitreeGo2FlatStockSingleRewardDifferenceEnvCfg):
    unitree_term = "action_rate"
    stock_term = "action_rate_l2"


@configclass
class UnitreeGo2FlatStockDofPosLimitsRewardEnvCfg(_UnitreeGo2FlatStockSingleRewardDifferenceEnvCfg):
    unitree_term = "dof_pos_limits"


@configclass
class UnitreeGo2FlatStockEnergyRewardEnvCfg(_UnitreeGo2FlatStockSingleRewardDifferenceEnvCfg):
    unitree_term = "energy"


@configclass
class UnitreeGo2FlatStockJointPosRewardEnvCfg(_UnitreeGo2FlatStockSingleRewardDifferenceEnvCfg):
    unitree_term = "joint_pos"


@configclass
class UnitreeGo2FlatStockFeetAirTimeRewardEnvCfg(_UnitreeGo2FlatStockSingleRewardDifferenceEnvCfg):
    unitree_term = "feet_air_time"


@configclass
class UnitreeGo2FlatStockAirTimeVarianceRewardEnvCfg(_UnitreeGo2FlatStockSingleRewardDifferenceEnvCfg):
    unitree_term = "air_time_variance"


@configclass
class UnitreeGo2FlatStockFeetSlideRewardEnvCfg(_UnitreeGo2FlatStockSingleRewardDifferenceEnvCfg):
    unitree_term = "feet_slide"


@configclass
class UnitreeGo2FlatStockUndesiredContactsRewardEnvCfg(_UnitreeGo2FlatStockSingleRewardDifferenceEnvCfg):
    unitree_term = "undesired_contacts"


@configclass
class _UnitreeGo2FlatStockUndesiredContactsBodiesEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Replace stock undesired-contact reward with one Unitree body subset."""

    body_names: list[str] = []

    def __post_init__(self):
        super().__post_init__()
        reward = copy.deepcopy(RobotEnvCfg().rewards.undesired_contacts)
        reward.params["sensor_cfg"].body_names = self.body_names
        self.rewards.undesired_contacts = reward


@configclass
class UnitreeGo2FlatStockUndesiredHipContactsRewardEnvCfg(_UnitreeGo2FlatStockUndesiredContactsBodiesEnvCfg):
    body_names = [".*_hip"]


@configclass
class UnitreeGo2FlatStockUndesiredThighContactsRewardEnvCfg(_UnitreeGo2FlatStockUndesiredContactsBodiesEnvCfg):
    body_names = [".*_thigh"]


@configclass
class UnitreeGo2FlatStockUndesiredCalfContactsRewardEnvCfg(_UnitreeGo2FlatStockUndesiredContactsBodiesEnvCfg):
    body_names = [".*_calf"]


@configclass
class UnitreeGo2FlatStockRewardsGroupEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Mechanical group plus Unitree reward terms only."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards = RobotEnvCfg().rewards


@configclass
class UnitreeGo2FlatStockTerminationsGroupEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Mechanical group plus Unitree termination terms/contact sensor only."""

    def __post_init__(self):
        super().__post_init__()
        unitree = RobotEnvCfg()
        self.terminations = unitree.terminations
        self.scene.contact_forces = unitree.scene.contact_forces
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class UnitreeGo2FlatStockRewardDoneGroupEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Mechanical group plus Unitree rewards, terminations, and contact sensor only."""

    def __post_init__(self):
        super().__post_init__()
        unitree = RobotEnvCfg()
        self.rewards = unitree.rewards
        self.terminations = unitree.terminations
        self.scene.contact_forces = unitree.scene.contact_forces
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class UnitreeGo2FlatStockLearningSignalGroupEnvCfg(UnitreeGo2FlatStockMechanicalGroupEnvCfg):
    """Mechanical group plus Unitree observations/rewards/dones only."""

    def __post_init__(self):
        super().__post_init__()
        unitree = RobotEnvCfg()
        self.observations = unitree.observations
        self.rewards = unitree.rewards
        self.terminations = unitree.terminations
        # The Unitree command curriculum requires its paired command config
        # (`limit_ranges`), which belongs to the dynamics half of this split.
        # Disable it here to keep this child a valid learning-signal-only test;
        # test the curriculum-command interaction after both independent halves.
        self.curriculum = None
        self.scene.contact_forces = unitree.scene.contact_forces
        self.scene.contact_forces.update_period = self.sim.dt
