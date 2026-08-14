import gymnasium as gym

# Exact stock Isaac Lab Go2 flat-velocity baseline. This references the
# maintained stock configuration directly; subsequent diagnostic variants can
# replace one component at a time without duplicating its settings.
gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg:UnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-SelfCollision",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockSelfCollisionEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-MechanicalGroup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockMechanicalGroupEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

# Binary split of the task-level regression: Unitree reset/command/disturbance
# dynamics versus Unitree observations/rewards/terminations, both with stock PPO.
gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-DynamicsGroup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockDynamicsGroupEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-LearningSignalGroup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockLearningSignalGroupEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

# Second binary split of the learning-signal regression: observations versus
# rewards/terminations. Both retain stock commands/events and stock PPO.
gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-ObservationsGroup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockObservationsGroupEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-RewardDoneGroup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockRewardDoneGroupEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

# Fourth binary split of the failing reward branch: tracking/base terms versus
# joint/feet/contact penalties.
gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-CoreRewardsGroup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockCoreRewardsGroupEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-JointFeetRewardsGroup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockJointFeetRewardsGroupEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

# Fifth binary split of the failing joint/feet/contact reward branch.
gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-JointPenaltiesRewardsGroup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockJointPenaltiesRewardsGroupEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-FootContactRewardsGroup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockFootContactRewardsGroupEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

# Single-difference reward sweep. Each child retains stock rewards except for
# one Unitree term (or Unitree replacement for its corresponding stock term).
for _suffix, _cfg_class in (
    ("JointVelReward", "UnitreeGo2FlatStockJointVelRewardEnvCfg"),
    ("JointTorquesReward", "UnitreeGo2FlatStockJointTorquesRewardEnvCfg"),
    ("ActionRateReward", "UnitreeGo2FlatStockActionRateRewardEnvCfg"),
    ("DofPosLimitsReward", "UnitreeGo2FlatStockDofPosLimitsRewardEnvCfg"),
    ("EnergyReward", "UnitreeGo2FlatStockEnergyRewardEnvCfg"),
    ("JointPosReward", "UnitreeGo2FlatStockJointPosRewardEnvCfg"),
    ("FeetAirTimeReward", "UnitreeGo2FlatStockFeetAirTimeRewardEnvCfg"),
    ("AirTimeVarianceReward", "UnitreeGo2FlatStockAirTimeVarianceRewardEnvCfg"),
    ("FeetSlideReward", "UnitreeGo2FlatStockFeetSlideRewardEnvCfg"),
    ("UndesiredContactsReward", "UnitreeGo2FlatStockUndesiredContactsRewardEnvCfg"),
):
    gym.register(
        id=f"Unitree-Go2-Velocity-Flat-StockDVI-{_suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:{_cfg_class}",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        },
    )

# Final body-level split of the failing Unitree undesired-contact term.
for _suffix, _cfg_class in (
    ("UndesiredHipContactsReward", "UnitreeGo2FlatStockUndesiredHipContactsRewardEnvCfg"),
    ("UndesiredThighContactsReward", "UnitreeGo2FlatStockUndesiredThighContactsRewardEnvCfg"),
    ("UndesiredCalfContactsReward", "UnitreeGo2FlatStockUndesiredCalfContactsRewardEnvCfg"),
):
    gym.register(
        id=f"Unitree-Go2-Velocity-Flat-StockDVI-{_suffix}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:{_cfg_class}",
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        },
    )

# Third binary split: reward terms versus termination/contact semantics.
gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-RewardsGroup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockRewardsGroupEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Flat-StockDVI-TerminationsGroup",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stock_dvi_ablations:UnitreeGo2FlatStockTerminationsGroupEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

# Full Unitree RL Lab MDP on the already-tested mechanical group, but retain
# stock PPO. This is the parent task-level regression.
gym.register(
    id="Unitree-Go2-Velocity-Flat-StockPPO",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# IsaacLab 3.0 / Newton note: upstream ``velocity_env_cfg.py`` dropped the
# separate ``RobotFlatEnvCfg`` / ``RobotFlatPlayEnvCfg`` classes because the
# Newton-era base env is already configured for an infinite plane. We keep
# the ``-Flat`` task id registered for backwards compatibility by pointing
# it at the same base configs.
gym.register(
    id="Unitree-Go2-Velocity-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Velocity-Rough",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobotRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.rough_env_cfg:RobotRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Sport / agile velocity tracking on flat terrain ---
#
# Higher command ranges (up to ±1.5 m/s forward) and relaxed orientation
# penalty so the robot can tilt during aggressive gaits.  Flat terrain
# is deliberate — rough terrain forces conservative gaits that conflict
# with the agility goal.
gym.register(
    id="Unitree-Go2-Velocity-Flat-Sport",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sport_env_cfg:RobotSportEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.sport_env_cfg:RobotSportPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Sport on gentle rough terrain (robustness fine-tuning) ---
#
# Same sport rewards as the flat variant but with gentle random-uniform
# bumps (5-30 mm) and terrain curriculum.  Resume from flat-sport
# checkpoint to preserve the agile gait.
gym.register(
    id="Unitree-Go2-Velocity-Rough-Sport",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.sport_env_cfg:RobotSportRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.sport_env_cfg:RobotSportRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Gallop / bound gait on flat terrain ---
#
# Asymmetric bounding gait: front pair and rear pair move in unison
# with a flight phase between each pair's ground contact.  Commands
# up to ±2.5 m/s forward.  Cheetah-like dynamic locomotion.
gym.register(
    id="Unitree-Go2-Velocity-Gallop",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gallop_env_cfg:RobotGallopEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gallop_env_cfg:RobotGallopPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Gallop / bound on gentle rough terrain ---
#
# Same gallop rewards as flat variant but with gentle random-uniform
# bumps (5-30 mm) and terrain curriculum.  Resume from flat-gallop
# checkpoint to preserve the bound gait.
gym.register(
    id="Unitree-Go2-Velocity-Rough-Gallop",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.gallop_env_cfg:RobotGallopRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.gallop_env_cfg:RobotGallopRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Pace gait on flat terrain ---
#
# Lateral pacing gait: same-side legs move in unison (FL+RL, then FR+RR).
# Produces a characteristic lateral rocking motion.
gym.register(
    id="Unitree-Go2-Velocity-Pace",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace_env_cfg:RobotPaceEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.pace_env_cfg:RobotPacePlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Pace gait on gentle rough terrain ---
gym.register(
    id="Unitree-Go2-Velocity-Rough-Pace",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.pace_env_cfg:RobotPaceRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.pace_env_cfg:RobotPaceRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Stabilization / fall-recovery (scenario-based resets) ---
#
# Uses PresetCfg: run with ``presets=newton`` for Newton backend.
gym.register(
    id="Unitree-Go2-Stabilize",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stabilization_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.stabilization_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Bipedal walking (rear-legs stance, flat spawn). ---
#
# Uses the custom ``PositiveRewardManagerBasedRLEnv`` entry point so the
# per-step total reward can be clipped to ``>= 0`` (Legged-Gym's
# ``only_positive_rewards`` trick, enabled by default on the env cfg).
gym.register(
    id="Unitree-Go2-Bipedal-Walk",
    entry_point=f"{__name__}.bipedal_env:PositiveRewardManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalWalkEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalWalkPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Bipedal walking on rough terrain ---
gym.register(
    id="Unitree-Go2-Bipedal-Walk-Rough",
    entry_point=f"{__name__}.bipedal_env:PositiveRewardManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalWalkRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalWalkRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Bipedal stand-up (quadruped → rear-stance, flat terrain) ---
#
# Gentle stand-up policy: starts from quadruped prone, terminates when
# the robot reaches bipedal stance and holds it calmly.
# Uses PresetCfg: run with ``presets=newton_mjwarp`` for Newton backend.
gym.register(
    id="Unitree-Go2-Bipedal-Standup",
    entry_point=f"{__name__}.bipedal_env:PositiveRewardManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standup_env_cfg:RobotStandupEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.standup_env_cfg:RobotStandupPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Bipedal stand-up on gentle rough terrain ---
gym.register(
    id="Unitree-Go2-Bipedal-Standup-Rough",
    entry_point=f"{__name__}.bipedal_env:PositiveRewardManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.standup_env_cfg:RobotStandupRoughEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.standup_env_cfg:RobotStandupRoughPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

