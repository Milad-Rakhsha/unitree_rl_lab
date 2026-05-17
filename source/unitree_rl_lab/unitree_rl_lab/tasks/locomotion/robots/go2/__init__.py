import gymnasium as gym

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

