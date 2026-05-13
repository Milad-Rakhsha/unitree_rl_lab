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

