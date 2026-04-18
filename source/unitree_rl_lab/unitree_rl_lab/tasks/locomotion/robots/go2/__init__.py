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

gym.register(
    id="Unitree-Go2-Velocity-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

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

# -- Bipedal (two-legged) walking on the hind legs. ---
gym.register(
    id="Unitree-Go2-Bipedal-Rear",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalRearEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalRearPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Bipedal-Rear-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalRearFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalRearFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Bipedal (two-legged) walking on the front legs. ---
gym.register(
    id="Unitree-Go2-Bipedal-Front",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalFrontEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalFrontPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Bipedal-Front-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalFrontFlatEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalFrontFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

# -- Recommendation-B variants: flat spawn + positive-only rewards + strong pushes.
#    Use these if the base bipedal variants plateau. They require the custom
#    ``PositiveRewardManagerBasedRLEnv`` entry point so per-step reward is
#    floored at 0 (Legged-Gym's ``only_positive_rewards`` trick).
_BIPEDAL_FLATINIT_ENTRY = f"{__name__}.bipedal_env:PositiveRewardManagerBasedRLEnv"

gym.register(
    id="Unitree-Go2-Bipedal-Rear-FlatInit",
    entry_point=_BIPEDAL_FLATINIT_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalRearFlatInitEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalRearFlatInitPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-Bipedal-Front-FlatInit",
    entry_point=_BIPEDAL_FLATINIT_ENTRY,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalFrontFlatInitEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.bipedal_env_cfg:RobotBipedalFrontFlatInitPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
