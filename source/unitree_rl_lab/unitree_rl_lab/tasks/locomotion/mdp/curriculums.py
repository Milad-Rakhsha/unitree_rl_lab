from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# Minimum episode duration (in seconds) for an env to contribute to the
# curriculum's reward metric. Very short episodes are dominated by spawn /
# randomization transients and add noise to the per-step-reward estimate
# without carrying meaningful tracking information.
_MIN_EPISODE_LENGTH_S = 1.0


def _pooled_per_step_reward(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
) -> torch.Tensor | None:
    """Length-weighted per-step reward across ``env_ids``.

    ``reward_manager._episode_sums`` accumulates ``r_t * weight * step_dt``
    per step, so dividing an env's accumulator by that env's actual episode
    duration (in seconds) recovers ``weight * <r_t>``. Envs that run for
    fewer than :data:`_MIN_EPISODE_LENGTH_S` seconds are dropped to reduce
    noise from spawn transients.

    Args:
        env: Environment instance.
        env_ids: Environment indices resetting at the current step. At
            curriculum-compute time, ``env.episode_length_buf[env_ids]``
            still holds the pre-reset step count for each of these envs
            (IsaacLab resets the buffer *after* the curriculum manager
            runs in :meth:`ManagerBasedRLEnv._reset_idx`).
        reward_term_name: Name of the reward term to pool over.

    Returns:
        Pooled per-step reward ``weight * <r_t>`` as a 0-d tensor, or
        ``None`` if no env in ``env_ids`` cleared the minimum-length
        filter.
    """
    episode_sums = env.reward_manager._episode_sums[reward_term_name][env_ids]
    episode_lengths_s = env.episode_length_buf[env_ids].float() * env.step_dt
    valid = episode_lengths_s >= _MIN_EPISODE_LENGTH_S
    if not bool(valid.any()):
        return None
    return episode_sums[valid].sum() / episode_lengths_s[valid].sum()


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    reward_threshold_fraction: float = 0.8,
) -> torch.Tensor:
    """Widen the ``lin_vel_x`` / ``lin_vel_y`` command ranges when tracking is good.

    At each episode boundary, a length-weighted per-step tracking reward
    is pooled across the resetting envs and compared to
    ``reward_term.weight * reward_threshold_fraction``; when it exceeds
    the threshold, the ``ranges.lin_vel_x`` / ``ranges.lin_vel_y``
    intervals are grown by :math:`\\pm 0.1` m/s and clamped to
    ``limit_ranges``.

    Args:
        env: Environment instance.
        env_ids: Environment indices that reset at the current step.
        reward_term_name: Name of the linear-velocity tracking reward
            term used to gate widening.
        reward_threshold_fraction: Fraction of ``reward_term.weight`` the
            pooled per-step reward must exceed to advance the curriculum.
            Tighter kernels (smaller ``std``) cap the achievable mean
            reward below the perfect tracker's ``weight``; use a smaller
            fraction to keep the curriculum progressing in that regime.

    Note:
        EXPERIMENTAL (2026-04-18): replaced the earlier
        ``mean(episode_sums) / max_episode_length_s`` normalization with
        length-weighted pooling across valid envs. The old normalization
        divided every env's cumulative reward by the full episode
        duration even when the env had early-terminated, which
        systematically under-estimated per-step reward for tasks with
        frequent fall-based terminations (bipedal locomotion). See
        :func:`_pooled_per_step_reward` for the math.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    if env.common_step_counter % env.max_episode_length == 0:
        reward = _pooled_per_step_reward(env, env_ids, reward_term_name)
        if reward is not None and reward > reward_term.weight * reward_threshold_fraction:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
    reward_threshold_fraction: float = 0.8,
) -> torch.Tensor:
    """Widen the ``ang_vel_z`` command range when yaw tracking is good.

    At each episode boundary, a length-weighted per-step tracking reward
    is pooled across the resetting envs and compared to
    ``reward_term.weight * reward_threshold_fraction``; when it exceeds
    the threshold, ``ranges.ang_vel_z`` is grown by :math:`\\pm 0.1`
    rad/s and clamped to ``limit_ranges``.

    Args:
        env: Environment instance.
        env_ids: Environment indices that reset at the current step.
        reward_term_name: Name of the yaw-rate tracking reward term used
            to gate widening.
        reward_threshold_fraction: Fraction of ``reward_term.weight`` the
            pooled per-step reward must exceed to advance the curriculum.
            Tighter kernels (smaller ``std``) cap the achievable mean
            reward below the perfect tracker's ``weight``; use a smaller
            fraction to keep the curriculum progressing in that regime.

    Note:
        EXPERIMENTAL (2026-04-18): replaced the earlier
        ``mean(episode_sums) / max_episode_length_s`` normalization with
        length-weighted pooling across valid envs. The old normalization
        divided every env's cumulative reward by the full episode
        duration even when the env had early-terminated, which
        systematically under-estimated per-step reward for tasks with
        frequent fall-based terminations (bipedal locomotion). See
        :func:`_pooled_per_step_reward` for the math.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)

    if env.common_step_counter % env.max_episode_length == 0:
        reward = _pooled_per_step_reward(env, env_ids, reward_term_name)
        if reward is not None and reward > reward_term.weight * reward_threshold_fraction:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
