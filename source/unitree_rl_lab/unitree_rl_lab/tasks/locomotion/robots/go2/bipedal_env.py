"""Custom env classes for the Go2 bipedal task.

This module is intentionally tiny. It exists because we need one narrow
deviation from stock :class:`isaaclab.envs.ManagerBasedRLEnv`: clipping
the per-step total reward to ``>= 0`` (Legged-Gym's ``only_positive_rewards``
trick). Everything else stays in :mod:`bipedal_env_cfg`.
"""

from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnv


class PositiveRewardManagerBasedRLEnv(ManagerBasedRLEnv):
    """``ManagerBasedRLEnv`` with per-step total reward floored at ``0``.

    The individual reward term values (and their episodic sums used for
    TensorBoard logging) are left untouched — penalties still drive shaping
    correctly. Only the scalar the RL algorithm sees in :attr:`reward_buf`
    is clamped. This emulates the ``only_positive_rewards`` flag in
    Legged-Gym / arclab-hku's bipedal reference, which is reported to help
    bootstrap hard-to-explore tasks (the policy is never punished into a
    net-negative step and thus keeps trying).
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        # Honor the config-level flag. When ``only_positive_rewards``
        # is ``False`` the class behaves exactly like the stock
        # ``ManagerBasedRLEnv`` — useful for variants that rely on
        # negative total rewards to punish specific "cheat" strategies
        # (e.g. heavy swing-foot contact penalty).
        if not getattr(cfg, "only_positive_rewards", False):
            return

        _orig_compute = self.reward_manager.compute

        def _compute_clamped(dt: float):
            buf = _orig_compute(dt)
            buf.clamp_(min=0.0)
            return buf

        self.reward_manager.compute = _compute_clamped  # type: ignore[method-assign]
