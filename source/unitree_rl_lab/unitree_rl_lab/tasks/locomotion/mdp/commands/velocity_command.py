from __future__ import annotations

from dataclasses import MISSING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


def _pitch_invariant_yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    """Yaw-only quaternion extracted from the horizontal projection of body-Y.

    Unlike :func:`~isaaclab.utils.math.yaw_quat` (ZYX Euler extraction),
    this variant does not hit a singularity at a ``±pi/2`` body pitch,
    which is exactly the bipedal stance target for a quadruped standing
    on its hind legs. See
    :func:`~unitree_rl_lab.tasks.locomotion.robots.go2.bipedal_mdp.bipedal_yaw_quat`
    for the detailed rationale. The two helpers agree at flat poses.
    """
    qx = quat[..., 0]
    qy = quat[..., 1]
    qz = quat[..., 2]
    qw = quat[..., 3]
    y_w_x = 2.0 * (qx * qy - qw * qz)
    y_w_y = 1.0 - 2.0 * (qx * qx + qz * qz)
    yaw = torch.atan2(-y_w_x, y_w_y)
    out = torch.zeros_like(quat)
    out[..., 2] = torch.sin(0.5 * yaw)
    out[..., 3] = torch.cos(0.5 * yaw)
    return math_utils.normalize(out)


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command whose debug arrows are drawn in the
    gravity-aligned (yaw-only) frame.

    The upstream ``UniformVelocityCommand`` builds both the goal-velocity
    and current-velocity arrows using the full base rotation, which makes
    the arrows point along body-X (vertical in world space) whenever the
    base is pitched -- e.g. a Go2 standing on its hind legs. This subclass
    overrides the debug callback so:

    * The goal arrow rotates the yaw-frame command only by the base's
      yaw-only quaternion, so it always points along the world horizontal
      plane.
    * The current arrow is drawn directly from the world-frame linear
      velocity (``root_lin_vel_w``), so it reflects actual horizontal
      motion regardless of body tilt.

    The yaw-extraction method is selected via
    :attr:`UniformLevelVelocityCommandCfg.pitch_invariant_yaw`. When set
    the command uses a body-Y-based extraction that stays continuous
    across a ``±pi/2``-pitch singularity (required for bipedal stances);
    otherwise the standard :func:`~isaaclab.utils.math.yaw_quat` is used.
    """

    cfg: "UniformLevelVelocityCommandCfg"

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        base_pos_w = wp.to_torch(self.robot.data.root_pos_w).clone()
        base_pos_w[:, 2] += 0.5

        base_quat_w = wp.to_torch(self.robot.data.root_quat_w)
        if getattr(self.cfg, "pitch_invariant_yaw", False):
            yaw_q = _pitch_invariant_yaw_quat(base_quat_w)
        else:
            yaw_q = math_utils.yaw_quat(base_quat_w)

        vel_des_arrow_scale, vel_des_arrow_quat = self._xy_to_arrow(
            self.command[:, :2], frame_rot=yaw_q
        )
        vel_arrow_scale, vel_arrow_quat = self._xy_to_arrow(
            wp.to_torch(self.robot.data.root_lin_vel_w)[:, :2], frame_rot=None
        )

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _xy_to_arrow(
        self, xy_velocity: torch.Tensor, frame_rot: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build arrow scale and orientation for an xy velocity.

        Args:
            xy_velocity: Horizontal velocity, shape (N, 2). When ``frame_rot``
                is given the velocity is interpreted in that rotated frame
                (e.g. yaw frame for commands); when ``frame_rot`` is ``None``
                the velocity is assumed to be expressed in the world frame.
            frame_rot: Optional quaternion rotation to apply to the arrow
                (e.g. ``yaw_quat(base)``). If ``None`` the arrow is built
                directly in world frame.

        Returns:
            Tuple of ``(arrow_scale, arrow_quat)`` suitable for
            :class:`~isaaclab.markers.VisualizationMarkers.visualize`.
        """
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        if frame_rot is not None:
            arrow_quat = math_utils.quat_mul(frame_rot, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type[UniformLevelVelocityCommand] = UniformLevelVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING

    pitch_invariant_yaw: bool = False
    """If true, extract the yaw component of the base orientation from the
    horizontal projection of body-Y rather than via the standard ZYX Euler
    formula. Required for tasks where the base sits at a ``±pi/2`` pitch
    (bipedal stances on hind legs), where the ZYX extraction is singular.
    """
