// Orientation-based fall checks using ArticulationData (see State_RLBase).

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace isaaclab
{
namespace mdp
{

/** Quadruped / flat-terrain check: angle between projected gravity and +body_z (Isaac Lab ``bad_orientation``). */
inline bool bad_orientation(ManagerBasedRLEnv* env, float limit_angle = 1.0)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::fabs(std::acos(-data[2])) > limit_angle;
}

/**
 * Stance-aware deploy-side fall safety: angle between projected gravity in
 * the body frame and a *unit* desired direction. Use for bipedal policies
 * where the nominal pose is pitched ~90° (``desired_gravity`` e.g.
 * ``(-1,0,0)`` for rear-legs stance).
 *
 * Unlike :cpp:func:`bad_orientation`, this does *not* mirror a training
 * termination: the ``Unitree-Go2-Bipedal-Walk`` env does not terminate on
 * orientation (the policy must learn to recover within an episode). The
 * deploy check here is a last-resort cut-off that drops the robot to
 * Passive on a genuine flip-over, regardless of what the policy does.
 */
inline bool bad_target_orientation(ManagerBasedRLEnv* env, float limit_angle, const std::vector<float>& desired_gravity)
{
    if (desired_gravity.size() != 3)
    {
        return false;
    }
    auto & data = env->robot->data.projected_gravity_b;
    const float gx = data[0];
    const float gy = data[1];
    const float gz = data[2];
    const float dx = desired_gravity[0];
    const float dy = desired_gravity[1];
    const float dz = desired_gravity[2];
    float cos_sim = gx * dx + gy * dy + gz * dz;
    cos_sim = std::max(-1.0f, std::min(1.0f, cos_sim));
    return std::acos(cos_sim) > limit_angle;
}

/**
 * Roll-only fall safety: trips when the magnitude of the projected-gravity
 * body-Y component exceeds ``limit``. ``projected_gravity_b[1]`` equals
 * ``sin(roll_angle)`` regardless of body pitch, so this check is invariant
 * under the large pitch excursions a bipedal / stand-up policy deliberately
 * produces. Use alongside :cpp:func:`bad_target_orientation` to catch
 * sideways tip-overs *before* they escalate to a full flip:
 *
 * - ``bad_target_orientation`` covers total deviation from target stance
 *   (pitch + roll combined).
 * - ``bad_roll_gravity_y`` covers sideways-only failure modes so the cut-off
 *   can be tighter for roll than for pitch.
 *
 * ``limit`` is in units of ``sin(angle)``. 0.6 ≈ 37°, 0.4 ≈ 24°.
 */
inline bool bad_roll_gravity_y(ManagerBasedRLEnv* env, float limit)
{
    auto & data = env->robot->data.projected_gravity_b;
    return std::fabs(data[1]) > limit;
}

}
}