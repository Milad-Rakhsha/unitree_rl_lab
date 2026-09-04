#include "FSM/State_RLBase.h"
#include "param.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "LinearInterpolator.h"
#include <unitree/common/time/time_tool.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
float clamp_std(float v)
{
    return std::max(v, 1.0e-6f);
}
}  // namespace

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    // Resolve the surprise-trip target FSM. Priority:
    //   1. ``surprise.target_fsm`` in this state's config (explicit).
    //   2. "Stabilize" if it exists.
    //   3. "FixStand" as a last-resort fallback.
    if (cfg["surprise"] && cfg["surprise"]["target_fsm"])
    {
        surprise_target_fsm_ = cfg["surprise"]["target_fsm"].as<std::string>();
    }
    else if (FSMStringMap.right.count("Stabilize"))
    {
        surprise_target_fsm_ = "Stabilize";
    }
    else if (FSMStringMap.right.count("FixStand"))
    {
        surprise_target_fsm_ = "FixStand";
    }

    parse_surprise_cfg(cfg);
    parse_intro_cfg(cfg);
    parse_auto_transition_cfg(cfg);
    parse_csv_log_cfg(cfg);

    // Auto-fall safety: if the body tilts past ``bad_orientation_limit`` rad
    // from nominal orientation, drop straight to Passive.
    //
    // Quadruped RL uses Isaac Lab ``bad_orientation`` (angle vs +body_z) and
    // typically mirrors the training termination (match the training
    // ``limit_angle`` for best behaviour). For bipedal FSM states, set
    // ``bad_orientation_desired_gravity`` in YAML to enable the stance-aware
    // variant (``bad_target_orientation``, angle vs a unit desired direction);
    // the current bipedal training env has no orientation termination, so
    // this value is a pure deploy-side fall cut-off.
    const float bad_orientation_limit = cfg["bad_orientation_limit"].as<float>(1.0f);

    if (cfg["bad_orientation_desired_gravity"])
    {
        auto desired_gravity = cfg["bad_orientation_desired_gravity"].as<std::vector<float>>();
        if (desired_gravity.size() == 3)
        {
            this->registered_checks.emplace_back(
                std::make_pair(
                    [this, bad_orientation_limit, desired_gravity, state_string]() -> bool {
                        std::lock_guard<std::mutex> lock(observation_mutex);
                        const bool bad =
                            isaaclab::mdp::bad_target_orientation(env.get(), bad_orientation_limit, desired_gravity);
                        if (bad)
                        {
                            const auto& g = env->robot->data.projected_gravity_b;
                            const float gx = g[0], gy = g[1], gz = g[2];
                            const float dx = desired_gravity[0], dy = desired_gravity[1], dz = desired_gravity[2];
                            float c = gx * dx + gy * dy + gz * dz;
                            c = std::max(-1.0f, std::min(1.0f, c));
                            const float ang = std::acos(c);
                            spdlog::warn(
                                "State '{}' bad_target_orientation -> Passive: angle={:.3f} > limit={:.3f}, "
                                "g_b=({:.3f},{:.3f},{:.3f}), desired=({:.3f},{:.3f},{:.3f})",
                                state_string,
                                ang,
                                bad_orientation_limit,
                                gx,
                                gy,
                                gz,
                                dx,
                                dy,
                                dz);
                        }
                        return bad;
                    },
                    FSMStringMap.right.at("Passive")
                )
            );
        }
        else
        {
            spdlog::warn(
                "State '{}': bad_orientation_desired_gravity must have 3 elements (got {}); using quadruped check.",
                state_string,
                desired_gravity.size()
            );
            this->registered_checks.emplace_back(
                std::make_pair(
                    [this, bad_orientation_limit, state_string]() -> bool {
                        std::lock_guard<std::mutex> lock(observation_mutex);
                        const bool bad = isaaclab::mdp::bad_orientation(env.get(), bad_orientation_limit);
                        if (bad)
                        {
                            const auto& g = env->robot->data.projected_gravity_b;
                            const float gx = g[0], gy = g[1], gz = g[2];
                            const float ang = std::fabs(std::acos(std::max(-1.0f, std::min(1.0f, -gz))));
                            spdlog::warn(
                                "State '{}' bad_orientation (quad fallback) -> Passive: "
                                "angle={:.3f} > limit={:.3f}, g_b=({:.3f},{:.3f},{:.3f})",
                                state_string,
                                ang,
                                bad_orientation_limit,
                                gx,
                                gy,
                                gz);
                        }
                        return bad;
                    },
                    FSMStringMap.right.at("Passive")
                )
            );
        }
    }
    else
    {
        this->registered_checks.emplace_back(
            std::make_pair(
                [this, bad_orientation_limit, state_string]() -> bool {
                    std::lock_guard<std::mutex> lock(observation_mutex);
                    const bool bad = isaaclab::mdp::bad_orientation(env.get(), bad_orientation_limit);
                    if (bad)
                    {
                        const auto& g = env->robot->data.projected_gravity_b;
                        const float gx = g[0], gy = g[1], gz = g[2];
                        const float ang = std::fabs(std::acos(std::max(-1.0f, std::min(1.0f, -gz))));
                        spdlog::warn(
                            "State '{}' bad_orientation (quad) -> Passive: angle={:.3f} > limit={:.3f}, "
                            "g_b=({:.3f},{:.3f},{:.3f})",
                            state_string,
                            ang,
                            bad_orientation_limit,
                            gx,
                            gy,
                            gz);
                    }
                    return bad;
                },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    // EXPERIMENTAL (2026-04-18): roll-only fall safety. Registered alongside
    // the pitch-inclusive ``bad_orientation`` / ``bad_target_orientation``
    // check above so the state can trip on sideways tip-over with a tighter
    // threshold than the (pitch-tolerant) total-angle check. Only activates
    // when the YAML sets ``bad_roll_limit`` (float, units of sin(angle)).
    if (cfg["bad_roll_limit"])
    {
        const float bad_roll_limit = cfg["bad_roll_limit"].as<float>();
        this->registered_checks.emplace_back(
            std::make_pair(
                [this, bad_roll_limit, state_string]() -> bool {
                    std::lock_guard<std::mutex> lock(observation_mutex);
                    const bool bad = isaaclab::mdp::bad_roll_gravity_y(env.get(), bad_roll_limit);
                    if (bad)
                    {
                        const auto& g = env->robot->data.projected_gravity_b;
                        spdlog::warn(
                            "State '{}' bad_roll_gravity_y -> Passive: |g_b_y|={:.3f} > limit={:.3f}, "
                            "g_b=({:.3f},{:.3f},{:.3f})",
                            state_string,
                            std::fabs(g[1]),
                            bad_roll_limit,
                            g[0],
                            g[1],
                            g[2]);
                    }
                    return bad;
                },
                FSMStringMap.right.at("Passive")
            )
        );
        spdlog::info(
            "State '{}' roll fall-safety enabled: |g_b_y| > {:.3f} -> Passive.",
            state_string,
            bad_roll_limit);
    }

    if (!surprise_target_fsm_.empty() && FSMStringMap.right.count(surprise_target_fsm_))
    {
        const int surprise_target_fsm = FSMStringMap.right.at(surprise_target_fsm_);
        this->registered_checks.emplace_back(
            std::make_pair(
                [&]()->bool{ return surprise_trip_check(); },
                surprise_target_fsm
            )
        );
    }
    else if (!surprise_target_fsm_.empty())
    {
        spdlog::warn(
            "Surprise target FSM '{}' not found; surprise transition disabled for state '{}'.",
            surprise_target_fsm_,
            state_string
        );
    }

    if (auto_transition_enabled_ && FSMStringMap.right.count(auto_transition_target_))
    {
        const int target_id = FSMStringMap.right.at(auto_transition_target_);
        registered_checks.emplace_back(
            std::make_pair(
                [this]() -> bool {
                    if (!auto_transition_enabled_ || state_enter_wall_time_s_ <= 0.0)
                    {
                        return false;
                    }
                    // Don't auto-transition while the intro phase is still playing.
                    if (intro_running.load())
                    {
                        return false;
                    }
                    const double now = static_cast<double>(unitree::common::GetCurrentTimeMillisecond()) * 1e-3;
                    return (now - state_enter_wall_time_s_) >= static_cast<double>(auto_transition_duration_s_);
                },
                target_id
            )
        );
        spdlog::info(
            "State '{}' will auto-transition to '{}' after {:.2f} s (wall clock).",
            state_string,
            auto_transition_target_,
            auto_transition_duration_s_
        );
    }
    else if (auto_transition_enabled_)
    {
        spdlog::warn(
            "State '{}' auto_transition target '{}' not found; disabled.",
            state_string,
            auto_transition_target_
        );
        auto_transition_enabled_ = false;
    }
}

void State_RLBase::parse_intro_cfg(const YAML::Node& cfg)
{
    if (!cfg["intro"])
    {
        return;
    }

    try
    {
        ts_intro = cfg["intro"]["ts"].as<std::vector<float>>();
        qs_intro = cfg["intro"]["qs"].as<std::vector<std::vector<float>>>();
    }
    catch (const std::exception& e)
    {
        spdlog::error("Failed to parse 'intro' keyframes for state '{}': {}", getStateString(), e.what());
        ts_intro.clear();
        qs_intro.clear();
        return;
    }

    if (ts_intro.size() < 2 || ts_intro.size() != qs_intro.size())
    {
        spdlog::error(
            "State '{}' intro config invalid: ts.size()={}, qs.size()={} (need >=2 and equal).",
            getStateString(),
            ts_intro.size(),
            qs_intro.size()
        );
        ts_intro.clear();
        qs_intro.clear();
        return;
    }

    // An empty first keyframe is a sentinel: ``enter()`` will pin it to the
    // currently commanded joint state so the intro starts from wherever the
    // previous FSM left off. All other keyframes must be fully specified.
    const size_t joint_count = env->robot->data.joint_stiffness.size();
    if (qs_intro.front().empty())
    {
        qs_intro.front().assign(joint_count, 0.0f);
    }
    for (size_t i = 0; i < qs_intro.size(); ++i)
    {
        if (qs_intro[i].size() != joint_count)
        {
            spdlog::error(
                "State '{}' intro keyframe {} has {} joints (expected {}).",
                getStateString(),
                i,
                qs_intro[i].size(),
                joint_count
            );
            ts_intro.clear();
            qs_intro.clear();
            return;
        }
    }

    if (cfg["intro"]["kp"]) intro_kp = cfg["intro"]["kp"].as<std::vector<float>>();
    if (cfg["intro"]["kd"]) intro_kd = cfg["intro"]["kd"].as<std::vector<float>>();

    spdlog::info(
        "State '{}' intro enabled: {} keyframes, duration {:.2f}s (custom kp: {}, custom kd: {}).",
        getStateString(),
        ts_intro.size(),
        ts_intro.back() - ts_intro.front(),
        !intro_kp.empty(),
        !intro_kd.empty()
    );
}

void State_RLBase::parse_auto_transition_cfg(const YAML::Node& cfg)
{
    if (!cfg["auto_transition"] || !cfg["auto_transition"]["enabled"].as<bool>(false))
    {
        return;
    }

    auto node = cfg["auto_transition"];
    if (!node["target_fsm"])
    {
        spdlog::warn("State '{}' auto_transition missing 'target_fsm'.", getStateString());
        return;
    }

    auto_transition_enabled_ = true;
    auto_transition_target_ = node["target_fsm"].as<std::string>();
    auto_transition_duration_s_ = node["duration_s"].as<float>(4.0f);
}

void State_RLBase::parse_csv_log_cfg(const YAML::Node& cfg)
{
    if (!cfg["csv_log"] || !cfg["csv_log"]["enabled"].as<bool>(false))
    {
        return;
    }
    csv_log_enabled_ = true;
    if (cfg["csv_log"]["path"])
    {
        csv_log_path_cfg_ = cfg["csv_log"]["path"].as<std::string>();
    }
}

void State_RLBase::csv_open()
{
    if (!csv_log_enabled_) return;

    std::filesystem::path path;
    if (!csv_log_path_cfg_.empty())
    {
        path = csv_log_path_cfg_;
        if (path.is_relative())
        {
            path = param::proj_dir / path;
        }
    }
    else
    {
        const auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::tm local_tm{};
        localtime_r(&tt, &local_tm);
        std::ostringstream stamp;
        stamp << std::put_time(&local_tm, "%Y%m%d_%H%M%S");
        path = param::proj_dir / "log" / "deploy_fsm" /
               (getStateString() + "_" + stamp.str() + ".csv");
    }

    try
    {
        std::filesystem::create_directories(path.parent_path());
    }
    catch (const std::exception& e)
    {
        spdlog::error("Failed to create CSV log dir '{}': {}", path.parent_path().string(), e.what());
        csv_log_enabled_ = false;
        return;
    }

    csv_file_.open(path);
    if (!csv_file_.is_open())
    {
        spdlog::error("Failed to open CSV log at '{}'.", path.string());
        csv_log_enabled_ = false;
        return;
    }

    const int n = static_cast<int>(env->robot->data.joint_ids_map.size());
    csv_file_ << "t_wall,state,intro_elapsed";
    for (int i = 0; i < n; ++i) csv_file_ << ",qpos_" << i;
    for (int i = 0; i < n; ++i) csv_file_ << ",qvel_" << i;
    csv_file_ << ",base_roll,base_pitch,base_yaw";
    csv_file_ << ",base_vx_b,base_vy_b,base_vz_b";
    csv_file_ << ",base_wx_b,base_wy_b,base_wz_b";
    for (int i = 0; i < n; ++i) csv_file_ << ",action_" << i;
    csv_file_ << ",cmd_vx,cmd_vy,cmd_yaw";
    csv_file_ << "\n";
    csv_file_.flush();
    spdlog::info("State '{}' CSV log started: {}", getStateString(), path.string());
}

void State_RLBase::csv_close()
{
    if (csv_file_.is_open())
    {
        csv_file_.flush();
        csv_file_.close();
        spdlog::info("State '{}' CSV log stopped.", getStateString());
    }
}

void State_RLBase::csv_write_row(
    double wall_s,
    double intro_elapsed_s,
    const std::vector<float>& action_raw)
{
    if (!csv_log_enabled_ || !csv_file_.is_open()) return;

    Eigen::VectorXf q;
    Eigen::VectorXf qv;
    Eigen::Quaternionf quat;
    Eigen::Vector3f w_b;
    float joystick_lx = 0.0f;
    float joystick_ly = 0.0f;
    float joystick_rx = 0.0f;
    bool has_joystick = false;
    {
        std::lock_guard<std::mutex> lock(observation_mutex);
        q = env->robot->data.joint_pos;
        qv = env->robot->data.joint_vel;
        quat = env->robot->data.root_quat_w;
        w_b = env->robot->data.root_ang_vel_b;
        if (auto* joystick = env->robot->data.joystick)
        {
            joystick_lx = joystick->lx();
            joystick_ly = joystick->ly();
            joystick_rx = joystick->rx();
            has_joystick = true;
        }
    }

    const float qw = quat.w(), qx = quat.x(), qy = quat.y(), qz = quat.z();
    const float roll = std::atan2(2.0f * (qw * qx + qy * qz), 1.0f - 2.0f * (qx * qx + qy * qy));
    const float pitch_arg = std::max(-1.0f, std::min(1.0f, 2.0f * (qw * qy - qz * qx)));
    const float pitch = std::asin(pitch_arg);
    const float yaw = std::atan2(2.0f * (qw * qz + qx * qy), 1.0f - 2.0f * (qy * qy + qz * qz));

    float cmd_vx = 0.0f, cmd_vy = 0.0f, cmd_wz = 0.0f;
    if (has_joystick && env->cfg["commands"] && env->cfg["commands"]["base_velocity"])
    {
        const auto ranges = env->cfg["commands"]["base_velocity"]["ranges"];
        cmd_vx = std::clamp(joystick_ly, ranges["lin_vel_x"][0].as<float>(), ranges["lin_vel_x"][1].as<float>());
        cmd_vy = std::clamp(-joystick_lx, ranges["lin_vel_y"][0].as<float>(), ranges["lin_vel_y"][1].as<float>());
        cmd_wz = std::clamp(-joystick_rx, ranges["ang_vel_z"][0].as<float>(), ranges["ang_vel_z"][1].as<float>());
    }

    csv_file_ << std::fixed << std::setprecision(6);
    csv_file_ << wall_s << "," << getStateString() << "," << intro_elapsed_s;
    for (int i = 0; i < q.size(); ++i) csv_file_ << "," << q[i];
    for (int i = 0; i < qv.size(); ++i) csv_file_ << "," << qv[i];
    csv_file_ << "," << roll << "," << pitch << "," << yaw;
    csv_file_ << ",nan,nan,nan";
    csv_file_ << "," << w_b[0] << "," << w_b[1] << "," << w_b[2];
    for (float a : action_raw) csv_file_ << "," << a;
    csv_file_ << "," << cmd_vx << "," << cmd_vy << "," << cmd_wz;
    csv_file_ << "\n";
    // Flush every row — the whole point of this log is to capture the
    // trajectory up to a crash, which would otherwise leave the tail in
    // a buffered stream.
    csv_file_.flush();
}

bool State_RLBase::intro_active() const
{
    return intro_running.load();
}


void State_RLBase::pre_run()
{
    std::lock_guard<std::mutex> lock(observation_mutex);
    FSMState::pre_run();
    // Fall-safety and observations read ``env->robot``; without this, ``projected_gravity_b``
    // only updates on the policy thread (~decimation rate) and can stay at ctor-time / zero IMU
    // samples — ``bad_orientation`` then trips immediately after state entry.
    env->robot->update();
}

void State_RLBase::enter()
{
    surprise_trip.store(false);
    surprise_ema = 0.0f;
    surprise_over_threshold_count = 0;
    surprise_step_count = 0;
    has_prev_transition = false;
    run_warning_count_ = 0;

    state_enter_wall_time_s_ = static_cast<double>(unitree::common::GetCurrentTimeMillisecond()) * 1e-3;

    const size_t joint_count = env->robot->data.joint_stiffness.size();
    const bool has_intro = !ts_intro.empty() && !qs_intro.empty();

    // Pin the first intro keyframe to the currently-commanded joint state so
    // the interpolation starts smoothly from wherever the previous FSM left off.
    if (has_intro)
    {
        std::vector<float> q0(joint_count);
        for (size_t i = 0; i < joint_count; ++i)
        {
            q0[i] = lowcmd->msg_.motor_cmd()[i].q();
        }
        {
            std::lock_guard<std::mutex> lock(intro_mutex);
            qs_intro[0] = q0;
        }
        intro_t0_s = state_enter_wall_time_s_;
        intro_running.store(true);
    }

    for (size_t i = 0; i < joint_count; ++i)
    {
        const bool intro_kp_valid = has_intro && !intro_kp.empty() && i < intro_kp.size();
        const bool intro_kd_valid = has_intro && !intro_kd.empty() && i < intro_kd.size();
        lowcmd->msg_.motor_cmd()[i].kp() =
            intro_kp_valid ? intro_kp[i] : env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() =
            intro_kd_valid ? intro_kd[i] : env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    env->robot->update();
    csv_open();
    policy_thread_running.store(true);
    policy_thread = std::thread([this] { this->policy_loop(); });
}

std::vector<float> State_RLBase::intro_current_q()
{
    std::lock_guard<std::mutex> lock(intro_mutex);
    const double now = static_cast<double>(unitree::common::GetCurrentTimeMillisecond()) * 1e-3;
    const float t = static_cast<float>(now - intro_t0_s);
    if (ts_intro.empty() || qs_intro.empty())
    {
        return {};
    }
    if (t >= ts_intro.back())
    {
        intro_running.store(false);
    }
    return linear_interpolate(t, ts_intro, qs_intro);
}

void State_RLBase::exit()
{
    policy_thread_running.store(false);
    if (policy_thread.joinable())
    {
        policy_thread.join();
    }
    intro_running.store(false);
    state_enter_wall_time_s_ = 0.0;
    csv_close();
}

void State_RLBase::policy_loop()
{
    using clock = std::chrono::high_resolution_clock;
    const std::chrono::duration<double> desired_duration(env->step_dt);
    const auto dt = std::chrono::duration_cast<clock::duration>(desired_duration);
    auto sleep_till = clock::now() + dt;

    {
        std::lock_guard<std::mutex> lock(observation_mutex);
        env->reset();
    }
    // Raw policy outputs are affine-mapped to joint targets (scale * x + offset).
    // ``latest_action`` must hold *processed* positions for run() — not raw zeros.
    // Sending raw 0 as q() collapses the legs until the first ONNX step (~one period).
    {
        std::vector<float> zero_raw(env->action_manager->total_action_dim(), 0.0f);
        env->action_manager->process_action(zero_raw);
        std::lock_guard<std::mutex> lock(action_mutex);
        latest_action = env->action_manager->processed_actions();
    }

    bool intro_kp_applied = !intro_kp.empty() || !intro_kd.empty();
    const bool had_intro = !ts_intro.empty();
    // EXPERIMENTAL (2026-04-18): ``intro_cleanup_pending`` fires once on the
    // first iteration *after* ``intro_running`` transitions to false. It is
    // a strict superset of the legacy ``intro_kp_applied`` gate so the
    // post-intro obs refresh runs even for states that don't override the
    // PD gains during the intro (e.g. BipedalRear).
    bool intro_cleanup_pending = had_intro;

    while (policy_thread_running.load())
    {
        // EXPERIMENTAL (2026-04-18): refresh the observation history at the
        // intro->policy handover. During the 1 s intro, ``run()`` discards
        // the policy's outputs and writes ``intro_current_q()`` instead, but
        // the policy thread keeps running inference at ``step_dt`` and
        // accumulating observations. Those observations reflect the intro
        // ramp (joints track an open-loop target), not the policy's own
        // commands, so by intro-end the obs history carries a
        // ``(joint_pos_rel, last_action)`` pairing that never occurs during
        // training (where ``last_action`` *caused* ``joint_pos``). Calling
        // ``env->reset()`` here repopulates each term's history with the
        // current (intro-end) observation and zeroes ``_action`` so the
        // first post-intro inference sees a clean buffer. ``latest_action``
        // is re-seeded with the processed zero action (= ``default_joint_pos``)
        // so ``run()`` on the FSM thread writes the intro-end target for the
        // handful of milliseconds before the fresh inference below lands.
        if (intro_cleanup_pending && !intro_running.load())
        {
            {
                std::lock_guard<std::mutex> lock(observation_mutex);
                env->reset();
            }
            {
                std::vector<float> zero_raw(env->action_manager->total_action_dim(), 0.0f);
                env->action_manager->process_action(zero_raw);
                std::lock_guard<std::mutex> lock(action_mutex);
                latest_action = env->action_manager->processed_actions();
            }
            if (intro_kp_applied)
            {
                for (size_t i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
                {
                    lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
                    lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
                }
                intro_kp_applied = false;
            }
            intro_cleanup_pending = false;
            spdlog::info("State '{}' intro complete; policy now in control.", getStateString());
        }

        env->episode_length += 1;
        std::unordered_map<std::string, std::vector<float>> obs_map;
        {
            std::lock_guard<std::mutex> lock(observation_mutex);
            env->robot->update();
            obs_map = env->observation_manager->compute();
        }

        std::vector<float> action;
        try
        {
            action = env->alg->act(obs_map);
        }
        catch (const std::exception& e)
        {
            spdlog::error("ONNX policy act() threw in state '{}': {}", getStateString(), e.what());
            throw;
        }

        env->action_manager->process_action(action);

        {
            std::lock_guard<std::mutex> lock(action_mutex);
            latest_action = env->action_manager->processed_actions();
        }

        if (csv_log_enabled_ && csv_file_.is_open())
        {
            const double now_s = static_cast<double>(unitree::common::GetCurrentTimeMillisecond()) * 1e-3;
            const double intro_elapsed = (state_enter_wall_time_s_ > 0.0)
                ? (now_s - state_enter_wall_time_s_)
                : 0.0;
            csv_write_row(now_s, intro_elapsed, action);
        }

        if (surprise_enabled.load() && obs_map.count("obs"))
        {
            update_surprise_gate(obs_map.at("obs"), action);
        }

        std::this_thread::sleep_until(sleep_till);
        sleep_till += dt;
    }
}

bool State_RLBase::parse_surprise_cfg(const YAML::Node& cfg)
{
    if (!cfg["surprise"] || !cfg["surprise"]["enabled"].as<bool>(false))
    {
        spdlog::info("Surprise gate disabled for state '{}'.", this->getStateString());
        return false;
    }

    auto surprise_cfg = cfg["surprise"];
    if (!surprise_cfg["model_path"] || !surprise_cfg["stats_path"])
    {
        spdlog::warn("Surprise gate requested but 'model_path' or 'stats_path' is missing.");
        return false;
    }

    std::filesystem::path model_path = surprise_cfg["model_path"].as<std::string>();
    std::filesystem::path stats_path = surprise_cfg["stats_path"].as<std::string>();
    if (model_path.is_relative()) model_path = param::proj_dir / model_path;
    if (stats_path.is_relative()) stats_path = param::proj_dir / stats_path;
    surprise_threshold = surprise_cfg["threshold"].as<float>(surprise_threshold);
    surprise_ema_alpha = surprise_cfg["ema_alpha"].as<float>(surprise_ema_alpha);
    surprise_min_steps = surprise_cfg["min_steps"].as<int>(surprise_min_steps);
    surprise_consecutive_steps = surprise_cfg["consecutive_steps"].as<int>(surprise_consecutive_steps);
    surprise_use_ema = surprise_cfg["use_ema"].as<bool>(surprise_use_ema);
    surprise_log_every_steps = surprise_cfg["log_every_steps"].as<int>(surprise_log_every_steps);

    YAML::Node stats;
    try
    {
        stats = YAML::LoadFile(stats_path);
    }
    catch (const std::exception& e)
    {
        spdlog::error("Failed to load surprise stats '{}': {}", stats_path.string(), e.what());
        return false;
    }

    if (!stats["obs_mean"] || !stats["obs_std"] || !stats["action_mean"] || !stats["action_std"])
    {
        spdlog::error("Invalid surprise stats file '{}'.", stats_path.string());
        return false;
    }

    obs_mean = stats["obs_mean"].as<std::vector<float>>();
    obs_std = stats["obs_std"].as<std::vector<float>>();
    action_mean = stats["action_mean"].as<std::vector<float>>();
    action_std = stats["action_std"].as<std::vector<float>>();
    for (auto& s : obs_std) s = clamp_std(s);
    for (auto& s : action_std) s = clamp_std(s);

    if (!setup_surprise_model(model_path))
    {
        return false;
    }

    surprise_enabled.store(true);
    spdlog::info(
        "Surprise gate enabled (threshold: {:.3f}, alpha: {:.3f}, min_steps: {}, consecutive: {}, use_ema: {}, log_every: {}).",
        surprise_threshold,
        surprise_ema_alpha,
        surprise_min_steps,
        surprise_consecutive_steps,
        surprise_use_ema,
        surprise_log_every_steps
    );
    return true;
}

bool State_RLBase::setup_surprise_model(const std::filesystem::path& model_path)
{
    try
    {
        surprise_session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        surprise_session = std::make_unique<Ort::Session>(
            surprise_ort_env,
            model_path.string().c_str(),
            surprise_session_options
        );

        surprise_input_name_storage.clear();
        surprise_input_names.clear();
        for (size_t i = 0; i < surprise_session->GetInputCount(); ++i)
        {
            auto input_name = surprise_session->GetInputNameAllocated(i, surprise_allocator);
            surprise_input_name_storage.push_back(input_name.get());
        }
        for (auto& s : surprise_input_name_storage) surprise_input_names.push_back(s.c_str());

        surprise_concat_obs_action_input = false;
        if (surprise_input_names.size() == 2)
        {
            surprise_obs_shape = surprise_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            surprise_action_shape = surprise_session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
            for (auto& d : surprise_obs_shape) if (d <= 0) d = 1;
            for (auto& d : surprise_action_shape) if (d <= 0) d = 1;
        }
        else if (surprise_input_names.size() == 1)
        {
            surprise_concat_obs_action_input = true;
            surprise_obs_shape = surprise_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            surprise_action_shape.clear();
            for (auto& d : surprise_obs_shape) if (d <= 0) d = 1;
            const int64_t expected = static_cast<int64_t>(obs_mean.size() + action_mean.size());
            int64_t last = 1;
            if (!surprise_obs_shape.empty()) last = surprise_obs_shape.back();
            if (last != expected)
            {
                spdlog::error(
                    "Surprise ONNX single-input last dim {} != obs_dim + action_dim ({} + {} = {}).",
                    last,
                    obs_mean.size(),
                    action_mean.size(),
                    expected
                );
                return false;
            }
            spdlog::info("Surprise ONNX: using single concatenated input (obs, then action), dim {}.", expected);
        }
        else
        {
            spdlog::error("Surprise ONNX must have 1 input (concat obs+action) or 2 inputs (obs, action).");
            return false;
        }

        surprise_output_name_storage.clear();
        surprise_output_names.clear();
        for (size_t i = 0; i < surprise_session->GetOutputCount(); ++i)
        {
            auto output_name = surprise_session->GetOutputNameAllocated(i, surprise_allocator);
            surprise_output_name_storage.push_back(output_name.get());
        }
        for (auto& s : surprise_output_name_storage) surprise_output_names.push_back(s.c_str());

        if (surprise_output_names.size() == 1)
        {
            surprise_outputs_single_tensor = true;
            surprise_output_shape = surprise_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        }
        else if (surprise_output_names.size() == 2)
        {
            surprise_outputs_single_tensor = false;
            surprise_output_shape = surprise_session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            surprise_output_shape_alt = surprise_session->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
        }
        else
        {
            spdlog::error("Surprise ONNX must have 1 or 2 outputs.");
            return false;
        }
    }
    catch (const std::exception& e)
    {
        spdlog::error("Failed to load surprise ONNX '{}': {}", model_path.string(), e.what());
        return false;
    }
    return true;
}

void State_RLBase::update_surprise_gate(const std::vector<float>& obs, const std::vector<float>& action)
{
    if (!has_prev_transition)
    {
        prev_obs = obs;
        prev_action = action;
        has_prev_transition = true;
        return;
    }

    if (obs.size() != obs_mean.size() || prev_obs.size() != obs_mean.size() ||
        action.size() != action_mean.size() || prev_action.size() != action_mean.size())
    {
        if (surprise_step_count == 0)
        {
            spdlog::warn(
                "Surprise dimensions mismatch (obs: {}, obs_mean: {}, action: {}, action_mean: {}).",
                obs.size(), obs_mean.size(), action.size(), action_mean.size()
            );
        }
        prev_obs = obs;
        prev_action = action;
        return;
    }

    std::vector<float> obs_norm(prev_obs.size());
    std::vector<float> action_norm(prev_action.size());
    std::vector<float> next_obs_norm(obs.size());
    for (size_t i = 0; i < obs_norm.size(); ++i)
    {
        obs_norm[i] = (prev_obs[i] - obs_mean[i]) / obs_std[i];
        next_obs_norm[i] = (obs[i] - obs_mean[i]) / obs_std[i];
    }
    for (size_t i = 0; i < action_norm.size(); ++i)
    {
        action_norm[i] = (prev_action[i] - action_mean[i]) / action_std[i];
    }

    float surprise = 0.0f;
    try
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        std::vector<Ort::Value> inputs;
        if (surprise_concat_obs_action_input)
        {
            surprise_concat_buffer_.resize(obs_norm.size() + action_norm.size());
            std::memcpy(
                surprise_concat_buffer_.data(),
                obs_norm.data(),
                obs_norm.size() * sizeof(float)
            );
            std::memcpy(
                surprise_concat_buffer_.data() + obs_norm.size(),
                action_norm.data(),
                action_norm.size() * sizeof(float)
            );
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                surprise_concat_buffer_.data(),
                surprise_concat_buffer_.size(),
                surprise_obs_shape.data(),
                surprise_obs_shape.size()
            ));
        }
        else
        {
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info, obs_norm.data(), obs_norm.size(), surprise_obs_shape.data(), surprise_obs_shape.size()
            ));
            inputs.push_back(Ort::Value::CreateTensor<float>(
                memory_info,
                action_norm.data(),
                action_norm.size(),
                surprise_action_shape.data(),
                surprise_action_shape.size()
            ));
        }

        auto outputs = surprise_session->Run(
            Ort::RunOptions{nullptr},
            surprise_input_names.data(),
            inputs.data(),
            inputs.size(),
            surprise_output_names.data(),
            surprise_output_names.size()
        );

        const float* mean_ptr = nullptr;
        const float* logvar_ptr = nullptr;
        if (surprise_outputs_single_tensor)
        {
            const float* combined = outputs[0].GetTensorData<float>();
            const size_t obs_dim = next_obs_norm.size();
            mean_ptr = combined;
            logvar_ptr = combined + obs_dim;
        }
        else
        {
            mean_ptr = outputs[0].GetTensorData<float>();
            logvar_ptr = outputs[1].GetTensorData<float>();
        }

        for (size_t i = 0; i < next_obs_norm.size(); ++i)
        {
            const float logvar = std::max(-10.0f, std::min(2.0f, logvar_ptr[i]));
            const float var = std::exp(logvar);
            const float diff = next_obs_norm[i] - mean_ptr[i];
            surprise += 0.5f * (logvar + (diff * diff) / var);
        }
    }
    catch (const std::exception& e)
    {
        if (surprise_step_count == 0)
        {
            spdlog::error("Surprise inference failed: {}", e.what());
        }
        prev_obs = obs;
        prev_action = action;
        return;
    }

    if (surprise_step_count == 0)
    {
        surprise_ema = surprise;
    }
    else
    {
        surprise_ema = surprise_ema_alpha * surprise + (1.0f - surprise_ema_alpha) * surprise_ema;
    }
    surprise_step_count += 1;

    if (surprise_log_every_steps > 0 && (surprise_step_count % surprise_log_every_steps) == 0)
    {
        spdlog::info(
            "Surprise stats: raw={:.3f}, ema={:.3f}, threshold={:.3f}, step={}",
            surprise,
            surprise_ema,
            surprise_threshold,
            surprise_step_count
        );
    }

    const float gate_value = surprise_use_ema ? surprise_ema : surprise;
    if (surprise_step_count >= surprise_min_steps && gate_value > surprise_threshold)
    {
        surprise_over_threshold_count += 1;
        if (surprise_over_threshold_count >= surprise_consecutive_steps && !surprise_trip.load())
        {
            surprise_trip.store(true);
            spdlog::warn(
                "Surprise tripped. raw={:.3f}, ema={:.3f}, threshold={:.3f}, gate={:.3f}. Switching to Stabilize.",
                surprise,
                surprise_ema,
                surprise_threshold,
                gate_value
            );
        }
    }
    else
    {
        surprise_over_threshold_count = 0;
    }

    prev_obs = obs;
    prev_action = action;
}

bool State_RLBase::surprise_trip_check()
{
    return surprise_enabled.load() && surprise_trip.load();
}

void State_RLBase::run()
{
    // During the intro phase, command joints directly from the interpolated
    // keyframes (motor-index-ordered) and ignore the policy output.
    if (intro_running.load())
    {
        const auto q = intro_current_q();
        if (!q.empty())
        {
            for (size_t i = 0; i < q.size(); ++i)
            {
                lowcmd->msg_.motor_cmd()[i].q() = q[i];
            }
            return;
        }
    }

    std::vector<float> action;
    {
        std::lock_guard<std::mutex> lock(action_mutex);
        action = latest_action;
    }
    // EXPERIMENTAL (2026-04-18): previously, an empty ``latest_action`` here
    // fell back to ``action_manager->processed_actions()``. On the very first
    // FSM tick after ``enter()``, if the policy worker thread had not yet run
    // ``process_action(zero_raw)``, the ActionManager's internal
    // ``_processed_actions`` vector is still zero-initialised from its
    // constructor. The old fallback then wrote q=0 to every motor, which at
    // kp=25 with the Go2 in a quadruped stand produces a huge knee-extension
    // torque spike (25 * 1.5 ≈ 37 N·m on each calf) and flips the robot
    // backward. Holding the previous q_cmd instead — which ``enter()`` leaves
    // untouched from the previous state (e.g. FixStand's held stance) — is
    // safe regardless of who has or hasn't initialised what.
    if (action.empty())
    {
        if (run_warning_count_ < 5)
        {
            spdlog::warn(
                "State '{}' run(): latest_action empty; holding previous q_cmd this tick.",
                getStateString());
            run_warning_count_ += 1;
        }
        return;
    }
    if (action.size() < env->robot->data.joint_ids_map.size())
    {
        if (run_warning_count_ < 5)
        {
            spdlog::warn(
                "State '{}' run(): action size {} < joint_ids_map {} — skipping q_cmd (check deploy / ONNX).",
                getStateString(),
                action.size(),
                env->robot->data.joint_ids_map.size());
            run_warning_count_ += 1;
        }
        return;
    }
    for (int i(0); i < env->robot->data.joint_ids_map.size(); i++)
    {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}