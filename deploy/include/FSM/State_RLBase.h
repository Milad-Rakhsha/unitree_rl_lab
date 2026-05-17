// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

// RL policy state: ManagerBasedRLEnv + ONNX policy on a worker thread at
// step_dt; FSM thread applies processed joint targets at ~1 kHz. Supports
// optional intro keyframes, auto_transition, surprise gate, and orientation
// fall checks. See deploy/README.md.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "onnxruntime_cxx_api.h"
#include <atomic>
#include <fstream>
#include <mutex>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    void enter();

    /// Sync articulation from ``lowstate`` every FSM tick so fall checks and obs use current IMU.
    void pre_run() override;
    void run();
    void exit();

private:
    void policy_loop();
    bool parse_surprise_cfg(const YAML::Node& cfg);
    bool setup_surprise_model(const std::filesystem::path& model_path);
    void update_surprise_gate(const std::vector<float>& obs, const std::vector<float>& action);
    bool surprise_trip_check();

    void parse_intro_cfg(const YAML::Node& cfg);
    void parse_auto_transition_cfg(const YAML::Node& cfg);
    bool intro_active() const;
    std::vector<float> intro_current_q();

    // CSV trajectory logging for sim-to-sim and deploy diffing. Enabled per
    // state via a YAML ``csv_log: { enabled: true, path: ... }`` block. The
    // schema matches ``scripts/rsl_rl/play_deploy_fsm.py`` so rows can be
    // diffed directly. ``base_v*_b`` columns are written as ``nan`` because
    // deploy has no base-linear-velocity estimate.
    //
    // Semantic caveat: during the intro phase the ``action_*`` columns hold
    // the raw ONNX output (which ``run()`` discards in favour of the intro
    // ramp), whereas the IIL CSV logs ``(target - offset) / scale``. The
    // other columns (qpos/qvel/base pose/cmd_v*) are apples-to-apples from t=0.
    void parse_csv_log_cfg(const YAML::Node& cfg);
    void csv_open();
    void csv_close();
    void csv_write_row(double wall_s, double intro_elapsed_s, const std::vector<float>& action_raw);

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};

    std::mutex action_mutex;
    std::vector<float> latest_action;

    /// Rate-limit counter for actionable warnings raised inside run() (empty
    /// latest_action, action size < joint_ids_map). Reset in enter().
    int run_warning_count_ = 0;

    // Optional intro keyframe phase: on enter(), linearly interpolate joint
    // positions through (ts_intro, qs_intro) before handing off to the learned
    // policy. Useful to bring the robot from its current (e.g. FixStand) pose
    // into a stance matching the policy's training distribution.
    std::vector<float> ts_intro;
    std::vector<std::vector<float>> qs_intro;
    std::vector<float> intro_kp;
    std::vector<float> intro_kd;
    std::atomic<bool> intro_running{false};
    double intro_t0_s = 0.0;
    std::mutex intro_mutex;

    // Generic auto-transition: after this state has been active for
    // ``auto_transition_duration_s`` wall seconds, switch to the FSM named by
    // ``auto_transition_target_``. Replaces the hardcoded Stabilize-only logic.
    bool auto_transition_enabled_ = false;
    std::string auto_transition_target_;
    float auto_transition_duration_s_ = 0.0f;
    double state_enter_wall_time_s_ = 0.0;

    std::atomic<bool> surprise_enabled{false};
    std::atomic<bool> surprise_trip{false};
    float surprise_threshold = 120.0f;
    float surprise_ema_alpha = 0.1f;
    float surprise_ema = 0.0f;
    int surprise_min_steps = 25;
    int surprise_consecutive_steps = 3;
    bool surprise_use_ema = true;
    int surprise_log_every_steps = 50;
    int surprise_over_threshold_count = 0;
    int surprise_step_count = 0;

    std::vector<float> obs_mean;
    std::vector<float> obs_std;
    std::vector<float> action_mean;
    std::vector<float> action_std;
    std::vector<float> prev_obs;
    std::vector<float> prev_action;
    bool has_prev_transition = false;

    // Name of the FSM to switch to when surprise trips. Configurable per state
    // via ``surprise.target_fsm``; falls back to "Stabilize" then "FixStand".
    std::string surprise_target_fsm_;

    bool csv_log_enabled_ = false;
    std::string csv_log_path_cfg_;
    std::ofstream csv_file_;

    Ort::Env surprise_ort_env{ORT_LOGGING_LEVEL_WARNING, "surprise_model"};
    Ort::SessionOptions surprise_session_options;
    std::unique_ptr<Ort::Session> surprise_session;
    Ort::AllocatorWithDefaultOptions surprise_allocator;
    std::vector<std::string> surprise_input_name_storage;
    std::vector<const char*> surprise_input_names;
    std::vector<std::string> surprise_output_name_storage;
    std::vector<const char*> surprise_output_names;
    std::vector<int64_t> surprise_obs_shape;
    std::vector<int64_t> surprise_action_shape;
    std::vector<int64_t> surprise_output_shape;
    std::vector<int64_t> surprise_output_shape_alt;
    bool surprise_outputs_single_tensor = false;
    /// When true, ONNX has one input tensor ``[batch, obs_dim + action_dim]`` (obs then action).
    bool surprise_concat_obs_action_input = false;
    std::vector<float> surprise_concat_buffer_;
};

REGISTER_FSM(State_RLBase)
