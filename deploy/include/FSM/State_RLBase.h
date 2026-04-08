// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "onnxruntime_cxx_api.h"
#include <atomic>
#include <mutex>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    void enter();

    void run();
    void exit();

private:
    void policy_loop();
    bool parse_surprise_cfg(const YAML::Node& cfg);
    bool setup_surprise_model(const std::filesystem::path& model_path);
    void update_surprise_gate(const std::vector<float>& obs, const std::vector<float>& action);
    bool surprise_trip_check();

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};

    std::mutex action_mutex;
    std::vector<float> latest_action;

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

    bool stabilize_auto_velocity_enabled_ = false;
    float stabilize_auto_velocity_duration_s_ = 4.0f;
    double stabilize_enter_wall_time_s_ = 0.0;
};

REGISTER_FSM(State_RLBase)
