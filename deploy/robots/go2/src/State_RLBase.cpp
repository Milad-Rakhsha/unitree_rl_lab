#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include <unitree/common/time/time_tool.hpp>
#include <cmath>

namespace
{
float clamp_std(float v)
{
    return std::max(v, 1.0e-6f);
}
}

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

    parse_surprise_cfg(cfg);

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );

    int surprise_target_fsm = 0;
    if (FSMStringMap.right.count("Stabilize"))
    {
        surprise_target_fsm = FSMStringMap.right.at("Stabilize");
    }
    else if (FSMStringMap.right.count("FixStand"))
    {
        surprise_target_fsm = FSMStringMap.right.at("FixStand");
        spdlog::info("Surprise fallback: Stabilize FSM not found; using FixStand.");
    }

    if (surprise_target_fsm != 0)
    {
        this->registered_checks.emplace_back(
            std::make_pair(
                [&]()->bool{ return surprise_trip_check(); },
                surprise_target_fsm
            )
        );
    }
    else
    {
        spdlog::warn("Surprise enabled but neither Stabilize nor FixStand FSM exists; surprise transition disabled.");
    }

    if (state_string == "Stabilize")
    {
        auto stab_cfg = cfg["auto_velocity"];
        if (stab_cfg && stab_cfg["enabled"].as<bool>(false))
        {
            stabilize_auto_velocity_enabled_ = true;
            stabilize_auto_velocity_duration_s_ = stab_cfg["duration_s"].as<float>(stabilize_auto_velocity_duration_s_);
            if (FSMStringMap.right.count("Velocity"))
            {
                registered_checks.emplace_back(
                    std::make_pair(
                        [this]() -> bool {
                            if (!stabilize_auto_velocity_enabled_ || stabilize_enter_wall_time_s_ <= 0.0)
                            {
                                return false;
                            }
                            const double now = static_cast<double>(unitree::common::GetCurrentTimeMillisecond()) * 1e-3;
                            return (now - stabilize_enter_wall_time_s_) >= static_cast<double>(stabilize_auto_velocity_duration_s_);
                        },
                        FSMStringMap.right.at("Velocity")
                    )
                );
                spdlog::info(
                    "Stabilize auto-return to Velocity after {:.2f} s (wall clock).",
                    stabilize_auto_velocity_duration_s_
                );
            }
            else
            {
                spdlog::warn("Stabilize.auto_velocity enabled but Velocity FSM not found.");
            }
        }
    }
}

void State_RLBase::enter()
{
    surprise_trip.store(false);
    surprise_ema = 0.0f;
    surprise_over_threshold_count = 0;
    surprise_step_count = 0;
    has_prev_transition = false;

    if (getStateString() == "Stabilize" && stabilize_auto_velocity_enabled_)
    {
        stabilize_enter_wall_time_s_ = static_cast<double>(unitree::common::GetCurrentTimeMillisecond()) * 1e-3;
    }
    else
    {
        stabilize_enter_wall_time_s_ = 0.0;
    }

    for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
    {
        lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
        lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
        lowcmd->msg_.motor_cmd()[i].dq() = 0;
        lowcmd->msg_.motor_cmd()[i].tau() = 0;
    }

    env->robot->update();
    policy_thread_running.store(true);
    policy_thread = std::thread([this] { this->policy_loop(); });
}

void State_RLBase::exit()
{
    policy_thread_running.store(false);
    if (policy_thread.joinable())
    {
        policy_thread.join();
    }
}

void State_RLBase::policy_loop()
{
    using clock = std::chrono::high_resolution_clock;
    const std::chrono::duration<double> desired_duration(env->step_dt);
    const auto dt = std::chrono::duration_cast<clock::duration>(desired_duration);
    auto sleep_till = clock::now() + dt;

    env->reset();
    latest_action.assign(env->action_manager->total_action_dim(), 0.0f);

    while (policy_thread_running.load())
    {
        env->episode_length += 1;
        env->robot->update();
        const auto obs_map = env->observation_manager->compute();
        const auto action = env->alg->act(obs_map);
        env->action_manager->process_action(action);

        {
            std::lock_guard<std::mutex> lock(action_mutex);
            latest_action = env->action_manager->processed_actions();
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

        if (surprise_input_names.size() != 2)
        {
            spdlog::error("Surprise ONNX must have exactly 2 inputs (obs, action).");
            return false;
        }

        surprise_obs_shape = surprise_session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        surprise_action_shape = surprise_session->GetInputTypeInfo(1).GetTensorTypeAndShapeInfo().GetShape();
        for (auto& d : surprise_obs_shape) if (d <= 0) d = 1;
        for (auto& d : surprise_action_shape) if (d <= 0) d = 1;

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
        auto obs_tensor = Ort::Value::CreateTensor<float>(
            memory_info, obs_norm.data(), obs_norm.size(), surprise_obs_shape.data(), surprise_obs_shape.size()
        );
        auto action_tensor = Ort::Value::CreateTensor<float>(
            memory_info, action_norm.data(), action_norm.size(), surprise_action_shape.data(), surprise_action_shape.size()
        );
        std::vector<Ort::Value> inputs;
        inputs.push_back(std::move(obs_tensor));
        inputs.push_back(std::move(action_tensor));

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
    std::vector<float> action;
    {
        std::lock_guard<std::mutex> lock(action_mutex);
        action = latest_action;
    }
    if (action.empty())
    {
        action = env->action_manager->processed_actions();
    }
    if (action.size() < env->robot->data.joint_ids_map.size())
    {
        return;
    }
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}