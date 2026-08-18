// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

// Keyframed stand: interpolates qs over ts with PD gains; optional
// auto_velocity transition to another FSM after the trajectory plus delay.

#pragma once

#include "FSMState.h"
#include "LinearInterpolator.h"

class State_FixStand : public FSMState
{
public:
    State_FixStand(int state, std::string state_string = "FixStand") 
    : FSMState(state, state_string) 
    {
        ts_ = param::config["FSM"]["FixStand"]["ts"].as<std::vector<float>>();
        qs_ = param::config["FSM"]["FixStand"]["qs"].as<std::vector<std::vector<float>>>();
        assert(ts_.size() == qs_.size());

        auto cfg = param::config["FSM"]["FixStand"];
        if (cfg["auto_velocity"] && cfg["auto_velocity"]["enabled"].as<bool>(false))
        {
            auto_velocity_enabled_ = true;
            auto_velocity_delay_s_ = cfg["auto_velocity"]["delay_s"].as<float>(auto_velocity_delay_s_);
            if (FSMStringMap.right.count("Velocity"))
            {
                registered_checks.emplace_back(
                    std::make_pair(
                        [this]()->bool{
                            const double now = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
                            const double t = now - t0_;
                            const double min_t = (ts_.empty() ? 0.0 : ts_.back()) + auto_velocity_delay_s_;
                            return auto_velocity_enabled_ && (t0_ > 0.0) && (t > min_t);
                        },
                        FSMStringMap.right.at("Velocity")
                    )
                );
            }
            else
            {
                spdlog::warn("FixStand.auto_velocity enabled but Velocity FSM not found.");
            }
        }
    }

    void enter()
    {
        // set gain
        static auto kp = param::config["FSM"]["FixStand"]["kp"].as<std::vector<float>>();
        static auto kd = param::config["FSM"]["FixStand"]["kd"].as<std::vector<float>>();
        for(int i(0); i < kp.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp[i];
            motor.kd() = kd[i];
            motor.dq() = motor.tau() = 0;
        }


        // Start interpolation from the measured pose, not the previous command.
        // On Passive -> FixStand the prior command is commonly q=0 while the
        // simulator/hardware is being held in a nominal stand. Interpolating
        // from that stale command folded all legs through an unsupported pose
        // and inverted the base before the RL state was entered.
        std::vector<float> q0;
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            for (int i = 0; i < kp.size(); ++i)
            {
                q0.push_back(lowstate->msg_.motor_state()[i].q());
            }
        }
        qs_[0] = q0;
        t0_ = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3;
    }

    void run()
    {
        float t = (double)unitree::common::GetCurrentTimeMillisecond() * 1e-3 - t0_;
        auto q = linear_interpolate(t, ts_, qs_);
        
        for(int i(0); i < q.size(); ++i) {
            lowcmd->msg_.motor_cmd()[i].q() = q[i];
        }
    }

private:
    double t0_ = 0.0;
    std::vector<float> ts_;
    std::vector<std::vector<float>> qs_;
    bool auto_velocity_enabled_ = false;
    float auto_velocity_delay_s_ = 0.0f;
};

REGISTER_FSM(State_FixStand)