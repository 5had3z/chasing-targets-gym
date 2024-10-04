#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <ranges>
#include <span>

namespace py = pybind11;

int64_t plan_ahead_steps = 10;
float forward_weight = 12;
float obstacle_weight = 6666;
float max_acceleration = 0.4;

using FloatArray2 = py::detail::unchecked_reference<float, 2>;

struct Robot
{
    float x;
    float y;
    float t;
    float dx;
    float dy;
    float dt;
    float vL;
    float vR;
};

using Robots = std::vector<Robot>;
using RobotsView = std::span<Robot>;
using ConstRobotsView = std::span<const Robot>;

struct Action
{
    float vL;
    float vR;
};

using Actions = std::vector<Action>;
using ActionsView = std::span<Action>;

RobotsView asRobots(py::array_t<float> arr)
{
    RobotsView view(reinterpret_cast<Robot*>(arr.mutable_data()), static_cast<std::size_t>(arr.shape(0)));
    return view;
}

ConstRobotsView asConstRobots(py::array_t<float> arr)
{
    ConstRobotsView view(reinterpret_cast<const Robot*>(arr.data()), static_cast<std::size_t>(arr.shape(0)));
    return view;
}

class Planner
{
public:
    Planner(double agent_radius, double dt, double max_velocity)
        : mAgentRad{static_cast<float>(agent_radius)}
        , mDt{static_cast<float>(dt)}
        , mMaxVel{static_cast<float>(max_velocity)}
        , mTau{static_cast<float>(dt * plan_ahead_steps)} {};

    py::dict operator()(py::dict obs)
    {
        auto vLCurrent = obs["vL"].cast<py::array_t<float>>().unchecked<1>();
        auto vRCurrent = obs["vR"].cast<py::array_t<float>>().unchecked<1>();

        auto robotsCurrent = asConstRobots(obs["robots"].cast<py::array_t<float>>());
        auto robotsFuture = asConstRobots(obs["future_robot"].cast<py::array_t<float>>());
        auto futureTarget = obs["future_target"].cast<py::array_t<float>>().unchecked<2>();
        const auto nRobots = robotsCurrent.size();

        auto vLAction = py::array_t<float>(nRobots);
        auto vLResult = vLAction.mutable_unchecked<1>();
        auto vRAction = py::array_t<float>(nRobots);
        auto vRResult = vRAction.mutable_unchecked<1>();
        for (std::size_t rIdx = 0; rIdx < nRobots; ++rIdx)
        {
            std::tie(vLResult[rIdx], vRResult[rIdx])
                = chooseAction(vLCurrent[rIdx], vRCurrent[rIdx], robotsCurrent, robotsFuture, futureTarget, rIdx);
        }

        py::dict actions;
        actions["vL"] = vLAction;
        actions["vR"] = vRAction;

        return actions;
    }

private:
    Actions makeActions(float vL, float vR) const noexcept
    {
        Actions actions;
        actions.reserve(9);

        const std::array dv{-mDt * max_acceleration, 0.f, mDt * max_acceleration};
        for (auto L : dv)
        {
            for (auto R : dv)
            {
                Action a{vL + L, vR + R};
                if (-mMaxVel < a.vL && a.vL < mMaxVel && -mMaxVel < a.vR && a.vR < mMaxVel)
                {
                    actions.emplace_back(std::move(a));
                }
            }
        }

        return actions;
    }

    Robots predictPosition(ActionsView actions, const Robot& robot) const
    {
        Robots newRobotPos(actions.size(), robot);
        for (std::size_t idx = 0; idx < actions.size(); ++idx)
        {
            float dx, dy;
            const auto& action = actions[idx];
            const auto vDiff = action.vR - action.vL;
            if (std::abs(vDiff) < 1e-3) // Straight motion
            {
                dx = action.vL * std::cos(robot.t);
                dy = action.vL * std::sin(robot.t);
            }
            else // Turning motion
            {
                const auto R = mAgentRad * (action.vR + action.vL) / (vDiff + std::numeric_limits<float>::epsilon());
                const auto new_t = vDiff / (mAgentRad * 2.f) + robot.t;
                dx = R * (std::sin(new_t) - std::sin(robot.t));
                dy = -R * (std::cos(new_t) - std::cos(robot.t));
            }
            newRobotPos[idx].x += mTau * dx;
            newRobotPos[idx].y += mTau * dy;
        }
        return newRobotPos;
    }

    std::pair<float, float> chooseAction(
        float vL, float vR, ConstRobotsView robots, ConstRobotsView robotsFut, FloatArray2 target, std::size_t robotIdx)
    {
        auto actions = makeActions(vL, vR);
        auto newRobotPos = predictPosition(actions, robots[robotIdx]);
        return {-1, -1};
    }

    float mAgentRad;
    float mDt;
    float mMaxVel;
    float mTau;
};

PYBIND11_MODULE(_planner, m)
{
    py::class_<Planner>(m, "Planner").def(py::init<double, double, double>()).def("__call__", &Planner::operator());
}
