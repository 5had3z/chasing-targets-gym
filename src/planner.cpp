#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <iostream>
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
};

struct RobotArrayView
{
    FloatArray2 _data;
    const Robot& operator[](std::size_t idx) const noexcept
    {
        return *reinterpret_cast<const Robot*>(_data.data(idx, 0));
    }

    std::size_t size() const noexcept
    {
        return _data.shape(0);
    }
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
using ConstActionsView = std::span<const Action>;

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

        auto robotsCurrent = RobotArrayView(obs["current_robot"].cast<py::array_t<float>>().unchecked<2>());
        auto robotsFuture = RobotArrayView(obs["future_robot"].cast<py::array_t<float>>().unchecked<2>());
        auto robotTargetIdx = obs["robot_target_idx"].cast<py::array_t<int64_t>>().unchecked<1>();
        auto futureTargets = obs["future_target"].cast<py::array_t<float>>().unchecked<2>();

        const auto nRobots = robotsCurrent.size();
        auto vLAction = py::array_t<float>(nRobots);
        auto vLResult = vLAction.mutable_unchecked<1>();
        auto vRAction = py::array_t<float>(nRobots);
        auto vRResult = vRAction.mutable_unchecked<1>();
        for (std::size_t rIdx = 0; rIdx < nRobots; ++rIdx)
        {
            std::array<float, 2> futureTarget;
            futureTarget[0] = futureTargets(robotTargetIdx[rIdx], 0);
            futureTarget[1] = futureTargets(robotTargetIdx[rIdx], 1);
            std::tie(vLResult[rIdx], vRResult[rIdx])
                = chooseAction(vLCurrent[rIdx], vRCurrent[rIdx], robotsCurrent[rIdx], robotsFuture, futureTarget, rIdx);
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
                Action a = {.vL = vL + L, .vR = vR + R};
                if (-mMaxVel < a.vL && a.vL < mMaxVel && -mMaxVel < a.vR && a.vR < mMaxVel)
                {
                    actions.emplace_back(std::move(a));
                }
            }
        }

        return actions;
    }

    Robots predictPosition(ConstActionsView actions, const Robot& robot) const
    {
        Robots newRobots(actions.size(), robot);
        for (std::size_t idx = 0; idx < actions.size(); ++idx)
        {
            float dx, dy;
            const auto& action = actions[idx];
            const auto vDiff = action.vR - action.vL;
            if (std::abs(vDiff) < 1e-3f) // Straight motion
            {
                dx = action.vL * std::cos(robot.t);
                dy = action.vL * std::sin(robot.t);
            }
            else // Turning motion
            {
                const auto R = mAgentRad * (action.vR + action.vL) / vDiff;
                const auto new_t = vDiff / (mAgentRad * 2.f) + robot.t;
                dx = R * (std::sin(new_t) - std::sin(robot.t));
                dy = -R * (std::cos(new_t) - std::cos(robot.t));
            }
            newRobots[idx].x += mTau * dx;
            newRobots[idx].y += mTau * dy;
        }

        return newRobots;
    }

    float closestObstacleDistance(const Robot& robot, RobotArrayView obstacles, std::size_t robotIdx)
    {
        float minDist = std::numeric_limits<float>::max();
        for (std::size_t idx = 0; idx < obstacles.size(); ++idx)
        {
            if (idx == robotIdx)
            {
                continue;
            }
            const float dist
                = std::sqrt(std::pow(obstacles[idx].x - robot.x, 2.f) + std::pow(obstacles[idx].y - robot.y, 2.f));
            minDist = std::min(minDist, dist);
        }
        return minDist - mAgentRad * 2.f;
    }

    std::pair<float, float> chooseAction(float vL, float vR, const Robot& robot, RobotArrayView robotsFut,
        std::span<const float, 2> target, std::size_t robotIdx)
    {
        const auto actions = makeActions(vL, vR);
        const auto newRobotPos = predictPosition(actions, robot);

        auto targetDist = [&target](const Robot& r)
        { return std::sqrt(std::pow(r.x - target[0], 2.f) + std::pow(r.y - target[1], 2.f)); };

        const float prevTargetDist = targetDist(robot);
        std::vector<float> distScore(newRobotPos.size());
        std::ranges::transform(newRobotPos, distScore.begin(),
            [&](const Robot& r) { return forward_weight * (prevTargetDist - targetDist(r)); });

        std::vector<float> obstacleCost(newRobotPos.size());
        std::ranges::transform(newRobotPos, obstacleCost.begin(),
            [&](const Robot& r)
            {
                const float distanceToObstacle = closestObstacleDistance(r, robotsFut, robotIdx);
                if (distanceToObstacle < mAgentRad)
                {
                    return obstacle_weight * (mAgentRad - distanceToObstacle);
                }
                return 0.f;
            });

        auto maxScore = std::numeric_limits<float>::lowest();
        std::size_t argmax = 0;
        for (std::size_t idx = 0; idx < actions.size(); ++idx)
        {
            const auto score = distScore[idx] - obstacleCost[idx];
            if (score > maxScore)
            {
                argmax = idx;
                maxScore = score;
            }
        }

        return {actions[argmax].vL, actions[argmax].vR};
    }

    float mAgentRad;
    float mDt;
    float mMaxVel;
    float mTau;
};

struct Target
{
    float x;
    float y;
    float vx;
    float vy;
};

struct Boundary
{
    float minX;
    float minY;
    float maxX;
    float maxY;
};

void inplaceMoveTargets(py::array_t<float> targets, double dt, py::array_t<float> limits, int64_t nSteps)
{
    if (limits.ndim() != 1 && limits.shape(0) != 4)
    {
        throw std::runtime_error("Unexpected limits shape for inplaceMoveTargets");
    }
    if (targets.ndim() != 2 && targets.shape(1) != 4 && targets.strides(0) == 4 * sizeof(float))
    {
        throw std::runtime_error("Unexpected targets shape or stride for inplaceMoveTargets");
    }

    auto boundary = *reinterpret_cast<const Boundary*>(limits.data());
    auto targetsView = targets.mutable_unchecked<2>();
    for (std::size_t tIdx = 0; tIdx < targetsView.shape(0); ++tIdx)
    {
        auto target = reinterpret_cast<Target*>(targetsView.mutable_data(tIdx, 0));
        for (std::size_t step = 0; step < nSteps; ++step)
        {
            target->x += target->vx * dt;
            target->y += target->vy * dt;
            if (target->x < boundary.minX)
            {
                target->x = boundary.minX;
                target->vx *= -1;
            }
            else if (target->x > boundary.maxX)
            {
                target->x = boundary.maxX;
                target->vx *= -1;
            }

            if (target->y < boundary.minY)
            {
                target->y = boundary.minY;
                target->vy *= -1;
            }
            else if (target->y > boundary.maxY)
            {
                target->y = boundary.maxY;
                target->vy *= -1;
            }
        }
    }
}

PYBIND11_MODULE(_planner, m)
{
    py::class_<Planner>(m, "Planner")
        .def(py::init<double, double, double>(), py::arg("agent_radius"), py::arg("dt"), py::arg("max_velocity"))
        .def("__call__", &Planner::operator());

    m.def("inplace_move_targets", &inplaceMoveTargets, py::arg("targets"), py::arg("dt"), py::arg("limits"),
        py::arg("n_steps"));
}
