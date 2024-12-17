#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <iostream>
#include <ranges>
#include <span>

namespace py = pybind11;

int64_t plan_ahead_steps = 10;
float forward_weight = 12;
float obstacle_weight = 10'000;
float max_acceleration = 0.4;

using FloatArray2D = py::detail::unchecked_reference<float, 2>;

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
    FloatArray2D _data;

    auto x() const noexcept -> std::span<const float>
    {
        return this->_get_span(0);
    }

    auto y() const noexcept -> std::span<const float>
    {
        return this->_get_span(1);
    }

    auto t() const noexcept -> std::span<const float>
    {
        return this->_get_span(2);
    }

    auto dx() const noexcept -> std::span<const float>
    {
        return this->_get_span(3);
    }

    auto dy() const noexcept -> std::span<const float>
    {
        return this->_get_span(4);
    }

    auto dt() const noexcept -> std::span<const float>
    {
        return this->_get_span(5);
    }

    auto operator[](std::size_t idx) const noexcept -> Robot
    {
        return Robot(_data(0, idx), _data(1, idx), _data(2, idx), _data(3, idx), _data(4, idx), _data(5, idx));
    }

    std::size_t n_robots() const noexcept
    {
        return _data.shape(1);
    }

private:
    auto _get_span(std::size_t index) const noexcept -> std::span<const float>
    {
        return std::span(_data.data(index, 0), this->n_robots());
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

struct Pos
{
    float x;
    float y;
};

using Actions = std::vector<Action>;
using ActionsView = std::span<Action>;
using ConstActionsView = std::span<const Action>;

template <typename T>
constexpr auto l2_distance(T a, T b) noexcept -> T
{
    return std::sqrt(std::pow(a, 2) + std::pow(b, 2));
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

        auto robotsCurrent = RobotArrayView(obs["current_robot"].cast<py::array_t<float>>().unchecked<2>());
        auto robotsFuture = RobotArrayView(obs["future_robot"].cast<py::array_t<float>>().unchecked<2>());
        auto robotTargetIdx = obs["robot_target_idx"].cast<py::array_t<int64_t>>().unchecked<1>();
        auto futureTargets = obs["future_target"].cast<py::array_t<float>>().unchecked<2>();

        const auto nRobots = robotsCurrent.n_robots();
        auto vLAction = py::array_t<float>(nRobots);
        auto vLResult = vLAction.mutable_unchecked<1>();
        auto vRAction = py::array_t<float>(nRobots);
        auto vRResult = vRAction.mutable_unchecked<1>();
        for (std::size_t rIdx = 0; rIdx < nRobots; ++rIdx)
        {
            std::array<float, 2> futureTarget;
            futureTarget[0] = futureTargets(0, robotTargetIdx[rIdx]);
            futureTarget[1] = futureTargets(1, robotTargetIdx[rIdx]);
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

    std::vector<Pos> predictPosition(ConstActionsView actions, const Robot& robot) const
    {
        std::vector<Pos> newRobots(actions.size());
        std::ranges::transform(actions, newRobots.begin(),
            [&](const Action& a)
            {
                float dx, dy;
                const auto vDiff = a.vR - a.vL;
                if (std::abs(vDiff) < 1e-3f) // Straight motion
                {
                    dx = a.vL * std::cos(robot.t);
                    dy = a.vL * std::sin(robot.t);
                }
                else // Turning motion
                {
                    const auto R = mAgentRad * (a.vR + a.vL) / vDiff;
                    const auto new_t = vDiff / (mAgentRad * 2.f) + robot.t;
                    dx = R * (std::sin(new_t) - std::sin(robot.t));
                    dy = -R * (std::cos(new_t) - std::cos(robot.t));
                }
                return Pos{robot.x + mTau * dx, robot.y + mTau * dy};
            });
        return newRobots;
    }

    float closestObstacleDistance(const Pos& robot, const RobotArrayView& obstacles, std::size_t robotIdx)
    {
        std::vector<float> distances(obstacles.n_robots());
        std::transform(obstacles.x().begin(), obstacles.x().end(), obstacles.y().begin(), distances.begin(),
            [&](float x, float y) { return l2_distance(x - robot.x, y - robot.y); });
        distances[robotIdx] = std::numeric_limits<float>::max();
        return std::ranges::min(distances);
    }

    std::pair<float, float> chooseAction(float vL, float vR, const Robot& robot, const RobotArrayView& robotsFut,
        std::span<const float, 2> target, std::size_t robotIdx)
    {
        const auto actions = makeActions(vL, vR);
        const auto newRobotPos = predictPosition(actions, robot);

        auto targetDist = [&target](const Pos& p) { return l2_distance(p.x - target[0], p.y - target[1]); };

        const float prevTargetDist = targetDist(Pos{robot.x, robot.y});
        std::vector<float> distScore(newRobotPos.size());
        std::ranges::transform(newRobotPos, distScore.begin(),
            [&](const Pos& r) { return forward_weight * (prevTargetDist - targetDist(r)); });

        std::vector<float> obstacleCost(newRobotPos.size());
        std::ranges::transform(newRobotPos, obstacleCost.begin(),
            [&](const Pos& r)
            {
                const float distanceToObstacle = closestObstacleDistance(r, robotsFut, robotIdx);
                if (distanceToObstacle < 4 * mAgentRad)
                {
                    return obstacle_weight * (4 * mAgentRad - distanceToObstacle);
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

struct Boundary
{
    float minX;
    float minY;
    float maxX;
    float maxY;
};

/**
 * @brief Perform the moving algorithm on one dimension
 *
 * @param pos range of target positions
 * @param vel range of target velocities
 * @param dt timestep length
 * @param min_val min boundary
 * @param max_val max boundary
 * @param nSteps number of timesteps to iterate
 */
void inplaceMoveImpl(
    std::span<float> pos, std::span<float> vel, double dt, float min_val, float max_val, int64_t nSteps)
{
    for (std::size_t step = 0; step < nSteps; ++step)
    {
        for (std::size_t idx = 0; idx < pos.size(); ++idx)
        {
            auto& p = pos[idx];
            auto& v = vel[idx];
            p += v * dt;
            if (p < min_val)
            {
                p = min_val;
                v *= -1;
            }
            else if (p > max_val)
            {
                p = max_val;
                v *= -1;
            }
        }
    }
}

/**
 * @brief Inplace move the targets a number of timesteps into the future
 *
 * @param targets Array of targets of shape [[x,y,vx,vy], n_targets]
 * @param dt Timestep size
 * @param limits Limits of the arena to bounce off
 * @param nSteps Number of steps to move into the future
 */
void inplaceMoveTargets(py::array_t<float> targets, double dt, py::array_t<float> limits, int64_t nSteps)
{
    if (limits.ndim() != 1 && limits.shape(0) != 4)
    {
        throw std::runtime_error("Unexpected limits shape for inplaceMoveTargets");
    }
    if (targets.ndim() != 2 && targets.shape(0) != 4)
    {
        throw std::runtime_error("Unexpected targets shape or stride for inplaceMoveTargets");
    }

    const auto boundary = *reinterpret_cast<const Boundary*>(limits.data());
    auto targetsView = targets.mutable_unchecked<2>();
    const auto nTargets = targetsView.shape(1);

    inplaceMoveImpl(std::span(targetsView.mutable_data(0, 0), nTargets),
        std::span(targetsView.mutable_data(2, 0), nTargets), dt, boundary.minX, boundary.maxX, nSteps);

    inplaceMoveImpl(std::span(targetsView.mutable_data(1, 0), nTargets),
        std::span(targetsView.mutable_data(3, 0), nTargets), dt, boundary.minY, boundary.maxY, nSteps);
}

PYBIND11_MODULE(_planner, m)
{
    py::class_<Planner>(m, "Planner")
        .def(py::init<double, double, double>(), py::arg("agent_radius"), py::arg("dt"), py::arg("max_velocity"))
        .def("__call__", &Planner::operator());

    m.def("inplace_move_targets", &inplaceMoveTargets, py::arg("targets"), py::arg("dt"), py::arg("limits"),
        py::arg("n_steps"));
}
