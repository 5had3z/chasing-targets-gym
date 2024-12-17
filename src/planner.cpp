#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <iostream>
#include <numbers>
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

    std::vector<Pos> nextPosition(ConstActionsView actions, const Robot& robot) const
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
        const auto newRobotPos = nextPosition(actions, robot);

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

[[nodiscard]] constexpr auto wrapAngle(float th) noexcept -> float
{
    const auto pi = std::numbers::pi_v<float>;
    if (th > pi)
    {
        th -= 2 * pi;
    }
    else if (th < -pi)
    {
        th += 2 * pi;
    }
    return th;
}

enum class ChannelName
{
    x = 0,
    y = 1,
    t = 2,
    dx = 3,
    dy = 4,
    dt = 5,
    vL = 6,
    vR = 7
};

class Robots
{
public:
    float mMaxDv;
    float mDt;
    float mRadius;
    py::array_t<float> mRobots;

    Robots(int64_t n_robots, double radius, double dt, double accel_limit)
        : mMaxDv(static_cast<float>(accel_limit * dt))
        , mDt(static_cast<float>(dt))
        , mRadius(static_cast<float>(radius))
        , mRobots({py::ssize_t(8), py::ssize_t(n_robots)})
    {
    }

    void reset() noexcept
    {
        std::ranges::fill(mRobots.mutable_data(0, 0), mRobots.mutable_data(7, this->size() - 1), 0.f);
    }

    [[nodiscard]] auto size() const noexcept -> py::ssize_t
    {
        return mRobots.shape(1);
    }

    void step(py::dict actions)
    {
        auto updateMotor = [&](const std::string& key, ChannelName ch)
        {
            auto newRef = actions[key.c_str()].cast<py::array_t<float>>().unchecked<1>();
            auto motor = this->view_ch(ch);
            auto updateClipChange
                = [&](float old_, float new_) { return std::clamp(new_, old_ - mMaxDv, old_ + mMaxDv); };
            std::transform(motor.begin(), motor.end(), newRef.data(0), motor.begin(), updateClipChange);
        };

        updateMotor("vL", ChannelName::vL);
        updateMotor("vR", ChannelName::vR);

        // Calculate rate of change
        auto dxdydt = this->calculate_velocity();

        // Update state
        auto updateState = [&](ChannelName ch)
        {
            auto x = this->view_ch(ch);
            std::transform(x.begin(), x.end(), dxdydt.unchecked<2>().data(static_cast<int>(ch), 0), x.begin(),
                [&](float _x, float _dx) { return _x + mDt * _dx; });
        };
        updateState(ChannelName::x);
        updateState(ChannelName::y);
        updateState(ChannelName::t);
        auto t = this->view_ch(ChannelName::t);
        std::transform(t.begin(), t.end(), t.begin(), wrapAngle);
        std::memcpy(mRobots.mutable_data(3, 0), dxdydt.data(0, 0), dxdydt.size() * sizeof(float));
    }

    [[nodiscard]] auto forecast(std::optional<double> dt) -> py::array_t<float>
    {
        if (!dt.has_value())
        {
            dt = mDt;
        }
        // Calculate rate of change
        auto dxdydt = this->calculate_velocity();
        py::array_t<float> pred({py::ssize_t_cast(6), this->size()});

        // Update state
        auto calculateState = [&](ChannelName ch)
        {
            auto x = this->view_ch(ch);
            std::transform(x.begin(), x.end(), dxdydt.unchecked<2>().data(static_cast<int>(ch), 0),
                pred.mutable_unchecked<2>().mutable_data(static_cast<int>(ch), 0),
                [&](float _x, float _dx) { return _x + dt.value() * _dx; });
        };
        calculateState(ChannelName::x);
        calculateState(ChannelName::y);
        calculateState(ChannelName::t);
        auto t = std::span(pred.mutable_data(2, 0), this->size());
        std::transform(t.begin(), t.end(), t.begin(), wrapAngle);

        return pred;
    }

    [[nodiscard]] auto getArray(ChannelName ch) const noexcept -> py::array_t<float>
    {
        py::array_t<float> result({this->size()});
        std::memcpy(result.mutable_data(0), this->view_ch(ch).data(), this->size() * sizeof(float));
        return result;
    }

    void setArray(ChannelName ch, py::array_t<float> data)
    {
        std::memcpy(this->view_ch(ch).data(), data.data(0), data.size() * sizeof(float));
    }

private:
    [[nodiscard]] auto calculate_velocity() -> py::array_t<float>
    {
        auto theta = view_ch(ChannelName::t);
        auto vL = view_ch(ChannelName::vL);
        auto vR = view_ch(ChannelName::vR);
        std::vector<float> vDiff(this->size());
        std::transform(vR.begin(), vR.end(), vL.begin(), vDiff.begin(), std::minus{});

        py::array_t<float> dxdydt({py::ssize_t_cast(3), this->size()});
        auto dx = std::span(dxdydt.mutable_data(0, 0), this->size());
        auto dy = std::span(dxdydt.mutable_data(1, 0), this->size());
        auto dt = std::span(dxdydt.mutable_data(2, 0), this->size());

        // dt = vDiff / robot_width
        std::transform(
            vDiff.begin(), vDiff.end(), dt.begin(), [s = 1.f / (mRadius * 2.f)](float vD) { return vD * s; });

        for (std::size_t idx = 0; idx < this->size(); ++idx)
        {
            if (std::abs(vDiff[idx]) > 1e-3f)
            {
                const float R = mRadius * (vR[idx] + vL[idx]) / (vDiff[idx] + std::numeric_limits<float>::epsilon());
                dx[idx] = R * (std::sin(dt[idx] + theta[idx]) - std::sin(theta[idx]));
                dy[idx] = -R * (std::cos(dt[idx] + theta[idx]) - std::cos(theta[idx]));
            }
            else
            {
                dx[idx] = vL[idx] * std::cos(theta[idx]);
                dy[idx] = vL[idx] * std::sin(theta[idx]);
            }
        }

        return dxdydt;
    }

    [[nodiscard]] auto view_ch(ChannelName index) noexcept -> std::span<float>
    {
        return std::span(mRobots.mutable_data(static_cast<int>(index), 0), this->size());
    }

    [[nodiscard]] auto view_ch(ChannelName index) const noexcept -> std::span<const float>
    {
        return std::span(mRobots.data(static_cast<int>(index), 0), this->size());
    }
};

PYBIND11_MODULE(_planner, m)
{
    py::class_<Planner>(m, "Planner")
        .def(py::init<double, double, double>(), py::arg("agent_radius"), py::arg("dt"), py::arg("max_velocity"))
        .def("__call__", &Planner::operator());

    py::class_<Robots>(m, "Robots")
        .def(py::init<int64_t, double, double, double>(), py::arg("n_robots"), py::arg("radius"), py::arg("dt"),
            py::arg("accel_limit"))
        .def("__len__", &Robots::size)
        .def("reset", &Robots::reset)
        .def("step", &Robots::step)
        .def("forecast", &Robots::forecast)
        .def_readwrite("state", &Robots::mRobots)
        .def_readwrite("dt", &Robots::mDt)
        .def_readwrite("radius", &Robots::mRadius)
        .def_property_readonly("width", [](const Robots& self) { return self.mRadius * 2; })
        .def_property(
            "x", [](const Robots& self) { return self.getArray(ChannelName::x); },
            [](Robots& self, py::array_t<float> x) { self.setArray(ChannelName::x, x); })
        .def_property(
            "y", [](const Robots& self) { return self.getArray(ChannelName::y); },
            [](Robots& self, py::array_t<float> x) { self.setArray(ChannelName::y, x); })
        .def_property(
            "theta", [](const Robots& self) { return self.getArray(ChannelName::t); },
            [](Robots& self, py::array_t<float> x) { self.setArray(ChannelName::t, x); })
        .def_property(
            "vL", [](const Robots& self) { return self.getArray(ChannelName::vL); },
            [](Robots& self, py::array_t<float> x) { self.setArray(ChannelName::vL, x); })
        .def_property(
            "vR", [](const Robots& self) { return self.getArray(ChannelName::vR); },
            [](Robots& self, py::array_t<float> x) { self.setArray(ChannelName::vR, x); });

    m.def("inplace_move_targets", &inplaceMoveTargets, py::arg("targets"), py::arg("dt"), py::arg("limits"),
        py::arg("n_steps"));
}
