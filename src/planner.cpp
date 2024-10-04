#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <ranges>

namespace py = pybind11;

int64_t plan_ahead_steps = 10;
float forward_weight = 12;
float obstacle_weight = 6666;
float max_acceleration = 0.4;

using FloatArray = py::detail::unchecked_reference<float, 2>;

class Planner
{
public:
    Planner(double agent_radius, double dt, double max_velocity)
        : mAgentRad{static_cast<float>(agent_radius)}
        , mDt{static_cast<float>(dt)}
        , mMaxVel{static_cast<float>(max_velocity)} {};

    py::dict operator()(py::dict obs)
    {
        auto vLCurrent = obs["vL"].cast<py::array_t<float>>().unchecked<1>();
        auto vRCurrent = obs["vR"].cast<py::array_t<float>>().unchecked<1>();

        auto robotsCurrent = obs["robots"].cast<py::array_t<float>>().unchecked<2>();
        auto robotsFuture = obs["future_robot"].cast<py::array_t<float>>().unchecked<2>();
        auto futureTarget = obs["future_target"].cast<py::array_t<float>>().unchecked<2>();
        const auto nRobots = robotsCurrent.shape(0);

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
    py::array_t<float> makeActions(float vL, float vR) const noexcept
    {
        py::array_t<float> actions({9, 2});
        auto actionPtr = actions.mutable_data();

        const std::array dv{-mDt * max_acceleration, 0.f, mDt * max_acceleration};
        for (auto L : dv)
        {
            for (auto R : dv)
            {
                *actionPtr++ = vL + L;
                *actionPtr++ = vR + R;
            }
        }

        return actions;
    }

    std::pair<float, float> chooseAction(
        float vL, float vR, FloatArray robots, FloatArray robotsFut, FloatArray target, std::size_t robotIdx)
    {
        py::array_t<float> actions = makeActions(vL, vR);
        return {-1, -1};
    }

    float mAgentRad;
    float mDt;
    float mMaxVel;
};

PYBIND11_MODULE(_planner, m)
{
    py::class_<Planner>(m, "Planner").def(py::init<double, double, double>()).def("__call__", &Planner::operator());
}
