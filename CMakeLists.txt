cmake_minimum_required(VERSION 3.15)
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

python_add_library(_planner MODULE src/planner.cpp WITH_SOABI)
target_link_libraries(_planner PRIVATE pybind11::headers)
target_compile_features(_planner PRIVATE cxx_std_20)

install(TARGETS _planner DESTINATION chasing_targets_gym)
