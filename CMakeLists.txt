cmake_minimum_required(VERSION 3.16)
project(gtil_bench LANGUAGES C CXX)

if(DEFINED ENV{CONDA_PREFIX})
  set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX};${CMAKE_PREFIX_PATH}")
  message(STATUS "Detected Conda environment, CMAKE_PREFIX_PATH set to: ${CMAKE_PREFIX_PATH}")
else()
  message(STATUS "No Conda environment detected")
endif()

find_package(Treelite REQUIRED)
find_package(xgboost REQUIRED)

add_executable(gtil_bench main.cc)
target_link_libraries(gtil_bench PRIVATE treelite::treelite xgboost::xgboost)

set_target_properties(gtil_bench PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
)
