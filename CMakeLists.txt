cmake_minimum_required(VERSION 3.9)
project(LinearRegression)

set(CMAKE_CXX_STANDARD 11)

# root
find_package(ROOT REQUIRED)
include(${ROOT_USE_FILE})

# linear regression lib
include_directories(${PROJECT_SOURCE_DIR}/LinearRegressionLib)
add_subdirectory(LinearRegressionLib)

set(SOURCE_FILES main.cpp LinearRegressionLib/LinearRegressionModel.cpp LinearRegressionLib/LinearRegressionModel.h)
add_executable(LinearRegression ${SOURCE_FILES})

target_link_libraries(LinearRegression ${ROOT_LIBRARIES} dl Minuit)
target_link_libraries(LinearRegression LinearRegressionLib)