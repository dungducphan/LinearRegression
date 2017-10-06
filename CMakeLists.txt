cmake_minimum_required(VERSION 3.9)
project(LinearRegression)

set(CMAKE_CXX_STANDARD 11)

find_package(ROOT REQUIRED)
include(${ROOT_USE_FILE})

set(SOURCE_FILES main.cpp LinearRegressionLib/LinearRegressionModel.cpp LinearRegressionLib/LinearRegressionModel.h)
add_executable(LinearRegression ${SOURCE_FILES})

target_link_libraries(LinearRegression ${ROOT_LIBRARIES} Core)
target_link_libraries(LinearRegression gsl gslcblas)