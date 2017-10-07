include_directories(${PROJECT_SOURCE_DIR}/LinearRegressor)
file(GLOB LINEARREGRESSOR_SOURCE ${PROJECT_SOURCE_DIR}/LinearRegressor/*.cpp)
add_library(LinearRegressor ${LINEARREGRESSOR_SOURCE})