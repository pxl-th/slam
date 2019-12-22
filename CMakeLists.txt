cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 17)

project(slam)

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)

add_executable(main main.cpp)

add_subdirectory(src/calibration)
add_subdirectory(src/frame)
add_subdirectory(src/map)
add_subdirectory(src/tracking)

target_compile_options(main
    PRIVATE
    -Wall
    -wd5045 # Spectre related stuff
)
target_include_directories(main PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(main
    PRIVATE
    ${OpenCV_LIBS}
    ${Eigen3_LIBS}

    g2o::stuff
    g2o::core
    g2o::solver_csparse
    g2o::solver_dense
    g2o::solver_pcg
    g2o::solver_slam2d_linear
    g2o::solver_structure_only
    g2o::solver_eigen
    g2o::types_data
    g2o::types_icp
    g2o::types_sba
    g2o::types_sclam2d
    g2o::types_sim3
    g2o::types_slam2d
    g2o::types_slam3d
)
