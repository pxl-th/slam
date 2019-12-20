cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 17)

project(slam)

find_package(OpenCV CONFIG REQUIRED)

add_executable(main main.cpp)

add_subdirectory(src/calibration)
add_subdirectory(src/frame)
add_subdirectory(src/map)
add_subdirectory(src/tracking)

target_include_directories(main PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
