cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(slam)

find_package(OpenCV CONFIG REQUIRED)

add_executable(
    main
    main.cpp
    "${CMAKE_SOURCE_DIR}/src/calibration.cpp"
    "${CMAKE_SOURCE_DIR}/src/calibration_settings.cpp"
    "${CMAKE_SOURCE_DIR}/src/detector.cpp"
    "${CMAKE_SOURCE_DIR}/src/frame.cpp"
    "${CMAKE_SOURCE_DIR}/src/matcher.cpp"
)
target_include_directories(main PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
