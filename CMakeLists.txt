cmake_minimum_required(VERSION 3.0)
project(slam)

find_package(OpenCV CONFIG REQUIRED)

add_executable(
    main
    main.cpp
    "${CMAKE_SOURCE_DIR}/src/detector.cpp"
    "${CMAKE_SOURCE_DIR}/src/matcher.cpp"
    "${CMAKE_SOURCE_DIR}/src/frame.cpp"
)
target_include_directories(main PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
