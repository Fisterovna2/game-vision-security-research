cmake_minimum_required(VERSION 3.15)
project(GameVisionSecurityResearch VERSION 1.0.0 LANGUAGES CXX)

# C++17 standard required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Options
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_DOCS "Build documentation" OFF)

# Find required packages
find_package(OpenCV 4 REQUIRED)
find_package(Threads REQUIRED)

# DirectX (Windows only)
if(WIN32)
    find_package(DirectX QUIET)
    if(NOT DirectX_FOUND)
        message(STATUS "DirectX not found - using Windows SDK")
    endif()
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /WX- /permissive-)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# Source files
set(CAPTURE_SOURCES
    src/capture/desktop_duplication.cpp
    src/capture/capture_interface.cpp
)

set(VISION_SOURCES
    src/vision/object_detector.cpp
    src/vision/pattern_matcher.cpp
    src/vision/color_detector.cpp
)

set(ANALYSIS_SOURCES
    src/analysis/performance.cpp
    src/analysis/visualizer.cpp
)

set(UTILS_SOURCES
    src/utils/logger.cpp
    src/utils/config.cpp
)

# Main library
add_library(gvsr_core STATIC
    ${CAPTURE_SOURCES}
    ${VISION_SOURCES}
    ${ANALYSIS_SOURCES}
    ${UTILS_SOURCES}
)

target_link_libraries(gvsr_core
    ${OpenCV_LIBS}
    Threads::Threads
)

if(WIN32)
    target_link_libraries(gvsr_core
        d3d11
        dxgi
        dxguid
    )
endif()

# Examples
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Documentation
if(BUILD_DOCS)
    find_package(Doxygen)
    if(DOXYGEN_FOUND)
        add_subdirectory(docs)
    endif()
endif()

# Install targets
install(TARGETS gvsr_core
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY src/
    DESTINATION include/gvsr
    FILES_MATCHING PATTERN "*.h"
)

# Print configuration
message(STATUS "========================================")
message(STATUS "Game Vision Security Research")
message(STATUS "========================================")
message(STATUS "Version: ${PROJECT_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Build examples: ${BUILD_EXAMPLES}")
message(STATUS "Build tests: ${BUILD_TESTS}")
message(STATUS "Build docs: ${BUILD_DOCS}")
message(STATUS "========================================")
