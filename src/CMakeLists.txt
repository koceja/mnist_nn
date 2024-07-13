# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and version
project(neural_net VERSION 1.0)

# Specify the C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED True)

# Add the executable
add_executable(neural_net main.c)

# Include header files
target_include_directories(neural_net PUBLIC "${PROJECT_SOURCE_DIR}")