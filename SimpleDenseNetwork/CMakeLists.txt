cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(LINEAR LANGUAGES C CXX CUDA)

# Set CUDA Flags (adjust the compute capability as needed)
set(CMAKE_CUDA_FLAGS "-arch=sm_30 ${CMAKE_CUDA_FLAGS}")

# Include directories
include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

# Grab all the source and header files
file(GLOB headers "${PROJECT_SOURCE_DIR}/include/*.h" "${PROJECT_SOURCE_DIR}/include/*.cuh")
file(GLOB sources "${PROJECT_SOURCE_DIR}/src/*.c" "${PROJECT_SOURCE_DIR}/src/*.cu")

# Create the library
add_library(nn STATIC ${headers} ${sources})

# Set language standards
target_compile_features(nn PUBLIC c_std_11 cuda_std_11)

# Enable separable compilation for CUDA
set_target_properties(nn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Function to create an executable linking to the 'nn' library
function(nn_executable target source_files)
    add_executable(${target} ${source_files})
    target_link_libraries(${target} nn)
endfunction()
