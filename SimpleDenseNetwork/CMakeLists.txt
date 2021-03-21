    cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
    project(LINEAR)

    #set(CMAKE_CUDA_STANDARD 14)

    # language switch
    set(LINEAR_LANGUAGE "C" CACHE STRING "Selected language {C,CUDA}")

    # check for valid FOOBAR_LANGUAGE
    if(${LINEAR_LANGUAGE} STREQUAL "C")
    elseif(${LINEAR_LANGUAGE} STREQUAL "CUDA")
        # enable CUDA
        enable_language(CUDA)
        # CMAKE_CUDA_FLAGS is where gpu architecture should be specified
        set(CMAKE_CUDA_FLAGS "-arch compute_30 ${CMAKE_CUDA_FLAGS}")
    else()
        message(FATAL_ERROR "${LINEAR_LANGUAGE} is not a supported language.")
    endif()

    include_directories(
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/src
    )

    file(GLOB nn_source
            "${PROJECT_SOURCE_DIR}/include/*.h"
                "${PROJECT_SOURCE_DIR}/include/*.cuh"
            "${PROJECT_SOURCE_DIR}/src/*.cu"
                "${PROJECT_SOURCE_DIR}/src/*.c"
            )

    set_source_files_properties(${nn_source} PROPERTIES LANGUAGE ${LINEAR_LANGUAGE})

    add_library(nn STATIC ${nn_source} src/nn.c src/Linear.c include/Linear.h src/activation_functions.cu include/activation_functions.cuh src/cuda_activations.cu src/cuda_activations.cuh)


    # Request that particles be built with -std=c++11
    # As this is a public compile feature anything that links to
    # particles will also build with -std=c++11
    target_compile_features(nn PUBLIC c_std_11)

    # We need to explicitly state that we need all CUDA files in the
    # particle library to be built with -dc as the member functions
    # could be called by other libraries and executables
    set_target_properties( nn
            PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


    #add_executable(LinearNN main.c)

    #set_property(TARGET LinearNN
    #        PROPERTY CUDA_SEPARABLE_COMPILATION ON)

    function(nn_executable target source_files)
        add_executable(${target} ${source_files})
        set_source_files_properties(${source_files} PROPERTIES LANGUAGE ${LINEAR_LANGUAGE})
        target_link_libraries(${target} nn)
    endfunction()