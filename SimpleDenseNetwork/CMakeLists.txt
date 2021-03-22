    cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
    project(LINEAR)


    set(LINEAR_LANGUAGE "C" CACHE STRING "Selected language {C,CUDA}")


    if(${LINEAR_LANGUAGE} STREQUAL "C")
    elseif(${LINEAR_LANGUAGE} STREQUAL "CUDA")

        enable_language(CUDA)

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

    add_library(nn STATIC ${nn_source} src/nn.c src/Linear.c include/Linear.h src/activation_functions.cu include/activation_functions.cuh)


    target_compile_features(nn PUBLIC c_std_11)


    set_target_properties( nn
            PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


    function(nn_executable target source_files)
        add_executable(${target} ${source_files})
        set_source_files_properties(${source_files} PROPERTIES LANGUAGE ${LINEAR_LANGUAGE})
        target_link_libraries(${target} nn)
    endfunction()