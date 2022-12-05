cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(meshfree_solver_cpp LANGUAGES CUDA CXX)

enable_language(CXX)
enable_language(CUDA)

file(GLOB HEADERS "*.h")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

add_executable(reproducer ${HEADERS} reproducer.cu)
add_executable(reproducer_q_derivatives ${HEADERS} reproducer_q_derivatives.cu)

set_target_properties(reproducer PROPERTIES 
                    CUDA_SEPARABLE_COMPILATION ON
                    CXX_STANDARD 14
                )

set_target_properties(reproducer_q_derivatives PROPERTIES 
                    CUDA_SEPARABLE_COMPILATION ON
                    CXX_STANDARD 14
                )

target_compile_options(reproducer PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -mcmodel=medium
                       -O3
                       -lineinfo
                       --generate-line-info
                       -g
                       >)

                       
target_compile_options(reproducer_q_derivatives PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -mcmodel=medium
                       -O3
                       -lineinfo
                       --generate-line-info
                       -g
                       >)
