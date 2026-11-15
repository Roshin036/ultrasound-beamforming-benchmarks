cmake_minimum_required(VERSION 3.20)
project(ultrasound_beamforming LANGUAGES CXX)

add_executable(das_reference das_reference.cpp)
target_compile_features(das_reference PRIVATE cxx_std_20)
add_executable(das_cpu_naive das_cpu_naive.cpp)
target_compile_features(das_cpu_naive PRIVATE cxx_std_20)
add_executable(das_cpu_template das_cpu_template.cpp)
target_compile_features(das_cpu_template PRIVATE cxx_std_20)
add_executable(dmas_cpu_naive dmas_cpu_naive.cpp)
target_compile_features(dmas_cpu_naive PRIVATE cxx_std_20)
add_executable(dmas_cpu_template dmas_cpu_template.cpp)
target_compile_features(dmas_cpu_template PRIVATE cxx_std_20)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  add_executable(das_cuda_naive das_cuda_naive.cu)
  add_executable(das_cuda_template das_cuda_template.cu)
  add_executable(dmas_cuda_naive dmas_cuda_naive.cu)
  add_executable(dmas_cuda_template dmas_cuda_template.cu)
  set_target_properties(das_cuda_naive PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
  )
  set_target_properties(das_cuda_template PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
  )
  set_target_properties(dmas_cuda_naive PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
  )
  set_target_properties(dmas_cuda_template PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_ARCHITECTURES 75
  )
endif()

if(MSVC)
  target_compile_options(das_reference PRIVATE /W4 /permissive-)
  target_compile_options(das_cpu_naive PRIVATE /W4 /permissive-)
  target_compile_options(das_cpu_template PRIVATE /W4 /permissive-)
  target_compile_options(dmas_cpu_naive PRIVATE /W4 /permissive-)
  target_compile_options(dmas_cpu_template PRIVATE /W4 /permissive-)
else()
  target_compile_options(das_reference PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(das_cpu_naive PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(das_cpu_template PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(dmas_cpu_naive PRIVATE -Wall -Wextra -Wpedantic)
  target_compile_options(dmas_cpu_template PRIVATE -Wall -Wextra -Wpedantic)
endif()
