cmake_minimum_required(VERSION 3.18)

# Handle CI environment
option(CI_BUILD "Building in CI environment without physical GPU" OFF)

# Force set CUDA language before project() declaration in CI environments
if(CI_BUILD)
  set(CMAKE_CUDA_ARCHITECTURES 70)
  
  # Set environment variable for CUDA compiler if not already set
  if(NOT DEFINED ENV{CUDACXX} AND NOT DEFINED CMAKE_CUDA_COMPILER)
    if(EXISTS "/usr/local/cuda/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    elseif(EXISTS "/usr/local/cuda-11.8/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.8/bin/nvcc")
    endif()
  endif()
endif()

project(gpu_image_processing LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA standards and flags
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA architectures if not already set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75 80 86)
endif()

# Set include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Define source files
set(SOURCES
    src/image_processing.cu
    src/kernels.cu
    src/benchmarking.cu
    src/utils.cu
)

# Define header files
set(HEADERS
    include/image_processing.h
    include/kernels.h
    include/utils.h
)

# Define examples
set(EXAMPLES
    examples/basic_example.cu
    examples/benchmark_example.cu
)

# Define tests
set(TESTS
    tests/test_operations.cu
    tests/test_pipeline.cu
)

# Create library
add_library(gpu_image_processing STATIC ${SOURCES} ${HEADERS})
set_target_properties(gpu_image_processing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Create examples
foreach(example_file ${EXAMPLES})
    get_filename_component(example_name ${example_file} NAME_WE)
    add_executable(${example_name} ${example_file})
    target_link_libraries(${example_name} gpu_image_processing)
    set_target_properties(${example_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

# Create tests
if(BUILD_TESTING)
    enable_testing()
    foreach(test_file ${TESTS})
        get_filename_component(test_name ${test_file} NAME_WE)
        add_executable(${test_name} ${test_file})
        target_link_libraries(${test_name} gpu_image_processing)
        set_target_properties(${test_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()

# Installation
install(TARGETS gpu_image_processing
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES ${HEADERS} DESTINATION include/gpu_image_processing)

# Option to build examples
option(BUILD_EXAMPLES "Build the examples" ON)

# Print configuration summary
message(STATUS "Configuration Summary:")
message(STATUS "  CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Building examples: ${BUILD_EXAMPLES}")
message(STATUS "  Building tests: ${BUILD_TESTING}")