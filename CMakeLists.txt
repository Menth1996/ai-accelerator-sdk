cmake_minimum_required(VERSION 3.10)
project(AI_ACCELERATOR_SDK CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Source files
set(SDK_SOURCES
    src/inference_engine.cpp
    src/model_loader.cpp
    src/tensor.cpp
    src/ops/convolution.cpp
)

# Add library
add_library(ai_accelerator_sdk STATIC ${SDK_SOURCES})

# Install headers and library
install(TARGETS ai_accelerator_sdk DESTINATION lib)
install(DIRECTORY include/ai_accelerator_sdk/ DESTINATION include)

# Add example executable
add_executable(quantized_resnet examples/quantized_resnet.cpp)
target_link_libraries(quantized_resnet ai_accelerator_sdk)

# Add tests
enable_testing()
add_executable(test_inference tests/test_inference.cpp)
target_link_libraries(test_inference ai_accelerator_sdk)
add_test(NAME InferenceTest COMMAND test_inference)

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops ai_accelerator_sdk)
add_test(NAME OpsTest COMMAND test_ops)
