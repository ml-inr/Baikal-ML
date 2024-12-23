cmake_minimum_required(VERSION 3.16)
project(tflite_inference)

# set(CMAKE_CXX_STANDARD 20)

# Set the path to your TensorFlow root directory
set(TENSORFLOW_DIR "/home/plotnikovgp/download/tensorflow")

message(STATUS "TensorFlow source directory: ${TENSORFLOW_DIR}")
# Include TensorFlow Lite CMake build
set(TFLITE_ENABLE_XNNPACK ON)
add_subdirectory(
  "${TENSORFLOW_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite" EXCLUDE_FROM_ALL)
add_executable(tflite_inference tflite_inference.cpp)

# Link against TensorFlow Lite and set include directories
target_link_libraries(tflite_inference 
    tensorflow-lite
)
