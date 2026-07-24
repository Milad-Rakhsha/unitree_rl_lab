cmake_minimum_required(VERSION 3.12)
project(go2_controller)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Boost REQUIRED COMPONENTS program_options)
find_package(yaml-cpp REQUIRED)

# Architecture-aware ONNX Runtime selection.
# On the Jetson (aarch64) the onnxruntime package is usually in
# deploy/thirdparty/onnxruntime-linux-aarch64-*. Prefer a matching version
# if both include/ and lib/ exist, otherwise fall back to newer versions.
if(NOT DEFINED ONNXRUNTIME_DIR)
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
    foreach(_aarch64_ver "1.22.0" "1.24.4")
      set(_candidate "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-aarch64-${_aarch64_ver}")
      if(EXISTS "${_candidate}/include/onnxruntime_cxx_api.h" AND EXISTS "${_candidate}/lib")
        set(ONNXRUNTIME_DIR "${_candidate}")
        break()
      endif()
    endforeach()
    if(NOT DEFINED ONNXRUNTIME_DIR)
      message(FATAL_ERROR "Could not find a valid ARM64 ONNX Runtime in deploy/thirdparty/")
    endif()
  else()
    set(ONNXRUNTIME_DIR "${PROJECT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0")
  endif()
endif()

if(EXISTS "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
  set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
elseif(EXISTS "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1")
  set(ONNXRUNTIME_LIB "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so.1")
else()
  message(FATAL_ERROR "Could not find libonnxruntime.so in ${ONNXRUNTIME_DIR}/lib")
endif()

message(STATUS "Using ONNX Runtime: ${ONNXRUNTIME_DIR}")

include_directories(
  /usr/include/eigen3
  /usr/local/include/ddscxx # dds
  /usr/local/include/iceoryx/v2.0.2 # dds
  include
  ${ONNXRUNTIME_DIR}/include
  ${PROJECT_SOURCE_DIR}/../../include/
)

link_libraries(
  unitree_sdk2 ddsc ddscxx rt pthread # dds
  libboost_program_options.a libyaml-cpp.a fmt
  ${ONNXRUNTIME_LIB}
)

file(GLOB_RECURSE ADD_SRC_LIST
  src/*.cpp
)

add_library(${PROJECT_NAME}_lib ${ADD_SRC_LIST})
link_libraries(${PROJECT_NAME}_lib)

add_executable(go2_ctrl main.cpp)