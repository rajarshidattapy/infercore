cmake_minimum_required(VERSION 3.16)
project(infercore VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(INFERCORE_WITH_ONNX "Build the ONNX Runtime backend when available" ON)
option(INFERCORE_SANITIZERS "Build with ASan+UBSan (Linux/GCC/Clang)" OFF)
option(INFERCORE_BUILD_TESTS "Build unit tests" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ---------------------------------------------------------------- ONNX Runtime
set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of an ONNX Runtime C/C++ distribution")
set(IC_ONNX_FOUND OFF)
if(INFERCORE_WITH_ONNX)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
            PATHS ${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session
                  ${ONNXRUNTIME_ROOT}/include)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib)
  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    set(IC_ONNX_FOUND ON)
    message(STATUS "ONNX Runtime found: ${ONNXRUNTIME_LIBRARY}")
  else()
    message(STATUS "ONNX Runtime not found; building without the ONNX backend")
  endif()
endif()

# ---------------------------------------------------------------- sanitizer flags
if(INFERCORE_SANITIZERS)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -g)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---------------------------------------------------------------- core library
set(IC_SOURCES
    src/tensor.cpp
    src/thread_pool.cpp
    src/kernels.cpp
    src/json.cpp
    src/ops.cpp
    src/weights.cpp
    src/native_model.cpp
    src/onnx_model.cpp
    src/infercore_c.cpp)

add_library(infercore_static STATIC ${IC_SOURCES})
target_include_directories(infercore_static PUBLIC include)

add_library(infercore SHARED ${IC_SOURCES})
target_include_directories(infercore PUBLIC include)
target_compile_definitions(infercore PRIVATE INFERCORE_BUILD)
set_target_properties(infercore PROPERTIES CXX_VISIBILITY_PRESET hidden)

find_package(Threads REQUIRED)
foreach(t infercore_static infercore)
  target_link_libraries(${t} PUBLIC Threads::Threads)
  if(IC_ONNX_FOUND)
    target_compile_definitions(${t} PUBLIC INFERCORE_WITH_ONNX)
    target_include_directories(${t} PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(${t} PUBLIC ${ONNXRUNTIME_LIBRARY})
  endif()
  if(UNIX AND NOT APPLE)
    target_link_libraries(${t} PUBLIC dl)
  endif()
endforeach()

# Copy runtime DLLs next to a target on Windows / set rpath on Linux.
function(ic_deploy_runtime target)
  if(WIN32 AND IC_ONNX_FOUND)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll $<TARGET_FILE_DIR:${target}>)
  endif()
  if(WIN32)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              $<TARGET_FILE:infercore> $<TARGET_FILE_DIR:${target}>)
  endif()
endfunction()

if(UNIX)
  set_target_properties(infercore PROPERTIES BUILD_RPATH "$ORIGIN")
endif()

# ---------------------------------------------------------------- demo (dlopen/LoadLibrary)
add_executable(infercore_demo demo/demo.cpp)
target_include_directories(infercore_demo PRIVATE include)
if(UNIX AND NOT APPLE)
  target_link_libraries(infercore_demo PRIVATE dl Threads::Threads)
endif()
ic_deploy_runtime(infercore_demo)

# ---------------------------------------------------------------- benchmark
add_library(sqlite3_vendored STATIC third_party/sqlite/sqlite3.c)
target_include_directories(sqlite3_vendored PUBLIC third_party/sqlite)
add_executable(infercore_benchmark bench/benchmark.cpp)
target_link_libraries(infercore_benchmark PRIVATE infercore_static sqlite3_vendored)
ic_deploy_runtime(infercore_benchmark)

# ---------------------------------------------------------------- tests
if(INFERCORE_BUILD_TESTS)
  enable_testing()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/googletest/CMakeLists.txt)
    add_subdirectory(third_party/googletest)
  else()
    include(FetchContent)
    FetchContent_Declare(googletest
      URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  add_executable(test_kernels tests/test_kernels.cpp)
  target_link_libraries(test_kernels PRIVATE infercore_static GTest::gtest_main)
  add_executable(test_model tests/test_model.cpp)
  target_link_libraries(test_model PRIVATE infercore_static GTest::gtest_main)
  add_executable(test_capi tests/test_capi.cpp)
  target_link_libraries(test_capi PRIVATE infercore GTest::gtest_main)
  add_executable(test_golden_onnx tests/test_golden_onnx.cpp)
  target_link_libraries(test_golden_onnx PRIVATE infercore_static GTest::gtest_main)
  foreach(t test_kernels test_model test_capi test_golden_onnx)
    add_test(NAME ${t} COMMAND ${t} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    ic_deploy_runtime(${t})
  endforeach()
endif()
