# Vendored GoogleTest 1.15.2 (source: github.com/google/googletest, BSD-3-Clause).
add_library(gtest STATIC src/gtest-all.cc)
target_include_directories(gtest PUBLIC include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Threads REQUIRED)
target_link_libraries(gtest PUBLIC Threads::Threads)
add_library(gtest_main STATIC src/gtest_main.cc)
target_link_libraries(gtest_main PUBLIC gtest)
add_library(GTest::gtest ALIAS gtest)
add_library(GTest::gtest_main ALIAS gtest_main)
