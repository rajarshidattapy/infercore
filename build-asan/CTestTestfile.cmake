# CMake generated Testfile for 
# Source directory: /home/sandbox/infercore
# Build directory: /home/sandbox/infercore/build-asan
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_kernels "/home/sandbox/infercore/build-asan/test_kernels")
set_tests_properties(test_kernels PROPERTIES  WORKING_DIRECTORY "/home/sandbox/infercore" _BACKTRACE_TRIPLES "/home/sandbox/infercore/CMakeLists.txt;127;add_test;/home/sandbox/infercore/CMakeLists.txt;0;")
add_test(test_model "/home/sandbox/infercore/build-asan/test_model")
set_tests_properties(test_model PROPERTIES  WORKING_DIRECTORY "/home/sandbox/infercore" _BACKTRACE_TRIPLES "/home/sandbox/infercore/CMakeLists.txt;127;add_test;/home/sandbox/infercore/CMakeLists.txt;0;")
add_test(test_capi "/home/sandbox/infercore/build-asan/test_capi")
set_tests_properties(test_capi PROPERTIES  WORKING_DIRECTORY "/home/sandbox/infercore" _BACKTRACE_TRIPLES "/home/sandbox/infercore/CMakeLists.txt;127;add_test;/home/sandbox/infercore/CMakeLists.txt;0;")
add_test(test_golden_onnx "/home/sandbox/infercore/build-asan/test_golden_onnx")
set_tests_properties(test_golden_onnx PROPERTIES  WORKING_DIRECTORY "/home/sandbox/infercore" _BACKTRACE_TRIPLES "/home/sandbox/infercore/CMakeLists.txt;127;add_test;/home/sandbox/infercore/CMakeLists.txt;0;")
subdirs("third_party/googletest")
