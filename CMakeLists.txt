cmake_minimum_required(VERSION 3.18)
project(open_deep_tda VERSION 0.3.0 LANGUAGES CXX)

add_library(open_deep_tda_native STATIC src/persistence.cpp)
target_include_directories(open_deep_tda_native PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(open_deep_tda_native PUBLIC cxx_std_17)
set_target_properties(open_deep_tda_native PROPERTIES POSITION_INDEPENDENT_CODE ON CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(open_deep_tda_native PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)
if(BUILD_TESTING)
    add_executable(test_persistence tests/cpp/test_persistence.cpp)
    target_link_libraries(test_persistence PRIVATE open_deep_tda_native)
    add_test(NAME native_persistence COMMAND test_persistence)
endif()

option(OPEN_DEEP_TDA_BUILD_PYTHON "Build the optional pybind11 extension (requires pybind11)" OFF)
if(OPEN_DEEP_TDA_BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    # pip-installed pybind11 need not be in the platform CMake prefix path.
    execute_process(COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
                    OUTPUT_VARIABLE _pybind11_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE
                    ERROR_QUIET)
    if(_pybind11_cmake_dir)
        list(APPEND CMAKE_PREFIX_PATH "${_pybind11_cmake_dir}")
    endif()
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(_core bindings/module.cpp)
    target_link_libraries(_core PRIVATE open_deep_tda_native)
    install(TARGETS _core LIBRARY DESTINATION open_deep_tda)
    if(BUILD_TESTING)
        add_test(NAME native_binding COMMAND "${Python_EXECUTABLE}"
                 "${CMAKE_CURRENT_SOURCE_DIR}/tests/cpp/test_binding.py" "$<TARGET_FILE_DIR:_core>")
    endif()
endif()
