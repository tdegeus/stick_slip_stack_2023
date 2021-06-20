cmake_minimum_required(VERSION 3.16)

project(Layers)

option(Run "Compile Run" ON)
option(RunFixedBoundary "Compile RunFixedBoundary" ON)

execute_process(
    COMMAND python -c "from setuptools_scm import get_version; print(get_version(root='.'))"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE MYVERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if (Run)
    add_subdirectory(Run)
endif()

if (RunFixedBoundary)
    add_subdirectory(RunFixedBoundary)
endif()
