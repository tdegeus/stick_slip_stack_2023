cmake_minimum_required(VERSION 3.16)

project(Layers)

option(RunFixedLever "Compile RunFixedLever" ON)
option(RunFixedTop "Compile RunFixedTop" ON)
option(RunFreeLever "Compile RunFreeLever" ON)

execute_process(
    COMMAND python -c "from setuptools_scm import get_version; print(get_version(root='.'))"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE MYVERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if (RunFixedLever)
    add_subdirectory(RunFixedLever)
endif()

if (RunFixedTop)
    add_subdirectory(RunFixedTop)
endif()

if (RunFreeLever)
    add_subdirectory(RunFreeLever)
endif()
