cmake_minimum_required(VERSION 3.16)

project(Run)

option(ASSERT "Use assertions" OFF)
option(DEBUG "Switch on all possible assertions" OFF)

if(NOT DEFINED MYVERSION)
    execute_process(
        COMMAND python -c "from setuptools_scm import get_version; print(get_version(root='..'))"
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE MYVERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

message(STATUS "Building ${PROJECT_NAME} ${MYVERSION}")

set(CMAKE_BUILD_TYPE Release)

add_executable(${PROJECT_NAME} main.cpp)

set(HIGHFIVE_USE_BOOST 0)
set(HIGHFIVE_USE_XTENSOR 1)

find_package(FrictionQPotFEM REQUIRED)
find_package(prrng REQUIRED)
find_package(xtensor REQUIRED)
find_package(xsimd REQUIRED)
find_package(HighFive REQUIRED)
find_package(docopt REQUIRED)
find_package(fmt REQUIRED)
find_package(cpppath REQUIRED)

message(STATUS "FrictionQPotFEM v" ${FrictionQPotFEM_VERSION})
message(STATUS "GooseFEM v" ${GooseFEM_VERSION})
message(STATUS "GMatTensor v" ${GMatTensor_VERSION})
message(STATUS "QPot v" ${QPot_VERSION})
message(STATUS "prrng v" ${prrng_VERSION})
message(STATUS "xtensor v" ${xtensor_VERSION})
message(STATUS "xsimd v" ${xsimd_VERSION})
message(STATUS "HighFive v" ${HighFive_VERSION})
message(STATUS "docopt v" ${docopt_VERSION})
message(STATUS "fmt v" ${fmt_VERSION})
message(STATUS "cpppath v" ${cpppath_VERSION})

target_link_libraries(${PROJECT_NAME}
    FrictionQPotFEM
    prrng
    xtensor::optimize
    xtensor::use_xsimd
    HighFive
    docopt
    fmt::fmt
    cpppath
    FrictionQPotFEM::compiler_warnings)

target_compile_definitions(${PROJECT_NAME} PRIVATE "-DMYVERSION=\"${MYVERSION}\"")

if(DEBUG)
    target_link_libraries(${PROJECT_NAME} FrictionQPotFEM::debug)
elseif(ASSERT)
    target_link_libraries(${PROJECT_NAME} FrictionQPotFEM::assert)
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE -DNDEBUG)
endif()

if (APPLE)
    set_target_properties(${PROJECT_NAME} PROPERTIES MACOSX_RPATH ON)
else()
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_WITH_INSTALL_RPATH 1
        SKIP_BUILD_RPATH FALSE)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS ${PROJECT_NAME} DESTINATION bin)
