cmake_minimum_required(VERSION 3.5)
project(SDI)

#
# Dependencies
#

# DEBUG
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb -w")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")
set(CMAKE_CXX_FLAGS "-Werror=narrowing")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)

# find dependencies
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")
find_package(NTL REQUIRED)
find_package(GMP REQUIRED)
find_package(GMPXX REQUIRED)
find_package(Threads REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenMP)
find_package(pybind11 REQUIRED)
find_package(Boost REQUIRED COMPONENTS log locale)

if(OPENMP_FOUND)
    message(STATUS "OPENMP FOUND")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

if(Boost_FOUND)
    message( STATUS "Boost_INCLUDE_DIRS = ${Boost_INCLUDE_DIRS}.")
    message( STATUS "Boost_LIBRARIES = ${Boost_LIBRARIES}.")
    message( STATUS "Boost_LIB_VERSION = ${Boost_LIB_VERSION}.")
endif()

# sync submodules
include(UpdateSubmodules)
sync_submodules()

add_executable(experiment test/experiment.cpp)

target_include_directories(experiment PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_include_directories(experiment PUBLIC "/usr/local/include/rest_rpc/include")
target_include_directories(experiment PUBLIC "/usr/local/include/pybind11")
target_include_directories(experiment PUBLIC "/usr/include/python3.10")
target_include_directories(experiment PUBLIC ${GTEST_INCLUDE_DIRS})
target_include_directories(experiment PUBLIC ${Boost_INCLUDE_DIRS})

target_link_libraries(experiment
                      PRIVATE
                      "/usr/local/lib/libophelib.a"
                      ${LIBNTL_LIBRARIES}
                      ${LIBGMP_LIBRARIES}
                      ${LIBGMPXX_LIBRARIES}
                      Threads::Threads
                      ${GTEST_BOTH_LIBRARIES}
                      ${Boost_LIBRARIES}
                      pybind11::embed)