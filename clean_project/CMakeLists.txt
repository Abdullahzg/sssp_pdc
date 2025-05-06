cmake_minimum_required(VERSION 3.10)
project(SSSP_Update)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# Include METIS
set(METIS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include/metis")
set(METIS_LIBRARY "/usr/lib/x86_64-linux-gnu/libmetis.so")

# Add include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${MPI_INCLUDE_PATH}
    ${METIS_INCLUDE_DIR}
)

# Set source files
set(SOURCES
    src/main.cpp
    src/graph.cpp
    src/sssp_new.cpp
    src/graph_partitioner.cpp
)

# Add executable
add_executable(sssp_update ${SOURCES})

# Link libraries
target_link_libraries(sssp_update
    ${MPI_LIBRARIES}
    OpenMP::OpenMP_CXX
    ${METIS_LIBRARY}
)

# Add MPI flags
if(MPI_COMPILE_FLAGS)
    set_target_properties(sssp_update PROPERTIES
        COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
    set_target_properties(sssp_update PROPERTIES
        LINK_FLAGS "${MPI_LINK_FLAGS}")
endif()

# Add OpenMP flags
if(OpenMP_CXX_FOUND)
    target_link_libraries(sssp_update OpenMP::OpenMP_CXX)
endif()

# Add compiler flags
target_compile_options(sssp_update PRIVATE -Wall -Wextra -O3 -g)

# Add option to enable MPI profiling with mpiP
option(ENABLE_MPIP "Enable MPI profiling with mpiP" OFF)

if(ENABLE_MPIP)
    # Check if mpiP is available
    find_library(MPIP_LIBRARY NAMES mpiP)
    
    if(MPIP_LIBRARY)
        message(STATUS "Found mpiP library: ${MPIP_LIBRARY}")
        # Link mpiP to the target
        target_link_libraries(sssp_update ${MPIP_LIBRARY})
        # Define a preprocessor macro to indicate mpiP is enabled
        target_compile_definitions(sssp_update PRIVATE MPIP_ENABLED)
    else()
        message(WARNING "mpiP library not found, MPI profiling will be disabled")
    endif()
endif() 