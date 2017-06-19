project(discretizeGeom)

include_directories(${CMAKE_SOURCE_DIR}/src)
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})
find_package(MOAB REQUIRED)
include_directories(${MOAB_INCLUDE_DIR})

set(SRC
    src/discretize.cpp
    src/discretize.h
    discretize_geom.cpp)

add_executable(discretize_geom ${SRC})
target_link_libraries(discretize_geom MOAB mpi)
