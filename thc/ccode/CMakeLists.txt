CMake_minimum_required (VERSION 2.6)
project (THC++)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake/custom)

include_directories("${PROJECT_SOURCE_DIR}")

add_executable(thcpp.x src/h5helper.cpp src/context_handler.cpp
               src/kmeans.cpp src/interpolating_vectors.cpp src/main.cpp)

find_package(HDF5 COMPONENTS CXX)
find_package(FFTW)
if (FFTW_FOUND)
    include_directories(${FFTW_INCLUDES})
endif (FFTW_FOUND)
include(options_wrappers)
include(hdf5)
include(math_libs)
if (MPI_FOUND)
    include_directories(${MPI_INCLUDE_DIRS})
endif (MPI_FOUND)
target_link_libraries(thcpp.x ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${FFTW_LIBRARIES} ${MATH_LIBS} ${MPI_LIBRARIES})
