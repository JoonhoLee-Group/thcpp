project(THCPP)

add_library(thcpp SHARED h5helper.cpp context_handler.cpp
           kmeans.cpp qrcp.cpp interpolating_points.cpp
           interpolating_vectors.cpp
           molecular.cpp
           ${CMAKE_CURRENT_BINARY_DIR}/gitsha1.cpp)

include(GetGitRevisionDescription)
get_git_head_revision(GIT_REFSPEC GIT_SHA1)
git_local_changes(GIT_DIRTY)
configure_file("gitsha1.cpp.in"
               ${CMAKE_CURRENT_BINARY_DIR}/gitsha1.cpp @ONLY)
list(APPEND SOURCES "${CMAKE_CURRENT_BINARY_DIR}/gitsha1.cpp" gitsha1.h)

target_link_libraries(thcpp ${HDF5_C_LIBRARIES} ${HDF5_CXX_LIBRARIES} ${FFTW_LIBRARIES} ${MATH_LIBS} ${MPI_LIBRARIES})

add_executable(thcpp.x main.cpp)
target_link_libraries(thcpp.x PRIVATE thcpp)
