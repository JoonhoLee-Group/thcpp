project(thcpp_tests)

INCLUDE(${PROJECT_SOURCE_DIR}/../CMake/unit_test.cmake)
ADD_LIBRARY(catch_main catch_main.cpp)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/tests/bin)
TARGET_INCLUDE_DIRECTORIES(catch_main PUBLIC "${CMAKE_SOURCE_DIR}/lib/external")

EXECUTE_PROCESS(COMMAND cmake -E create_symlink
                "${CMAKE_SOURCE_DIR}/tests/data"
                "${CMAKE_BINARY_DIR}/tests/data")

ADD_EXECUTABLE("test_molecular" test_molecular.cpp)
TARGET_LINK_LIBRARIES(test_molecular catch_main thcpp)
target_include_directories(test_molecular PRIVATE ${CMAKE_SOURCE_DIR}/lib/external
                           ${CMAKE_SOURCE_DIR}/src)
ADD_MPI_UNIT_TEST("test_molecular" ${CMAKE_BINARY_DIR}/tests/bin/test_molecular 4)

ADD_EXECUTABLE("test_matrix" test_matrix.cpp)
TARGET_LINK_LIBRARIES(test_matrix catch_main thcpp)
target_include_directories(test_matrix PRIVATE ${CMAKE_SOURCE_DIR}/lib/external
                           ${CMAKE_SOURCE_DIR}/src)
ADD_MPI_UNIT_TEST("test_matrix" ${CMAKE_BINARY_DIR}/tests/bin/test_matrix 4)

ADD_EXECUTABLE("test_matrix_operations" test_matrix_operations.cpp)
TARGET_LINK_LIBRARIES(test_matrix_operations catch_main thcpp)
target_include_directories(test_matrix_operations PRIVATE ${CMAKE_SOURCE_DIR}/lib/external
                           ${CMAKE_SOURCE_DIR}/src)
ADD_MPI_UNIT_TEST("test_matrix_operations" ${CMAKE_BINARY_DIR}/tests/bin/test_matrix_operations 4)
