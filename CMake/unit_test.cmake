# Runs unit tests
FUNCTION( ADD_UNIT_TEST TESTNAME TEST_BINARY )
    IF ( HAVE_MPI )
        ADD_TEST(NAME ${TESTNAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 1 ${MPIEXEC_PREFLAGS} ${TEST_BINARY} ${ARGN})
        #SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1 )
    ELSE()
        ADD_TEST(NAME ${TESTNAME} COMMAND ${TEST_BINARY} ${ARGN})
        #SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1 )
    ENDIF()
    SET_PROPERTY(TEST ${TESTNAME} APPEND PROPERTY LABELS "unit")
ENDFUNCTION()

FUNCTION( ADD_MPI_UNIT_TEST TESTNAME TEST_BINARY PROC_COUNT )
    MESSAGE(STATUS "Adding unit test: ${TESTNAME}-np-${PROC_COUNT}")
    ADD_TEST(NAME ${TESTNAME} COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${PROC_COUNT} ${MPIEXEC_PREFLAGS} ${TEST_BINARY} ${ARGN})
    # Tests should be able to deal with any number of threads but mpi aware unit tests aren't
    # guaranteed yet.
    SET_TESTS_PROPERTIES( ${TESTNAME} PROPERTIES ENVIRONMENT OMP_NUM_THREADS=1 )
    SET_PROPERTY(TEST ${TESTNAME} APPEND PROPERTY LABELS "unit")
ENDFUNCTION()
