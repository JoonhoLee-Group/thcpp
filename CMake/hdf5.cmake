#.rst:
#
# Finds HDF5 and enables it, provided that the Fortran 2003 interface was
# compiled.
#
# Variables used::
#
#   ENABLE_HDF5
#
# Variables defined::
#
#   USE_HDF5
#   HDF5_Fortran_LIBRARIES
#   HDF5_INCLUDE_DIRS
#
# autocmake.yml configuration::
#
#   docopt: "--hdf5=<ENABLE_HDF5> Enable HDF5 [default: True]."
#   define: "'-DENABLE_HDF5=\"{0}\"'.format(arguments['--hdf5'])"

find_package(HDF5)
if (HDF5_FOUND)
    include_directories(${HDF5_INCLUDE_DIRS})
endif (HDF5_FOUND)
