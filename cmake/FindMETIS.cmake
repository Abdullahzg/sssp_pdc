# FindMETIS.cmake
# Finds the METIS library
#
# This will define the following variables:
#
#   METIS_FOUND        - True if the system has METIS
#   METIS_INCLUDE_DIRS - METIS include directory
#   METIS_LIBRARIES    - METIS libraries
#   METIS_VERSION      - METIS version

find_path(METIS_INCLUDE_DIR
    NAMES metis.h
    PATHS /usr/include /usr/local/include
)

find_library(METIS_LIBRARY
    NAMES metis
    PATHS /usr/lib /usr/local/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(METIS
    REQUIRED_VARS METIS_LIBRARY METIS_INCLUDE_DIR
)

if(METIS_FOUND)
    set(METIS_LIBRARIES ${METIS_LIBRARY})
    set(METIS_INCLUDE_DIRS ${METIS_INCLUDE_DIR})
endif()

mark_as_advanced(METIS_INCLUDE_DIR METIS_LIBRARY) 