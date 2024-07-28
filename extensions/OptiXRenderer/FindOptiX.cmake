# Find and setup OptiX.
# Search in the default install locations.

# First locate the OptiX search path.
set(OPTIX_WINDOWS_SEARCH_PATHS
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 6.5.0"
  "C:/Program Files/NVIDIA Corporation/OptiX SDK 6.5.0")

foreach (SEARCH_PATH ${OPTIX_WINDOWS_SEARCH_PATHS})
  if (EXISTS ${SEARCH_PATH}/bin64/ AND EXISTS ${SEARCH_PATH}/include/ AND EXISTS ${SEARCH_PATH}/lib64/)
    set(OPTIX_PATH ${SEARCH_PATH})
    break()
  endif()
endforeach()

# Find dlls, libs and include dirs.

# Include directory.
find_path(OPTIX_INCLUDE_DIRS optix.h
          PATHS
          ${OPTIX_PATH}/include
          DOC "The directory to include optix headers from"
)

# Find libraries.
find_library(OPTIX_LIB optix.6.5.0
             PATHS
             ${OPTIX_PATH}/lib64
             DOC "The main optix library"
)

find_library(OPTIX_U_LIB optixu.6.5.0
             PATHS
             ${OPTIX_PATH}/lib64
             DOC "The optix C++ namespace library"
)

set(OPTIX_LIBRARIES "${OPTIX_LIB}" "${OPTIX_U_LIB}")

# Find dlls
set(OPTIX_DLL "${OPTIX_PATH}/bin64/optix.6.5.0.dll")
set(OPTIX_U_DLL "${OPTIX_PATH}/bin64/optixu.6.5.0.dll")
set(OPTIX_DLLS "${OPTIX_DLL}" "${OPTIX_U_DLL}")

# Handle the QUIETLY and REQUIRED arguments and set OPTIX_FOUND to TRUE if
# all listed variables are set.
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OptiX DEFAULT_MSG OPTIX_LIB OPTIX_U_LIB OPTIX_DLL OPTIX_U_DLL OPTIX_INCLUDE_DIRS)