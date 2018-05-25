# Find and setup OptiX.
# Search in the default install locations.

MACRO(set_boolean variable)
  if(${ARGN})
    set(${variable} TRUE)
  else(${ARGN})
    set(${name} FALSE)
  endif(${ARGN})
ENDMACRO(set_boolean)

# First locate the OptiX search path.
set(OPTIX_WINDOWS_SEARCH_PATHS
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.1.0"
  "C:/Program Files/NVIDIA Corporation/OptiX SDK 5.1.0"
  "C:/ProgramData/NVIDIA Corporation/OptiX SDK 5.0.1"
  "C:/Program Files/NVIDIA Corporation/OptiX SDK 5.0.1")

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
find_library(OPTIX_LIB optix.1 optix.51
             PATHS
             ${OPTIX_PATH}/lib64
             DOC "The main optix library"
)

find_library(OPTIX_U_LIB optixu.1
             PATHS
             ${OPTIX_PATH}/lib64
             DOC "The optix C++ namespace library"
)

set(OPTIX_LIBRARIES "${OPTIX_LIB}" "${OPTIX_U_LIB}")

# Find dlls
if (EXISTS ${OPTIX_PATH}/bin64/optix.1.dll)
  set(OPTIX_DLL "${OPTIX_PATH}/bin64/optix.1.dll")
endif()

if (EXISTS ${OPTIX_PATH}/bin64/optix.51.dll)
  set(OPTIX_DLL "${OPTIX_PATH}/bin64/optix.51.dll")
endif()

if (EXISTS ${OPTIX_PATH}/bin64/optixu.1.dll)
  set(OPTIX_U_DLL "${OPTIX_PATH}/bin64/optixu.1.dll")
endif()

set(OPTIX_DLLS "${OPTIX_DLL}" "${OPTIX_U_DLL}")

# Handle the QUIETLY and REQUIRED arguments and set OPTIX_FOUND to TRUE if
# all listed variables are set.
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(OPTIX DEFAULT_MSG OPTIX_LIB OPTIX_U_LIB OPTIX_DLL OPTIX_U_DLL OPTIX_INCLUDE_DIRS)