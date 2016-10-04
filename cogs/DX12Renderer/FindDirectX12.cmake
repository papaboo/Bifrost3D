# Find and setup DirectX 12.
# Search in the default install locations.
# TODO Cache these things.

# List windows SDK versions.
set(WINDOWS_SDK_VERSIONS "10.0.14393.0")

foreach (SDK_VERSION ${WINDOWS_SDK_VERSIONS})
  set(SDK_INCLUDE_DIR "C:/Program Files (x86)/Windows Kits/10/Include/${SDK_VERSION}")
  set(INCLUDE_DIR "${SDK_INCLUDE_DIR}/um")
  set(INCLUDE_SHARED_DIR "${SDK_INCLUDE_DIR}/shared")
  set(LIB_DIR "C:/Program Files (x86)/Windows Kits/10/Lib/${SDK_VERSION}/um/x64")
  if (EXISTS ${INCLUDE_DIR} AND EXISTS ${INCLUDE_SHARED_DIR} AND EXISTS ${LIB_DIR})
    set(WINDOWS_SDK_VERSION ${SDK_VERSION})
    break()
  endif()
endforeach()

# Find dlls, libs and include dirs.

# Include directory.
find_path(DIRECTX_12_INCLUDE_DIRS d3d12.h
          PATHS
          ${INCLUDE_DIR}
          DOC "The directories to include DirectX 12 headers from"
)
set(DIRECTX_12_INCLUDE_DIRS "${SDK_INCLUDE_DIR}/shared" "${DIRECTX_12_INCLUDE_DIRS}")

# Find libraries.
find_library(DIRECTX_12_LIB D3D12
             PATHS
             ${LIB_DIR}
             DOC "The DirectX 12 library"
)
find_library(DXGI_LIB dxgi
             PATHS
             ${LIB_DIR}
             DOC "The DirectX 12 library"
)

set(DIRECTX_12_LIBRARIES "${DIRECTX_12_LIB}" "${DXGI_LIB}")

# TODO Copy dlls.
# Use the ones in system32 for now.

# Handle the QUIETLY and REQUIRED arguments and set OPTIX_FOUND to TRUE if
# all listed variables are set.
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DIRECTX_12 DEFAULT_MSG 
                                  DIRECTX_12_LIB DXGI_LIB
                                  DIRECTX_12_INCLUDE_DIRS)