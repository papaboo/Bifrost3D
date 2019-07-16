# Find and setup DirectX 11.

# Setup paths
set(WINDOWS_SDK_VERSION ${CMAKE_VS_WINDOWS_TARGET_PLATFORM_VERSION})
set(SDK_INCLUDE_DIR "C:/Program Files (x86)/Windows Kits/10/Include/${WINDOWS_SDK_VERSION}")
set(INCLUDE_DIR "${SDK_INCLUDE_DIR}/um")
set(LIB_DIR "C:/Program Files (x86)/Windows Kits/10/Lib/${WINDOWS_SDK_VERSION}/um/x64")
  
# Find dlls, libs and include dirs.

# Include directory.
find_path(DIRECTX_11_INCLUDE_DIR d3d11_1.h
          PATHS
          ${INCLUDE_DIR}
          DOC "The directory to include DirectX 11 headers from"
)

# Find libraries.
find_library(DIRECTX_11_LIB D3D11
             PATHS
             ${LIB_DIR}
             DOC "The DirectX 11 library"
)
find_library(DXGI_LIB dxgi
             PATHS
             ${LIB_DIR}
             DOC "The DXGI library"
)
find_library(D3D_COMPILER_LIB D3DCompiler
             PATHS
             ${LIB_DIR}
             DOC "The HLSL compiler library"
)

set(DIRECTX_11_LIBRARIES "${DIRECTX_11_LIB}" "${DXGI_LIB}" "${D3D_COMPILER_LIB}")

# Handle the QUIETLY and REQUIRED arguments and set OPTIX_FOUND to TRUE if
# all listed variables are set.
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)
find_package_handle_standard_args(DIRECTX_11 DEFAULT_MSG 
                                  DIRECTX_11_LIB DXGI_LIB D3D_COMPILER_LIB
                                  DIRECTX_11_INCLUDE_DIR)