set(LIBRARY_NAME "OptiX7Renderer")

# Visual studio versions https://cmake.org/cmake/help/latest/variable/MSVC_VERSION.html
if (WIN32 AND (MSVC_VERSION LESS_EQUAL 1929))
  message(WARNING "${LIBRARY_NAME} skipped: Requires Visual Studio Platform Toolset v143 or higher. Current MSVC version is ${MSVC_VERSION}")
  return()
endif()

find_package(CUDA 11.1)
if (CUDA_FOUND)
  set(CUDA_NVRTC_LIBRARY "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/nvrtc.lib") # TODO Hack until we switch to the CUDA Toolkit
else()
  message(WARNING "${LIBRARY_NAME} skipped: CUDA not installed.")
  return()
endif()

set(CMAKE_MODULE_PATH "${BIFROST_EXTENSIONS_DIR}/OptiX7Renderer" ${CMAKE_MODULE_PATH})
find_package(OptiX)
if (NOT OPTIX_FOUND)
  message(WARNING "${LIBRARY_NAME} skipped: OptiX not installed.")
  return()
endif()

# Select whether to use NVRTC or NVCC to generate PTX
set(CUDA_NVRTC_ENABLED ON CACHE BOOL "Use NVRTC to compile PTX at run-time instead of NVCC at build-time")

# Setup library

set(ROOT_SRC
  OptiXRenderer/CUDAUtils.h
  OptiXRenderer/Defines.h
  OptiXRenderer/PtxLoader.h
  OptiXRenderer/PtxLoader.cpp
  OptiXRenderer/PublicTypes.h
  OptiXRenderer/Renderer.h
  OptiXRenderer/Renderer.cpp
  OptiXRenderer/Types.h
)

set(SHADING_SRC
  OptiXRenderer/Shading/RGSolidColor.cu
)

set(NVCC_OPTIONS "--use_fast_math;--restrict")

if (CUDA_NVRTC_ENABLED)
  # Setup runtime compilation of .cu files

  set(CU_OBJ_SOURCES)
  foreach (FILE ${SHADING_SRC})
    get_filename_component(FILE_EXTENSION ${FILE} EXT)
    if (${FILE_EXTENSION} MATCHES ".cu")
      list(APPEND CU_OBJ_SOURCES ${FILE})
    endif()
  endforeach()

  # Create the rules to build the OBJ from the CUDA files.
  cuda_wrap_srcs(${target_name} OBJ GENERATED_FILES ${CU_OBJ_SOURCES} ${cmake_options} OPTIONS ${options})

else()

  # Create the folder to output shader PTX code to.
  set(CUDA_GENERATED_OUTPUT_DIR ${BIFROST_DATA_OUTPUT_DIRECTORY}/OptiXRenderer/ptx)
  file(MAKE_DIRECTORY ${CUDA_GENERATED_OUTPUT_DIR})

  # Setup shader compilation
  include_directories(. SYSTEM ${CUDA_INCLUDE_DIRS} ${OPTIX_INCLUDE_DIR}) # TODO Global include, use modern CMAKE CUDA setup to avoid this
  cuda_wrap_srcs(${LIBRARY_NAME} PTX GENERATED_FILES ${SHADING_SRC} OPTIONS ${NVCC_OPTIONS})
endif()

add_library(${LIBRARY_NAME} ${_cmake_options} ${ROOT_SRC} ${SHADING_SRC} ${GENERATED_FILES})

target_include_directories(${LIBRARY_NAME} PUBLIC . ${CUDA_INCLUDE_DIRS} PRIVATE ${OPTIX_INCLUDE_DIR})

target_link_libraries(${LIBRARY_NAME} PUBLIC
  Bifrost
  ${CUDA_LIBRARIES}
)

# Set defines used for NVRTC include paths. The include dirs are alway set to avoid compile errors or ifdef'ed code.
if (CUDA_NVRTC_ENABLED)
  target_compile_definitions(${LIBRARY_NAME} PRIVATE CUDA_NVRTC_ENABLED=1)
  target_link_libraries(${LIBRARY_NAME} LINK_PRIVATE ${CUDA_NVRTC_LIBRARY})
endif()
target_compile_definitions(${LIBRARY_NAME} PRIVATE
  OPTIX_RENDERER_INCLUDE_DIR="${BIFROST_EXTENSIONS_DIR}/OptiX7Renderer"
  CUDA_INCLUDE_DIR="${CUDA_INCLUDE_DIRS}" # CUDA only includes a single directory
  OPTIX_INCLUDE_DIR="${OPTIX_INCLUDE_DIR}")

source_group("" FILES ${ROOT_SRC})
source_group("Shading" FILES ${SHADING_SRC})
source_group("Shading\\ptx" FILES ${GENERATED_FILES})

set_target_properties(${LIBRARY_NAME} PROPERTIES 
 LINKER_LANGUAGE CXX
 FOLDER "Extensions"
)
