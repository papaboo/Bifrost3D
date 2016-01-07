set(LIBRARY_NAME "OptiXRenderer")

# Add current path to CMake's module path, so we can use our own FindOptiX.cmake.
set(CMAKE_MODULE_PATH
  "${COGWHEEL_COGS_DIR}/OptiXRenderer"
  ${CMAKE_MODULE_PATH}
)

find_package(CUDA 7.5 REQUIRED)
find_package(OptiX)

# TODO Just disable cog if CUDA or OptiX isn't found and report useful error messages about what to install to the user.
# Also disable when OS isn't windows, as nothing else has been tested yet

install_file(${OPTIX_1_DLL})
install_file(${OPTIX_U_1_DLL})

include_directories(. ${CUDA_INCLUDE_DIRS} ${OPTIX_INCLUDE_PATH})

set(ROOT_SRC
  OptiXRenderer/Kernel.h
  OptiXRenderer/Kernel.cu
  OptiXRenderer/Renderer.h
  OptiXRenderer/Renderer.cpp
  OptiXRenderer/Types.h
)

set(SHADING_SRC
  OptiXRenderer/Shading/NormalRendering.cu
  OptiXRenderer/Shading/PathTracing.cu
)

# Add the path to the ptx to a define.
set(OPTIX_PTX_DIR ${CMAKE_BINARY_DIR}/${LIBRARY_NAME})
add_definitions(-DOPTIXRENDERER_PTX_DIR="${OPTIX_PTX_DIR}") #TODO When distributing releases I need a better way of handling the path to the PTX.

# Construct the list of PTX files TODO Can I do this in a more generic way? By querying the library or FindCUDA for the list? Or replicating CUDA_ADD_EXECUTABLE here.
set(SHADING_PTX_SRC
  ${OPTIX_PTX_DIR}/${LIBRARY_NAME}_generated_NormalRendering.cu.ptx
  ${OPTIX_PTX_DIR}/${LIBRARY_NAME}_generated_PathTracing.cu.ptx
)

# Compile OptiX shader sources to ptx
set_source_files_properties(${SHADING_SRC}
  PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT PTX
)

cuda_add_library(${LIBRARY_NAME} ${ROOT_SRC} ${SHADING_SRC})

source_group("" FILES ${ROOT_SRC})
source_group("Shading" FILES ${SHADING_SRC})
source_group("Shading\\ptx" FILES ${SHADING_PTX_SRC}) # TODO Add the ptx files to a source gorup.

target_link_libraries(${LIBRARY_NAME}
  Gogwheel
  ${CUDA_LIBRARIES}
  ${OPTIX_LIBRARIES}
)

set_target_properties(${LIBRARY_NAME} PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)

if (WIN32)
  add_definitions(-DNOMINMAX) # OptiX math needs NOMINMAX defined. See optixu_math_namespace.h
endif()