set(LIBRARY_NAME "OptiXRenderer")

# Add current path to CMake's module path, so we can use our own FindOptiX.cmake.
set(CMAKE_MODULE_PATH
  "${COGWHEEL_COGS_DIR}/OptiXRenderer"
  ${CMAKE_MODULE_PATH}
)

find_package(CUDA 7.5)
find_package(OpenGL)
find_package(OptiX)

if (CUDA_FOUND AND OpenGL_FOUND AND OPTIX_FOUND AND WIN32)
  install_file(${OPTIX_1_DLL})
  install_file(${OPTIX_U_1_DLL})

  include_directories(. ${CUDA_INCLUDE_DIRS} ${OpenGL_INCLUDE_DIRS} ${OPTIX_INCLUDE_DIRS})

  set(ROOT_SRC
    OptiXRenderer/Defines.h
    OptiXRenderer/Distributions.h
    OptiXRenderer/EncodedNormal.h
    OptiXRenderer/Intersect.h
    OptiXRenderer/Kernel.h
    OptiXRenderer/Kernel.cu
    OptiXRenderer/RNG.h
    OptiXRenderer/Renderer.h
    OptiXRenderer/Renderer.cpp
    OptiXRenderer/RhoTexture.h
    OptiXRenderer/RhoTexture.cpp
    OptiXRenderer/TBN.h
    OptiXRenderer/Types.h
    OptiXRenderer/Utils.h
  )

  set(SHADING_SRC
    OptiXRenderer/Shading/IntersectSphere.cu
    OptiXRenderer/Shading/IntersectTriangle.cu
    OptiXRenderer/Shading/MonteCarlo.cu
    OptiXRenderer/Shading/NormalRendering.cu
    OptiXRenderer/Shading/PathTracing.cu
  )

  set(SHADING_BSDFS_SRC
    OptiXRenderer/Shading/BSDFs/Lambert.h
    OptiXRenderer/Shading/BSDFs/GGX.h
  )

  set(SHADING_LIGHTSOURCES_SRC
    OptiXRenderer/Shading/LightSources/LightSources.cu
    OptiXRenderer/Shading/LightSources/SphereLightImpl.h
  )

  set(SHADING_SHADING_MODELS_SRC
    OptiXRenderer/Shading/ShadingModels/DefaultShading.h
    OptiXRenderer/Shading/ShadingModels/DefaultShadingRho.h
  )

  # Add the path to the ptx to a define.
  set(OPTIX_PTX_DIR ${CMAKE_BINARY_DIR}/${LIBRARY_NAME})
  add_definitions(-DOPTIXRENDERER_PTX_DIR="${OPTIX_PTX_DIR}") #TODO For distributing releases I need a better way of handling the path to the PTX.

  # Create custom commands and targets for each file.
  set(_OPTIONS "")
  CUDA_WRAP_SRCS( ${LIBRARY_NAME} PTX _GENERATED_FILES ${SHADING_SRC} ${SHADING_LIGHTSOURCES_SRC} OPTIONS ${_OPTIONS} )

  # Compute the file name of the intermedate link file used for separable compilation.
  CUDA_COMPUTE_SEPARABLE_COMPILATION_OBJECT_FILE_NAME(LINK_FILE ${LIBRARY_NAME} "${${LIBRARY_NAME}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_library(${LIBRARY_NAME} ${_cmake_options}
    ${_GENERATED_FILES}
    ${ROOT_SRC}
    ${SHADING_SRC}
    ${SHADING_BSDFS_SRC}
    ${SHADING_LIGHTSOURCES_SRC}
    ${SHADING_SHADING_MODELS_SRC}
    ${LINK_FILE}
  )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${LIBRARY_NAME}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  CUDA_LINK_SEPARABLE_COMPILATION_OBJECTS("${LINK_FILE}" ${LIBRARY_NAME} "${_OPTIONS}" "${${LIBRARY_NAME}_SEPARABLE_COMPILATION_OBJECTS}")
  
  source_group("" FILES ${ROOT_SRC})
  source_group("Shading" FILES ${SHADING_SRC})
  source_group("Shading\\BSDFs" FILES ${SHADING_BSDFS_SRC})
  source_group("Shading\\LightSources" FILES ${SHADING_LIGHTSOURCES_SRC})
  source_group("Shading\\ShadingModels" FILES ${SHADING_SHADING_MODELS_SRC})
  source_group("Shading\\ptx" FILES ${_GENERATED_FILES}) # Add the ptx files to a source group.

  target_link_libraries(${LIBRARY_NAME}
    Gogwheel
    ${CUDA_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${OPTIX_LIBRARIES}
  )

  set_target_properties(${LIBRARY_NAME} PROPERTIES 
    LINKER_LANGUAGE CXX
    FOLDER "Cogs"
  )

  if (WIN32)
    add_definitions(-DNOMINMAX) # OptiX math needs NOMINMAX defined. See optixu_math_namespace.h
  endif()
else()
  if (NOT CUDA_FOUND)
    message(WARNING "CUDA not installed.")
  endif()
  if (NOT OpenGL_FOUND)
    message(WARNING "OpenGL not found.")
  endif()
  if (NOT OPTIX_FOUND)
    message(WARNING "OptiX not installed.")
  endif()
  if (NOT WIN32)
    message(WARNING "Only supported on windows.")
  endif()
  message(WARNING "Disabling OptiXRenderer.")
endif()