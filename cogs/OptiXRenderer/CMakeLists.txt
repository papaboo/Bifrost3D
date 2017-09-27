set(LIBRARY_NAME "OptiXRenderer")

# Add current path to CMake's module path, so we can use our own FindOptiX.cmake.
set(CMAKE_MODULE_PATH
  "${COGWHEEL_COGS_DIR}/OptiXRenderer"
  ${CMAKE_MODULE_PATH}
)

if (MSVC12)
    find_package(CUDA 7.5)
else(MSVC14)
    find_package(CUDA 8.0)
endif()
find_package(OptiX)

if (CUDA_FOUND AND OPTIX_FOUND AND WIN32)
  install_file(${OPTIX_1_DLL})
  install_file(${OPTIX_U_1_DLL})

  set(ROOT_SRC
    OptiXRenderer/Defines.h
    OptiXRenderer/Distributions.h
    OptiXRenderer/EnvironmentMap.h
    OptiXRenderer/EnvironmentMap.cpp
    OptiXRenderer/IBackend.h
    OptiXRenderer/Intersect.h
    OptiXRenderer/Kernel.h
    OptiXRenderer/Kernel.cu
    OptiXRenderer/OctahedralNormal.h
    OptiXRenderer/PresampledEnvironmentMap.h
    OptiXRenderer/PresampledEnvironmentMap.cpp
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
    OptiXRenderer/Shading/SimpleRGPs.cu
  )

  set(SHADING_BSDFS_SRC
    OptiXRenderer/Shading/BSDFs/Lambert.h
    OptiXRenderer/Shading/BSDFs/Burley.h
    OptiXRenderer/Shading/BSDFs/GGX.h
    OptiXRenderer/Shading/BSDFs/OrenNayar.h
  )

  set(SHADING_LIGHTSOURCES_SRC
    OptiXRenderer/Shading/LightSources/DirectionalLightImpl.h
    OptiXRenderer/Shading/LightSources/EnvironmentLightImpl.h
    OptiXRenderer/Shading/LightSources/LightImpl.h
    OptiXRenderer/Shading/LightSources/LightSources.cu
    OptiXRenderer/Shading/LightSources/PresampledEnvironmentLightImpl.h
    OptiXRenderer/Shading/LightSources/SphereLightImpl.h
  )

  set(SHADING_SHADING_MODELS_SRC
    OptiXRenderer/Shading/ShadingModels/DefaultShading.h
    OptiXRenderer/Shading/ShadingModels/LambertShading.h
  )

  # Add the path to the ptx to a define.
  set(CUDA_GENERATED_OUTPUT_DIR ${COGWHEEL_DATA_OUTPUT_DIRECTORY}/OptiXRenderer/ptx)
  file(MAKE_DIRECTORY ${CUDA_GENERATED_OUTPUT_DIR})

  include_directories(. ${CUDA_INCLUDE_DIRS} ${OPTIX_INCLUDE_DIRS})
  # TODO cuda_include_directories(. ${CUDA_INCLUDE_DIRS} ${OPTIX_INCLUDE_DIRS})
  # Create custom commands and targets for each file.
  set(_OPTIONS "-arch sm_30")
  cuda_wrap_srcs(${LIBRARY_NAME} PTX _GENERATED_FILES ${SHADING_SRC} ${SHADING_LIGHTSOURCES_SRC} OPTIONS ${_OPTIONS} )

  # Compute the file name of the intermedate link file used for separable compilation.
  cuda_compute_separable_compilation_object_file_name(LINK_FILE ${LIBRARY_NAME} "${${LIBRARY_NAME}_SEPARABLE_COMPILATION_OBJECTS}")

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
  cuda_link_separable_compilation_objects("${LINK_FILE}" ${LIBRARY_NAME} "${_OPTIONS}" "${${LIBRARY_NAME}_SEPARABLE_COMPILATION_OBJECTS}")

  source_group("" FILES ${ROOT_SRC})
  source_group("Shading" FILES ${SHADING_SRC})
  source_group("Shading\\BSDFs" FILES ${SHADING_BSDFS_SRC})
  source_group("Shading\\LightSources" FILES ${SHADING_LIGHTSOURCES_SRC})
  source_group("Shading\\ShadingModels" FILES ${SHADING_SHADING_MODELS_SRC})
  source_group("Shading\\ptx" FILES ${_GENERATED_FILES}) # Add the ptx files to a source group.

  target_include_directories(${LIBRARY_NAME}
    PUBLIC . ${CUDA_INCLUDE_DIRS} ${OPTIX_INCLUDE_DIRS}
)

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC Cogwheel ${CUDA_LIBRARIES} ${OPTIX_LIBRARIES}
    PRIVATE StbImageWriter
  )

  set_target_properties(${LIBRARY_NAME} PROPERTIES 
    LINKER_LANGUAGE CXX
    FOLDER "Cogs"
  )

  if (WIN32)
    target_compile_definitions(${LIBRARY_NAME} PRIVATE 
      NOMINMAX # OptiX math needs NOMINMAX defined. See optixu_math_namespace.h
    )
  endif()
else()
  if (NOT CUDA_FOUND)
    message(WARNING "OptiXRenderer: CUDA not installed.")
  endif()
  if (NOT OPTIX_FOUND)
    message(WARNING "OptiXRenderer: OptiX not installed.")
  endif()
  if (NOT WIN32)
    message(WARNING "OptiXRenderer: Only supported on windows.")
  endif()
endif()