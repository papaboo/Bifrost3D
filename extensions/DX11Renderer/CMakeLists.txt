set(LIBRARY_NAME "DX11Renderer")

# Add current path to CMake's module path, so we can use our own FindDirectX11.cmake.
set(CMAKE_MODULE_PATH
  "${BIFROST_EXTENSIONS_DIR}/DX11Renderer"
  ${CMAKE_MODULE_PATH}
)

find_package(DirectX11)

if (DIRECTX11_FOUND AND WIN32)
  set(ROOT_SRC
    DX11Renderer/CameraEffects.h
    DX11Renderer/CameraEffects.cpp
    DX11Renderer/Compositor.h
    DX11Renderer/Compositor.cpp
    DX11Renderer/ConstantBufferArray.h
    DX11Renderer/Defines.h
    DX11Renderer/OwnedResourcePtr.h
    DX11Renderer/PrefixSum.h
    DX11Renderer/Renderer.h
    DX11Renderer/Renderer.cpp
    DX11Renderer/SSAO.h
    DX11Renderer/SSAO.cpp
    DX11Renderer/Types.h
    DX11Renderer/Utils.h
    DX11Renderer/Utils.cpp
  )

  set(MANAGERS_SRC
    DX11Renderer/Managers/EnvironmentManager.h
    DX11Renderer/Managers/EnvironmentManager.cpp
    DX11Renderer/Managers/LightManager.cpp
    DX11Renderer/Managers/LightManager.h
    DX11Renderer/Managers/MaterialManager.h
    DX11Renderer/Managers/MaterialManager.cpp
    DX11Renderer/Managers/MeshManager.h
    DX11Renderer/Managers/MeshManager.cpp
    DX11Renderer/Managers/MeshModelManager.h
    DX11Renderer/Managers/MeshModelManager.cpp
    DX11Renderer/Managers/ShaderManager.h
    DX11Renderer/Managers/ShaderManager.cpp
    DX11Renderer/Managers/TextureManager.h
    DX11Renderer/Managers/TextureManager.cpp
    DX11Renderer/Managers/TransformManager.h
    DX11Renderer/Managers/TransformManager.cpp
  )

  set(SHADERS_SRC
    DX11Renderer/Shaders/Debug.hlsl
    DX11Renderer/Shaders/EnvironmentMap.hlsl
    DX11Renderer/Shaders/IBLConvolution.hlsl
    DX11Renderer/Shaders/LightSources.hlsl
    DX11Renderer/Shaders/ModelGBuffer.hlsl
    DX11Renderer/Shaders/ModelShading.hlsl
    DX11Renderer/Shaders/RNG.hlsl
    DX11Renderer/Shaders/SphereLight.hlsl
    DX11Renderer/Shaders/SSAO.hlsl
    DX11Renderer/Shaders/Utils.hlsl
  )

  set(SHADERS_BSDFS_SRC
    DX11Renderer/Shaders/BSDFs/Diffuse.hlsl
    DX11Renderer/Shaders/BSDFs/GGX.hlsl
  )

  set(SHADERS_CAMERAEFFECTS_SRC
    DX11Renderer/Shaders/CameraEffects/Bloom.hlsl
    DX11Renderer/Shaders/CameraEffects/ReduceExposureHistogram.hlsl
    DX11Renderer/Shaders/CameraEffects/ReduceLogAverageLuminance.hlsl
    DX11Renderer/Shaders/CameraEffects/Tonemapping.hlsl
    DX11Renderer/Shaders/CameraEffects/Utils.hlsl
  )

  set(SHADERS_COMPUTE_SRC
    DX11Renderer/Shaders/Compute/PrefixSum.hlsl
  )

  set(SHADERS_SHADING_MODELS_SRC
    DX11Renderer/Shaders/ShadingModels/DefaultShading.hlsl
    DX11Renderer/Shaders/ShadingModels/DiffuseShading.hlsl
    DX11Renderer/Shaders/ShadingModels/IShadingModel.hlsl
    DX11Renderer/Shaders/ShadingModels/Utils.hlsl
  )

  set(ALL_SHADER_SRC ${SHADERS_SRC} ${SHADERS_BSDFS_SRC} ${SHADERS_CAMERAEFFECTS_SRC} ${SHADERS_COMPUTE_SRC} ${SHADERS_SHADING_MODELS_SRC})

  add_library(${LIBRARY_NAME}
    ${ROOT_SRC}
    ${MANAGERS_SRC}
    ${ALL_SHADER_SRC}
  )

  source_group("" FILES ${ROOT_SRC})
  source_group("Managers" FILES ${MANAGERS_SRC})
  source_group("Shaders" FILES ${SHADERS_SRC})
  source_group("Shaders\\BSDFs" FILES ${SHADERS_BSDFS_SRC})
  source_group("Shaders\\CameraEffects" FILES ${SHADERS_CAMERAEFFECTS_SRC})
  source_group("Shaders\\Compute" FILES ${SHADERS_COMPUTE_SRC})
  source_group("Shaders\\ShadingModels" FILES ${SHADERS_SHADING_MODELS_SRC})

  target_include_directories(${LIBRARY_NAME} 
    PUBLIC . 
    PRIVATE ${DIRECTX11_INCLUDE_DIR}
  )

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC Bifrost
    PRIVATE ${DIRECTX11_LIBRARIES}
  )

  set_target_properties(${LIBRARY_NAME} PROPERTIES 
    LINKER_LANGUAGE CXX
    FOLDER "Extensions"
  )

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast")

  # set_source_files_properties(
  #   ${VERTEX_SHADER_SRC} PROPERTIES VS_SHADER_TYPE Vertex VS_SHADER_MODEL 5.0 VS_SHADER_ENTRYPOINT main
  # )

  # set_source_files_properties(
  #   ${PIXEL_SHADER_SRC} PROPERTIES VS_SHADER_TYPE Pixel VS_SHADER_MODEL 5.0 VS_SHADER_ENTRYPOINT main
  # )

  # Copy the shaders to the build directory.
  # TODO Copy or compile the cso files from release to bin output dir.
  foreach (SHADER_PATH ${ALL_SHADER_SRC})
    set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_PATH})
    set(DESTINATION_FILE ${BIFROST_DATA_OUTPUT_DIRECTORY}/${SHADER_PATH})

    add_custom_command(
      OUTPUT ${DESTINATION_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SOURCE_FILE} ${DESTINATION_FILE}
      COMMAND ${CMAKE_COMMAND} -E echo "Copy ${SHADER_PATH} to ${DESTINATION_FILE}"
      MAIN_DEPENDENCY ${SOURCE_FILE})
  endforeach()
else()
  if (NOT DIRECTX11_FOUND)
    message(WARNING "DX11Renderer: DirectX 11 not found.")
  endif()
  if (NOT WIN32)
    message(WARNING "DX11Renderer: DirectX 11 only supported on windows.")
  endif()
endif()