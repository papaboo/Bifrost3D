set(LIBRARY_NAME "DX11Renderer")

# Add current path to CMake's module path, so we can use our own FindDirectX11.cmake.
set(CMAKE_MODULE_PATH
  "${COGWHEEL_COGS_DIR}/DX11Renderer"
  ${CMAKE_MODULE_PATH}
)

find_package(DirectX11)

if (DIRECTX_11_FOUND AND WIN32)
  set(ROOT_SRC
    DX11Renderer/LightManager.h
    DX11Renderer/Renderer.h
    DX11Renderer/Renderer.cpp
    DX11Renderer/Types.h
  )

  set(VERTEX_SHADER_SRC
    DX11Renderer/Shaders/VertexShader.hlsl
  )

  set(PIXEL_SHADER_SRC
    DX11Renderer/Shaders/FragmentShader.hlsl
  )
  
  set(SHADER_INCLUDES
    DX11Renderer/Shaders/LightSources.hlsl
  )
  
  set(SHADER_SRC ${VERTEX_SHADER_SRC} ${PIXEL_SHADER_SRC} ${SHADER_INCLUDES})

  add_library(${LIBRARY_NAME}
    ${ROOT_SRC}
    ${SHADER_SRC}
  )

  source_group("" FILES ${ROOT_SRC})
  source_group("Shaders" FILES ${SHADER_SRC})

  target_include_directories(${LIBRARY_NAME} 
    PUBLIC . 
    PRIVATE ${DIRECTX_11_INCLUDE_DIRS}
  )

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC Gogwheel
    PRIVATE ${DIRECTX_11_LIBRARIES}
  )

  set_target_properties(${LIBRARY_NAME} PROPERTIES 
    LINKER_LANGUAGE CXX
    FOLDER "Cogs"
  )

  set_source_files_properties(
    ${VERTEX_SHADER_SRC} PROPERTIES VS_SHADER_TYPE Vertex VS_SHADER_MODEL 5.0 VS_SHADER_ENTRYPOINT main
  )

  set_source_files_properties(
    ${PIXEL_SHADER_SRC} PROPERTIES VS_SHADER_TYPE Pixel VS_SHADER_MODEL 5.0 VS_SHADER_ENTRYPOINT main
  )

  # Copy the shaders to the build directory.
  # TODO Is there a general concept of installation in CMake?
  # TODO Copy or compile the cso files from release to bin output dir.
  set(DATA_PATH "${COGWHEEL_DATA_OUTPUT_DIRECTORY}/DX11Renderer/Shaders")
  file(MAKE_DIRECTORY ${DATA_PATH})
  foreach (SHADER_PATH ${SHADER_SRC})
    set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_PATH})
    set(DESTINATION_FILE ${COGWHEEL_DATA_OUTPUT_DIRECTORY}/${SHADER_PATH})

    add_custom_command(
      OUTPUT ${DESTINATION_FILE}
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SOURCE_FILE} ${DESTINATION_FILE}
      MAIN_DEPENDENCY ${SOURCE_FILE})
  endforeach()
else()
  if (NOT DIRECTX_11_FOUND)
    message(WARNING "DX11Renderer: DirectX 11 not found.")
  endif()
  if (NOT WIN32)
    message(WARNING "DX11Renderer: DirectX 11 only supported on windows.")
  endif()
endif()