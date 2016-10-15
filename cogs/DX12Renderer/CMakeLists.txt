set(LIBRARY_NAME "DX12Renderer")

# Add current path to CMake's module path, so we can use our own FindWindowsSDK.cmake.
set(CMAKE_MODULE_PATH
  "${COGWHEEL_COGS_DIR}/DX12Renderer"
  ${CMAKE_MODULE_PATH}
)

find_package(DirectX12)

if (DIRECTX_12_FOUND AND WIN32)
  set(ROOT_SRC
    DX12Renderer/Renderer.h
    DX12Renderer/Renderer.cpp
  )

  set(VERTEX_SHADER_SRC
    DX12Renderer/Shaders/VertexShader.hlsl
  )

  set(PIXEL_SHADER_SRC
    DX12Renderer/Shaders/FragmentShader.hlsl
  )
  
  set(SHADER_SRC ${VERTEX_SHADER_SRC} ${PIXEL_SHADER_SRC})

  add_library(${LIBRARY_NAME}
    ${ROOT_SRC}
    ${SHADER_SRC}
  )

  source_group("" FILES ${ROOT_SRC})
  source_group("Shaders" FILES ${SHADER_SRC})

  target_include_directories(${LIBRARY_NAME} 
    PUBLIC . 
    PRIVATE ${DIRECTX_12_INCLUDE_DIRS}
  )

  target_link_libraries(${LIBRARY_NAME}
    PUBLIC Gogwheel
    PRIVATE ${DIRECTX_12_LIBRARIES}
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
  # TODO Copy the cso files from release to bin output dir.
  foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
    set(DATA_PATH "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CONFIG}/Data")
    file(MAKE_DIRECTORY ${DATA_PATH}/Shaders)
    foreach (SHADER_PATH ${SHADER_SRC})
      string(SUBSTRING ${SHADER_PATH} 13 -1 SHORT_SHADER_PATH)
      set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_PATH})
      set(DESTINATION_FILE ${DATA_PATH}/${SHORT_SHADER_PATH})

      add_custom_command(
        OUTPUT ${DESTINATION_FILE}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SOURCE_FILE} ${DESTINATION_FILE}
        MAIN_DEPENDENCY ${SOURCE_FILE})
    endforeach()
  endforeach ()
else()
  if (NOT DIRECTX_12_FOUND)
    message(WARNING "DX12Renderer: DirectX 12 not found.")
  endif()
  if (NOT WIN32)
    message(WARNING "DX12Renderer: DirectX 12 only supported on windows.")
  endif()
endif()