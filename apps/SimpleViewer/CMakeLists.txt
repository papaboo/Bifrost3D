set(PROJECT_NAME "SimpleViewer")

set(SRCS 
  CameraHandlers.h
  CameraHandlers.cpp
  main.cpp
)

set(GUI_SRCS
  GUI/RenderingGUI.h
  GUI/RenderingGUI.cpp
)

set(SCENES_SRCS
  Scenes/CornellBox.h
  Scenes/Diamond.glb
  Scenes/Glass.h
  Scenes/Glass.cpp
  Scenes/Material.h
  Scenes/Material.cpp
  Scenes/Opacity.h
  Scenes/Shaderball.glb
  Scenes/Sphere.cpp
  Scenes/Sphere.h
  Scenes/SphereLight.h
  Scenes/Test.h
  Scenes/Utils.h
  Scenes/Utils.cpp
  Scenes/Veach.h
  Scenes/WorldMask.png
)

add_executable(${PROJECT_NAME} ${SRCS} ${GUI_SRCS} ${SCENES_SRCS})

source_group("" FILES ${SRCS})
source_group("GUI" FILES ${GUI_SRCS})
source_group("Scenes" FILES ${SCENES_SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE .)

target_link_libraries(${PROJECT_NAME}
  DX11Renderer
  Bifrost
  glTFLoader
  ImGui
  ObjLoader
  StbImageLoader
  StbImageWriter
  Win32Driver
)

if (TARGET DX11OptiXAdaptor)
  add_definitions(-DOPTIX_FOUND)
  target_link_libraries(${PROJECT_NAME} DX11OptiXAdaptor OptiXRenderer)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps"
)

# Install shader ball and diamond
set(SHADER_BALL_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Scenes/Shaderball.glb)
set(SHADER_BALL_DESTINATION_PATH ${BIFROST_DATA_OUTPUT_DIRECTORY}/SimpleViewer/Shaderball.glb)
add_custom_command(
  OUTPUT ${SHADER_BALL_DESTINATION_PATH}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SHADER_BALL_SOURCE_PATH} ${SHADER_BALL_DESTINATION_PATH}
  MAIN_DEPENDENCY ${SHADER_BALL_SOURCE_PATH}
)

set(WORLD_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Scenes/WorldMask.png)
set(WORLD_DESTINATION_PATH ${BIFROST_DATA_OUTPUT_DIRECTORY}/SimpleViewer/WorldMask.png)
add_custom_command(
  OUTPUT ${WORLD_DESTINATION_PATH}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${WORLD_SOURCE_PATH} ${WORLD_DESTINATION_PATH}
  MAIN_DEPENDENCY ${WORLD_SOURCE_PATH}
)

set(DIAMOND_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/Scenes/Diamond.glb)
set(DIAMOND_DESTINATION_PATH ${BIFROST_DATA_OUTPUT_DIRECTORY}/SimpleViewer/Diamond.glb)
add_custom_command(
  OUTPUT ${DIAMOND_DESTINATION_PATH}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${DIAMOND_SOURCE_PATH} ${DIAMOND_DESTINATION_PATH}
  MAIN_DEPENDENCY ${DIAMOND_SOURCE_PATH}
)

if (MSVC)
  create_project_VS_user_file(${PROJECT_NAME})
endif()