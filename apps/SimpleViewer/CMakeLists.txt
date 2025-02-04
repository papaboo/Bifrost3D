set(PROJECT_NAME "SimpleViewer")

set(SRCS 
  CameraHandlers.h
  CameraHandlers.cpp
  main.cpp
)

set(RESOURCE_FILES
  Resources/Diamond.glb
  Resources/Shaderball.gltf
  Resources/WorldMask.png
)

set(GUI_SRCS
  GUI/RenderingGUI.h
  GUI/RenderingGUI.cpp
)

set(SCENES_SRCS
  Scenes/CornellBox.h
  Scenes/Glass.h
  Scenes/Glass.cpp
  Scenes/Material.h
  Scenes/Material.cpp
  Scenes/Opacity.h
  Scenes/Sphere.cpp
  Scenes/Sphere.h
  Scenes/SphereLight.h
  Scenes/Test.h
  Scenes/Utils.h
  Scenes/Utils.cpp
  Scenes/Veach.h
)

add_executable(${PROJECT_NAME} ${SRCS} ${RESOURCE_FILES} ${GUI_SRCS} ${SCENES_SRCS})

source_group("" FILES ${SRCS})
source_group("Resources" FILES ${RESOURCE_FILES})
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

# Install resources
foreach (RESOURCE_FILE_PATH ${RESOURCE_FILES})
  set(SOURCE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/${RESOURCE_FILE_PATH})
  set(DESTINATION_FILE ${BIFROST_DATA_OUTPUT_DIRECTORY}/SimpleViewer/${RESOURCE_FILE_PATH})
  
  add_custom_command(
    OUTPUT ${DESTINATION_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SOURCE_FILE} ${DESTINATION_FILE}
    COMMAND ${CMAKE_COMMAND} -E echo "Copy ${RESOURCE_FILE_PATH} to ${DESTINATION_FILE}"
    MAIN_DEPENDENCY ${SOURCE_FILE})
endforeach()

if (MSVC)
  create_project_VS_user_file(${PROJECT_NAME})
endif()