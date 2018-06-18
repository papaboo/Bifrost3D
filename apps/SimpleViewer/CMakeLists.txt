set(PROJECT_NAME "SimpleViewer")

set(SRCS 
  CornellBoxScene.h
  main.cpp
  MaterialScene.h
  OpacityScene.h
  SphereLightScene.h
  SphereScene.h
  TestScene.h
  VeachScene.h
)

set(GUI_SRCS
  GUI/RenderingGUI.h
  GUI/RenderingGUI.cpp
)

add_executable(${PROJECT_NAME} ${SRCS} ${GUI_SRCS})

source_group("" FILES ${SRCS})
source_group("GUI" FILES ${GUI_SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE .)

target_link_libraries(${PROJECT_NAME}
  DX11Renderer
  Cogwheel
  ImGui
  ObjLoader
  StbImageLoader
  Win32Driver
)

if (TARGET DX11OptiXAdaptor)
  add_definitions(-DOPTIX_FOUND)
  target_link_libraries(${PROJECT_NAME} DX11OptiXAdaptor OptiXRenderer)
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps"
)

if (MSVC)
  create_project_VS_user_file(${PROJECT_NAME})
endif()