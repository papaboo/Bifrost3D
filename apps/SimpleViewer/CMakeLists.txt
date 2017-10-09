set(PROJECT_NAME "SimpleViewer")

set(SRCS 
  CornellBoxScene.h
  main.cpp
  MaterialScene.h
  OpacityScene.h
  SphereScene.h
  TestScene.h
  VeachScene.h
)

add_executable(${PROJECT_NAME} ${SRCS})

source_group("" FILES ${SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE .)

target_link_libraries(${PROJECT_NAME}
  DX11Renderer
  Cogwheel
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