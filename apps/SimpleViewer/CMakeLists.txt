set(PROJECT_NAME "SimpleViewer")

include_directories(.)

set(SRCS 
  CornellBoxScene.h
  main.cpp
  MaterialScene.h
  SphereScene.h
  TestScene.h
)

add_executable(${PROJECT_NAME} ${SRCS})

source_group("" FILES ${SRCS})

target_link_libraries(${PROJECT_NAME}
  Gogwheel
  GLFWDriver
  ObjLoader
  OptiXRenderer
  StbImageLoader
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps"
)