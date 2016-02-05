set(PROJECT_NAME "SimpleViewer")

set(SRCS 
  main.cpp
)

add_executable(${PROJECT_NAME} ${SRCS})

source_group("" FILES ${SRCS})

target_link_libraries(${PROJECT_NAME}
  Gogwheel
  GLFWDriver
  ObjLoader
  OptiXRenderer
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps"
)