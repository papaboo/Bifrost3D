set(PROJECT_NAME "SmallPT")

find_package(OpenGL REQUIRED)

set(SRCS 
  main.cpp
  smallpt.h
)

add_executable(${PROJECT_NAME} ${SRCS})

target_include_directories(${PROJECT_NAME}
  PUBLIC .
  PRIVATE "${GLFW_LIBS_DIR}/include" ${OpenGL_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
  Bifrost
  glfw
  ${OPENGL_LIBRARIES}
)

source_group("" FILES ${SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps"
)