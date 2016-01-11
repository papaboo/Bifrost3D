set(PROJECT_NAME "SmallPT")

set(SRCS 
  main.cpp
  smallpt.h
)

add_executable(${PROJECT_NAME} ${SRCS})

include_directories(.)

source_group("" FILES ${SRCS})

# Find OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})
add_definitions(${OpenGL_DEFINITIONS})

target_link_libraries(${PROJECT_NAME}
  Gogwheel
  glfw
  ${OPENGL_LIBRARIES}
)

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps"
)