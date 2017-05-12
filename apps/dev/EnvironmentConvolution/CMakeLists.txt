set(PROJECT_NAME "EnvironmentConvolution")

find_package(OpenGL REQUIRED)

set(SRCS main.cpp)

add_executable(${PROJECT_NAME} ${SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE . ${OpenGL_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
  Cogwheel
  GLFWDriver
  StbImageLoader
  StbImageWriter
  TinyExr
  ${OPENGL_LIBRARIES}
)

source_group("" FILES ${SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps/Dev"
)