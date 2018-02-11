set(PROJECT_NAME "Komodo")

find_package(OpenGL REQUIRED)

set(SRCS main.cpp Comparer.h Tonemapper.h Tonemapper.cpp Utils.h)

add_executable(${PROJECT_NAME} ${SRCS})

target_include_directories(${PROJECT_NAME} PRIVATE . ${OpenGL_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME}
  Cogwheel
  AntTweakBar
  GLFWDriver
  ImageOperations
  StbImageLoader
  StbImageWriter
  TinyExr
  ${OPENGL_LIBRARIES}
)

source_group("" FILES ${SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Apps"
)