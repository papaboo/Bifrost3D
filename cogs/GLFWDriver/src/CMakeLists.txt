add_library(GLFWDriver GLFWDriver.h GLFWDriver.cpp)

target_include_directories(GLFWDriver
  PUBLIC .
  PRIVATE "${GLFW_LIBS_DIR}/include"
)

target_link_libraries(GLFWDriver
  PUBLIC Cogwheel
  PRIVATE glfw
)

source_group("" FILES GLFWDriver.h GLFWDriver.cpp)

set_target_properties(GLFWDriver PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)