include_directories(.)

add_library(GLFWDriver GLFWDriver.h GLFWDriver.cpp)

TARGET_LINK_LIBRARIES(GLFWDriver
  Gogwheel
  glfw
)


source_group("" FILES GLFWDriver.h GLFWDriver.cpp)

set_target_properties(GLFWDriver PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)