include_directories(.)

add_library(GLFWDriver GLFWDriver.h)

SOURCE_GROUP("" FILES GLFWDriver.h)

set_target_properties(GLFWDriver PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)