include_directories(.)

include("Core/Setup.cmake")
include("Math/Setup.cmake")
include("Scene/Setup.cmake")

add_library(Gogwheel ${CORE_SRCS} ${MATH_SRCS} ${SCENE_SRCS})

set_target_properties(Gogwheel PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)