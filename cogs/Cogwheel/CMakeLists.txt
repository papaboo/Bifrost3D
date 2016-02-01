include_directories(.)

include("Cogwheel/Assets/Setup.cmake")
include("Cogwheel/Core/Setup.cmake")
include("Cogwheel/Input/Setup.cmake")
include("Cogwheel/Math/Setup.cmake")
include("Cogwheel/Scene/Setup.cmake")

add_library(Gogwheel ${ASSETS_SRCS} ${CORE_SRCS} ${INPUT_SRCS} ${MATH_SRCS} ${SCENE_SRCS})

set_target_properties(Gogwheel PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)