include_directories(.)

add_library(StbImageLoader 
  StbImageLoader/StbImageLoader.h
  StbImageLoader/StbImageLoader.cpp
  StbImageLoader/stb_image.h
)

TARGET_LINK_LIBRARIES(StbImageLoader
  Gogwheel
)

source_group("StbImageLoader" FILES 
  StbImageLoader/StbImageLoader.h
  StbImageLoader/StbImageLoader.cpp
  StbImageLoader/stb_image.h
)

set_target_properties(StbImageLoader PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)