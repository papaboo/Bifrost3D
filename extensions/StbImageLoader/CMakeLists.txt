add_library(StbImageLoader 
  StbImageLoader/StbImageLoader.h
  StbImageLoader/StbImageLoader.cpp
  StbImageLoader/stb_image.h
)

target_include_directories(StbImageLoader PUBLIC .)

target_link_libraries(StbImageLoader
  PUBLIC Bifrost
)

source_group("StbImageLoader" FILES 
  StbImageLoader/StbImageLoader.h
  StbImageLoader/StbImageLoader.cpp
  StbImageLoader/stb_image.h
)

set_target_properties(StbImageLoader PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Extensions"
)