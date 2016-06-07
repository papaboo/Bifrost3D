include_directories(.)

add_library(StbImageWriter 
  StbImageWriter/StbImageWriter.h
  StbImageWriter/StbImageWriter.cpp
  StbImageWriter/stb_image_write.h
)

TARGET_LINK_LIBRARIES(StbImageWriter
  Gogwheel
)

source_group("StbImageWriter" FILES 
  StbImageWriter/StbImageWriter.h
  StbImageWriter/StbImageWriter.cpp
  StbImageWriter/stb_image_write.h
)

set_target_properties(StbImageWriter PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)