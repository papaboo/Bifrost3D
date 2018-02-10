add_library(ImageOperations 
  ImageOperations/Compare.h
  ImageOperations/Exposure.h
  ImageOperations/Exposure.cpp
)

target_include_directories(ImageOperations PUBLIC .)

target_link_libraries(ImageOperations
  PUBLIC Cogwheel
)

source_group("ImageOperations" FILES 
  ImageOperations/Compare.h
  ImageOperations/Exposure.h
  ImageOperations/Exposure.cpp
)

set_target_properties(ImageOperations PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)