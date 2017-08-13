add_library(ImageOperations 
  ImageOperations/Diff.h
  ImageOperations/Diff.cpp
)

target_include_directories(ImageOperations PUBLIC .)

target_link_libraries(ImageOperations PUBLIC Cogwheel)

source_group("ImageOperations" FILES 
  ImageOperations/Diff.h
  ImageOperations/Diff.cpp
)

set_target_properties(ImageOperations PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)