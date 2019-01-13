add_library(TinyExr
  TinyExr/TinyExr.h
  TinyExr/TinyExr.cpp
  TinyExr/tiny_exr.h
)

target_include_directories(TinyExr PUBLIC .)

target_link_libraries(TinyExr PUBLIC Bifrost)

source_group("TinyExr" FILES 
  TinyExr/TinyExr.h
  TinyExr/TinyExr.cpp
  TinyExr/tiny_exr.h
)

set_target_properties(TinyExr PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Extensions"
)
