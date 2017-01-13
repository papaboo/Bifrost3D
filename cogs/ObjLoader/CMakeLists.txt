add_library(ObjLoader 
  ObjLoader/ObjLoader.h
  ObjLoader/ObjLoader.cpp
  ObjLoader/tiny_obj_loader.h
)

target_include_directories(ObjLoader PUBLIC .)

target_link_libraries(ObjLoader PUBLIC Cogwheel)

source_group("ObjLoader" FILES 
  ObjLoader/ObjLoader.h
  ObjLoader/ObjLoader.cpp
  ObjLoader/tiny_obj_loader.h
)

set_target_properties(ObjLoader PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)