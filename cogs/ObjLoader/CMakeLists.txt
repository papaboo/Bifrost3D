include_directories(.)

add_library(ObjLoader 
  ObjLoader/ObjLoader.h
  ObjLoader/ObjLoader.cpp
  ObjLoader/tiny_obj_loader.h
)

TARGET_LINK_LIBRARIES(ObjLoader
  Gogwheel
)

source_group("ObjLoader" FILES 
  ObjLoader/ObjLoader.h
  ObjLoader/ObjLoader.cpp
  ObjLoader/tiny_obj_loader.h
)

set_target_properties(ObjLoader PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)