add_library(glTFLoader
  glTFLoader/json.hpp
  glTFLoader/GLTFLoader.h
  glTFLoader/GLTFLoader.cpp
  glTFLoader/tiny_gltf.h
)

target_include_directories(glTFLoader PUBLIC .)

target_link_libraries(glTFLoader PUBLIC Bifrost PRIVATE StbImageLoader)

source_group("glTFLoader" FILES 
  glTFLoader/json.hpp
  glTFLoader/glTFLoader.h
  glTFLoader/glTFLoader.cpp
  glTFLoader/tiny_gltf.h
)

set_target_properties(glTFLoader PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Extensions"
)
