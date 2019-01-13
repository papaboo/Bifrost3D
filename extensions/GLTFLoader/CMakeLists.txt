add_library(GLTFLoader
  GLTFLoader/json.hpp
  GLTFLoader/GLTFLoader.h
  GLTFLoader/GLTFLoader.cpp
  GLTFLoader/tiny_gltf.h
)

target_include_directories(GLTFLoader PUBLIC .)

target_link_libraries(GLTFLoader PUBLIC Bifrost PRIVATE StbImageLoader)

source_group("GLTFLoader" FILES 
  GLTFLoader/json.hpp
  GLTFLoader/GLTFLoader.h
  GLTFLoader/GLTFLoader.cpp
  GLTFLoader/tiny_gltf.h
)

set_target_properties(GLTFLoader PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Extensions"
)
