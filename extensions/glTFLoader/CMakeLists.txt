set(GLTF_LOADER_SRCS
  glTFLoader/json.hpp
  glTFLoader/glTFLoader.h
  glTFLoader/glTFLoader.cpp
  glTFLoader/tiny_gltf.h
)

add_library(glTFLoader ${GLTF_LOADER_SRCS})

target_include_directories(glTFLoader PUBLIC .)

target_link_libraries(glTFLoader PUBLIC Bifrost PRIVATE StbImageLoader)

source_group("glTFLoader" FILES ${GLTF_LOADER_SRCS})

set_target_properties(glTFLoader PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Extensions"
)
