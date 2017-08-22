# Setup ImageOperations as a header-only / INTERFACE library.

add_library(ImageOperations INTERFACE)

target_include_directories(ImageOperations INTERFACE .)

target_link_libraries(ImageOperations INTERFACE Cogwheel)


# HACK: INTERFACE libraries aren't shown in the IDE, so we create a custom target containing the sources.
# This should've been what target_sources does, but instead it adds the headers to each solution that links with the library.

# target_sources(ImageOperations INTERFACE
#   ${CMAKE_CURRENT_SOURCE_DIR}/ImageOperations/Compare.h
# )

add_custom_target(ImageOperationsIDE SOURCES ImageOperations/Compare.h)

source_group("ImageOperations" FILES 
  ImageOperations/Compare.h
)

set_target_properties(ImageOperationsIDE PROPERTIES 
  FOLDER "Cogs"
  PROJECT_LABEL "ImageOperations"
  LINKER_LANGUAGE CXX
)