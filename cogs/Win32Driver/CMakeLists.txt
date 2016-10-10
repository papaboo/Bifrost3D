add_library(Win32Driver Win32Driver.h Win32Driver.cpp)

target_include_directories(Win32Driver PUBLIC .)

target_link_libraries(Win32Driver
  PUBLIC Gogwheel
)

source_group("" FILES Win32Driver.h Win32Driver.cpp)

set_target_properties(Win32Driver PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)