include_directories(.)

add_library(Win32Driver Win32Driver.h Win32Driver.cpp)

TARGET_LINK_LIBRARIES(Win32Driver
  Gogwheel
)

source_group("" FILES Win32Driver.h Win32Driver.cpp)

set_target_properties(Win32Driver PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs"
)