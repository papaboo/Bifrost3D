set(CMAKE_MODULE_PATH "${COGWHEEL_COGS_DIR}/AntTweakBar" ${CMAKE_MODULE_PATH})
find_package(AntTweakBar REQUIRED)

add_library(AntTweakBar AntTweakBar/AntTweakBar.h AntTweakBar/AntTweakBar.cpp)

target_include_directories(AntTweakBar PUBLIC . ${ANT_TWEAK_BAR_INCLUDE_DIRS})

target_link_libraries(AntTweakBar PUBLIC Cogwheel ${ANT_TWEAK_BAR_LIBRARIES} )

source_group("AntTweakBar" FILES AntTweakBar/AntTweakBar.h AntTweakBar/AntTweakBar.cpp)

set_target_properties(AntTweakBar PROPERTIES 
  LINKER_LANGUAGE CXX
  FOLDER "Cogs")
