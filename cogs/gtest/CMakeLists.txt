set(SRCS
  gtest/gtest.h
  gtest/gtest-all.cpp
)

source_group("gtest" FILES ${SRCS})

add_library(gtest ${SRCS})

target_include_directories(gtest PUBLIC .)

set_target_properties(gtest PROPERTIES
  LINKER_LANGUAGE CXX
  FOLDER "Libs"
)