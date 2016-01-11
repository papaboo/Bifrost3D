include_directories(.)

set(SRCS
  gtest/gtest.h
  gtest/gtest-all.cpp
)

source_group("gtest" FILES ${SRCS})

add_library(gtest ${SRCS})

set_target_properties(gtest PROPERTIES LINKER_LANGUAGE CXX) 

set_target_properties(gtest PROPERTIES
  FOLDER "Libs"
)