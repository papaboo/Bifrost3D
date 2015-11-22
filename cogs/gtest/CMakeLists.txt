include_directories(.)

set(SRCS
  gtest/gtest.h
  gtest/gtest-all.cpp
)

SOURCE_GROUP("gtest" FILES ${SRCS})

add_library(gtest ${SRCS})

set_target_properties(gtest PROPERTIES LINKER_LANGUAGE CXX) 

set_target_properties(gtest PROPERTIES
  FOLDER "Cogs"
)