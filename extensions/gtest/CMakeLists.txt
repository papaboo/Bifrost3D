set(SRCS
  gtest/gtest.h
  gtest/gtest-all.cpp
)

source_group("gtest" FILES ${SRCS})

add_library(gtest ${SRCS})

target_include_directories(gtest PUBLIC .)

# Disable TR1 tuples as they are unused  and VS2017 warns about them.
target_compile_definitions(gtest PUBLIC GTEST_HAS_TR1_TUPLE=0)

set_target_properties(gtest PROPERTIES
  LINKER_LANGUAGE CXX
  FOLDER "Libs"
)