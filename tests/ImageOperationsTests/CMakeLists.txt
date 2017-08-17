set(PROJECT_NAME "ImageOperationsTests")

set(SRCS 
  DiffTest.h
  Expects.h
  main.cpp
)

add_executable(${PROJECT_NAME} ${SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE .)
target_link_libraries(${PROJECT_NAME} gtest Cogwheel ImageOperations)

source_group("" FILES ${SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Tests"
)