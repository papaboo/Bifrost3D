set(PROJECT_NAME "ImageOperationsTests")

set(SRCS 
  BlurTest.h
  CompareTest.h
  main.cpp
  StatisticsTest.h
)

add_executable(${PROJECT_NAME} ${SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE .)
target_link_libraries(${PROJECT_NAME} gtest Cogwheel ImageOperations)

source_group("" FILES ${SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Tests"
)