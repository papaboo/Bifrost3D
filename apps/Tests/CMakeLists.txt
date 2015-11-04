set(PROJECT_NAME "Tests")

set(SRCS 
  main.cpp
)

set(CORE_SRCS
  Core/ArrayTest.h
  Core/UniqueIDGeneratorTest.h
)

set(MATH_SRCS
  Math/QuaternionTest.h
  Math/TransformTest.h
)

set(SCENE_SRCS
  Scene/SceneNodeTest.h
  Scene/TransformTest.h
)

add_executable(${PROJECT_NAME} ${SRCS} ${CORE_SRCS} ${MATH_SRCS} ${SCENE_SRCS})

include_directories(.)

target_link_libraries(${PROJECT_NAME}
  Gogwheel
  gtest
)

source_group("" FILES ${SRCS})
source_group("Core" FILES ${CORE_SRCS})
source_group("Math" FILES ${MATH_SRCS})
source_group("Scene" FILES ${SCENE_SRCS})

# setup_gtest(${PROJECT_NAME})
