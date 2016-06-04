set(PROJECT_NAME "CogwheelTests")

set(SRCS 
  Expects.h
  main.cpp
)

set(ASSETS_SRCS
  Assets/ImageTest.h
  Assets/MaterialTest.h
  Assets/MeshModelTest.h
  Assets/MeshTest.h
  Assets/TextureTest.h
)

set(CORE_SRCS
  Core/ArrayTest.h
  Core/UniqueIDGeneratorTest.h
)

set(INPUT_SRCS
  Input/KeyboardTest.h
)

set(MATH_SRCS
  Math/MatrixTest.h
  Math/QuaternionTest.h
  Math/TransformTest.h
)

set(SCENE_SRCS
  Scene/CameraTest.h
  Scene/LightSourceTest.h
  Scene/SceneNodeTest.h
  Scene/TransformTest.h
)

add_executable(${PROJECT_NAME} ${SRCS} ${ASSETS_SRCS} ${CORE_SRCS} ${INPUT_SRCS} ${MATH_SRCS} ${SCENE_SRCS})

include_directories(.)

target_link_libraries(${PROJECT_NAME}
  gtest
  Gogwheel
)

source_group("" FILES ${SRCS})
source_group("Assets" FILES ${ASSETS_SRCS})
source_group("Core" FILES ${CORE_SRCS})
source_group("Input" FILES ${INPUT_SRCS})
source_group("Math" FILES ${MATH_SRCS})
source_group("Scene" FILES ${SCENE_SRCS})

# setup_gtest(${PROJECT_NAME})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Tests"
)