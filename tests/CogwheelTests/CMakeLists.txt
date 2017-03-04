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
  Core/BitmaskTest.h
  Core/UniqueIDGeneratorTest.h
)

set(INPUT_SRCS
  Input/KeyboardTest.h
)

set(MATH_SRCS
  Math/Distribution2DTest.h
  Math/MatrixTest.h
  Math/QuaternionTest.h
  Math/TransformTest.h
)

set(SCENE_SRCS
  Scene/CameraTest.h
  Scene/LightSourceTest.h
  Scene/SceneNodeTest.h
  Scene/SceneRootTest.h
  Scene/TransformTest.h
)

add_executable(${PROJECT_NAME} ${SRCS} ${ASSETS_SRCS} ${CORE_SRCS} ${INPUT_SRCS} ${MATH_SRCS} ${SCENE_SRCS})
target_include_directories(${PROJECT_NAME} PRIVATE .)
target_link_libraries(${PROJECT_NAME} gtest Cogwheel)

source_group("" FILES ${SRCS})
source_group("Assets" FILES ${ASSETS_SRCS})
source_group("Core" FILES ${CORE_SRCS})
source_group("Input" FILES ${INPUT_SRCS})
source_group("Math" FILES ${MATH_SRCS})
source_group("Scene" FILES ${SCENE_SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Tests"
)