set(PROJECT_NAME "BifrostTests")

set(SRCS 
  Expects.h
  main.cpp
)

set(ASSETS_SRCS
  Assets/ImageTest.h
  Assets/InfiniteAreaLightTest.h
  Assets/MaterialTest.h
  Assets/MediaTest.h
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
  Math/ColorTest.h
  Math/Distribution1DTest.h
  Math/Distribution2DTest.h
  Math/FixedPointTypesTest.h
  Math/ImageSamplingTest.h
  Math/LineTest.h
  Math/MatrixTest.h
  Math/OctahedralNormalTest.h
  Math/QuaternionTest.h
  Math/RNGTest.h
  Math/StatisticsTest.h
  Math/TransformTest.h
  Math/TypeTraitsTest.h
  Math/UtilsTest.h
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
target_link_libraries(${PROJECT_NAME} gtest Bifrost)

source_group("" FILES ${SRCS})
source_group("Assets" FILES ${ASSETS_SRCS})
source_group("Core" FILES ${CORE_SRCS})
source_group("Input" FILES ${INPUT_SRCS})
source_group("Math" FILES ${MATH_SRCS})
source_group("Scene" FILES ${SCENE_SRCS})

set_target_properties(${PROJECT_NAME} PROPERTIES
  FOLDER "Tests"
)