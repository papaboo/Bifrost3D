// Bifrost unit tests.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <Assets/ImageTest.h>
#include <Assets/InfiniteAreaLightTest.h>
#include <Assets/MaterialTest.h>
#include <Assets/MeshTest.h>
#include <Assets/MeshModelTest.h>
#include <Assets/TextureTest.h>

#include <Core/ArrayTest.h>
#include <Core/BitmaskTest.h>
#include <Core/UniqueIDGeneratorTest.h>

#include <Input/KeyboardTest.h>

#include <Math/Distribution1DTest.h>
#include <Math/Distribution2DTest.h>
#include <Math/MatrixTest.h>
#include <Math/OctahedralNormalTest.h>
#include <Math/QuaternionTest.h>
#include <Math/StatisticsTest.h>
#include <Math/TransformTest.h>
#include <Math/TypeTraitsTest.h>
#include <Math/UtilsTest.h>

#include <Scene/CameraTest.h>
#include <Scene/LightSourceTest.h>
#include <Scene/SceneNodeTest.h>
#include <Scene/SceneRootTest.h>
#include <Scene/TransformTest.h>

// NOTE
// To run a subset of test cases use a filter, e.g '--gtest_filter=*Distribution2D*'.
int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
