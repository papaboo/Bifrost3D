#include <gtest/gtest.h>

#include <Core/ArrayTest.h>
#include <Core/UniqueIDGeneratorTest.h>

#include <Input/KeyboardTest.h>

#include <Math/MatrixTest.h>
#include <Math/QuaternionTest.h>
#include <Math/TransformTest.h>

#include <Scene/SceneNodeTest.h>
#include <Scene/TransformTest.h>

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
