// Test Cogwheel Scene Node Transforms.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_TRANSFORM_TEST_H_
#define _COGWHEEL_SCENE_TRANSFORM_TEST_H_

#include <Scene/SceneNode.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Scene {

using namespace Math;

class Scene_TransformTest : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        SceneNodes::allocate(8u);
    }
    virtual void TearDown() {
        SceneNodes::deallocate();
    }

    static bool compareVectors(Vector3f lhs, Vector3f rhs) {
        return almostEqual(lhs, rhs, 10);
    }

    static bool compareQuaternions(Quaternionf lhs, Quaternionf rhs) {
        return almostEqual(lhs, rhs);
    }

    static bool compareTransforms(Transform lhs, Transform rhs) {
        return almostEqual(lhs.mTranslation, rhs.mTranslation)
            && almostEqual(lhs.mRotation, rhs.mRotation)
            && almostEqual(lhs.mScale, rhs.mScale);
    }
};

TEST_F(Scene_TransformTest, IdentityAsDefault) {
    SceneNode foo = SceneNodes::create("Foo");
    const Transform globalTrans = foo.getGlobalTransform();
    const Transform localTrans = foo.getLocalTransform();

    const Transform identity = Transform::identity();

    EXPECT_EQ(localTrans, identity);
    EXPECT_EQ(globalTrans, identity);
}

TEST_F(Scene_TransformTest, SetTransform) {
    SceneNode foo = SceneNodes::create("Foo");
    const Transform newTrans = Transform(Vector3f(3, 5, 9), normalize(Quaternionf(1, 2, 3, 4)), 3);
    foo.setGlobalTransform(newTrans);

    EXPECT_EQ(foo.getGlobalTransform(), newTrans);
}

TEST_F(Scene_TransformTest, HierarchicalTranslation) {
    SceneNode foo = SceneNodes::create("Foo");
    SceneNode bar = SceneNodes::create("Bar");
    bar.setParent(foo);

    const Vector3f translation = Vector3f(3, 5, 9);
    const Transform newTrans = Transform(translation);
    foo.setGlobalTransform(newTrans);
    bar.setGlobalTransform(Transform::identity());

    EXPECT_EQ(foo.getGlobalTransform().mTranslation, translation);
    EXPECT_EQ(bar.getLocalTransform().mTranslation, -translation);
    EXPECT_EQ(bar.getGlobalTransform().mTranslation, Vector3f::zero());
}

TEST_F(Scene_TransformTest, HierarchicalRotation) {
    SceneNode foo = SceneNodes::create("Foo");
    SceneNode bar = SceneNodes::create("Bar");
    bar.setParent(foo);

    const Quaternionf rotation = normalize(Quaternionf(1, 2, 3, 4));
    const Transform newTrans = Transform(Vector3f::zero(), rotation);
    foo.setGlobalTransform(newTrans);
    bar.setGlobalTransform(Transform::identity());

    EXPECT_EQ(foo.getGlobalTransform().mRotation, rotation);
    EXPECT_EQ(bar.getLocalTransform().mRotation, conjugate(rotation));
    EXPECT_EQ(bar.getGlobalTransform().mRotation, Quaternionf::identity());
}

TEST_F(Scene_TransformTest, LocalTransform) {
    SceneNode n0 = SceneNodes::create("n0");
    SceneNode n1 = SceneNodes::create("n1");
    n0.setGlobalTransform(Transform(Vector3f(1, 2, 3), Quaternionf::fromAngleAxis(45.0f, Vector3f(0, 1, 0))));
    n1.setGlobalTransform(Transform(Vector3f(1, 2, 3), Quaternionf::fromAngleAxis(-45.0f, Vector3f(0, 1, 0))));
    n1.setParent(n0);

    // Test position of transformed child node.
    EXPECT_EQ(n1.getLocalTransform().mTranslation, Vector3f::zero());

    // Test rotation of transformed child node.
    Quaternionf expectedLocalRot = Quaternionf::fromAngleAxis(-90.0f, Vector3f(0, 1, 0));
    EXPECT_PRED2(compareQuaternions, n1.getLocalTransform().mRotation, expectedLocalRot);

    // Verify that applying n0's global transform to n1's local transform yields n1's global transform.
    // TBH I'm amazed that this works with EXPECT_EQ.
    Transform computedN1Global = n0.getGlobalTransform() * n1.getLocalTransform();
    EXPECT_EQ(computedN1Global, n1.getGlobalTransform());
}

TEST_F(Scene_TransformTest, PreserveLocalTransformOnParentTransformation) {
    SceneNode n0 = SceneNodes::create("n0");
    SceneNode n1 = SceneNodes::create("n1");

    Transform n1LocalTrans = Transform(Vector3f(1, 2, 3), Quaternionf::fromAngleAxis(-45.0f, Vector3f::up()));
    n1.setGlobalTransform(n1LocalTrans);

    n1.setParent(n0);

    n0.setGlobalTransform(Transform(Vector3f(4, 2, 0), Quaternionf::fromAngleAxis(30.0f, Vector3f::forward())));

    EXPECT_PRED2(compareTransforms, n1LocalTrans, n1.getLocalTransform());
}

TEST_F(Scene_TransformTest, ComplexHierachy) {
    // Tests the following hierachy
    //    t0
    //   / | \
    // t1 t2  t5
    //    / \   \
    //   t3 t4   t6
    SceneNode n0 = SceneNodes::create("n0");
    SceneNode n1 = SceneNodes::create("n1");
    SceneNode n2 = SceneNodes::create("n2");
    SceneNode n3 = SceneNodes::create("n3");
    SceneNode n4 = SceneNodes::create("n4");
    SceneNode n5 = SceneNodes::create("n5");
    SceneNode n6 = SceneNodes::create("n6");

    n1.setParent(n0);
    n2.setParent(n0);
    n3.setParent(n2);
    n4.setParent(n2);
    n5.setParent(n0);
    n6.setParent(n5);

    n0.setGlobalTransform(Transform(Vector3f(3, 2, 1), Quaternionf::identity()));
    n1.setGlobalTransform(Transform::identity());

    n2.setGlobalTransform(Transform(Vector3f(1, 2, 3), Quaternionf::fromAngleAxis(45.0f, Vector3f(0, 1, 0))));
    n3.setGlobalTransform(Transform::identity());
    n4.setGlobalTransform(Transform(Vector3f(1, 2, 3), Quaternionf::fromAngleAxis(-45.0f, Vector3f(0, 1, 0))));

    n5.setGlobalTransform(Transform(Vector3f(4, 4, 4), Quaternionf::identity(), 0.5f));
    n6.setGlobalTransform(Transform(Vector3f(2, 2, 2), Quaternionf::fromAngleAxis(-45.0f, Vector3f(1, 0, 0)), 0.5f));

    // n0
    //     global: position: (3, 2, 1), rotation: (1, [0, 0, 0]), scale: 1
    //     local: position: (3, 2, 1), rotation: (1, [0, 0, 0]), scale: 1
    Transform n0Transform = Transform(Vector3f(3, 2, 1), Quaternionf::identity(), 1.0f);
    EXPECT_PRED2(compareTransforms, n0.getGlobalTransform(), n0Transform);
    EXPECT_PRED2(compareTransforms, n0.getLocalTransform(), n0Transform);

    // n1
    //     global: position: (0, 0, 0), rotation: (1, [0, 0, 0]), scale: 1
    //     local: position: (-3, -2, -1), rotation: (1, [0, 0, 0]), scale: 1
    Transform n1Global = Transform(Vector3f(0, 0, 0), Quaternionf::identity(), 1.0f);
    Transform n1Local = Transform(Vector3f(-3, -2, -1), Quaternionf::identity(), 1.0f);
    EXPECT_PRED2(compareTransforms, n1.getGlobalTransform(), n1Global);
    EXPECT_PRED2(compareTransforms, n1.getLocalTransform(), n1Local);

    // n2
    //     global: position: (1, 2, 3), rotation: (0.92388, [0, 0.382683, 0]), scale: 1
    //     local: position: (-2, 0, 2), rotation: (0.92388, [0, 0.382683, 0]), scale: 1
    Transform n2Global = Transform(Vector3f(1, 2, 3), Quaternionf(0.0f, 0.382683456f, 0.0f, 0.923879504f), 1.0f);
    Transform n2Local = Transform(Vector3f(-2, 0, 2), Quaternionf(0.0f, 0.382683456f, 0.0f, 0.923879504f), 1.0f);
    EXPECT_PRED2(compareTransforms, n2.getGlobalTransform(), n2Global);
    EXPECT_PRED2(compareTransforms, n2.getLocalTransform(), n2Local);

    // n3
    //     global: position: (0, 0, 0), rotation: (1, [0, 0, 0]), scale: 1
    //     local: position: (1.41421, -2, -2.82843), rotation: (0.92388, [0, -0.38268, 0]), scale: 1
    Transform n3Global = Transform::identity();
    Transform n3Local = Transform(Vector3f(1.41421342f, -2.0f, -2.82842708f), Quaternionf(0.0f, -0.382683456f, 0.0f, 0.923879504f), 1.0f);
    EXPECT_PRED2(compareTransforms, n3.getGlobalTransform(), n3Global);
    EXPECT_PRED2(compareTransforms, n3.getLocalTransform(), n3Local);

    // n4
    //     global: position: (1, 2, 3), rotation: (0.92388, [0, -0.38268, 0]), scale: 0.9999999
    //     local: position: (0, 0, 0), rotation: (0.70711, [0, -0.70711, 0]), scale: 1
    Transform n4Global = Transform(Vector3f(1, 2, 3), Quaternionf(0.0f, -0.382683456f, 0.0f, 0.923879504f), 1.0f);
    Transform n4Local = Transform(Vector3f::zero(), Quaternionf(0.0f, -0.707106829f, 0.0f, 0.707106709f), 1.0f);
    EXPECT_PRED2(compareTransforms, n4.getGlobalTransform(), n4Global);
    EXPECT_PRED2(compareTransforms, n4.getLocalTransform(), n4Local);

    // n5
    //     global: position: (4, 4, 4), rotation: (1, [0, 0, 0]), scale: 0.5
    //     local: position: (1, 2, 3), rotation: (1, [0, 0, 0]), scale: 0.5
    Transform n5Global = Transform(Vector3f(4, 4, 4), Quaternionf::identity(), 0.5f);
    Transform n5Local = Transform(Vector3f(1, 2, 3), Quaternionf::identity(), 0.5f);
    EXPECT_PRED2(compareTransforms, n5.getGlobalTransform(), n5Global);
    EXPECT_PRED2(compareTransforms, n5.getLocalTransform(), n5Local);

    // n6
    //     global: position: (2, 2, 2), rotation: (0.92388, [-0.38268, 0, 0]), scale: 0.5
    //     local: position: (-4, -4, -4), rotation: (0.92388, [-0.38268, 0, 0]), scale: 1
    Transform n6Global = Transform(Vector3f(2, 2, 2), Quaternionf(-0.382683456f, 0.0f, 0.0f, 0.923879504f), 0.5f);
    Transform n6Local = Transform(Vector3f(-4, -4, -4), Quaternionf(-0.382683456f, 0.0f, 0.0f, 0.923879504f), 1.0f);
    EXPECT_PRED2(compareTransforms, n6.getGlobalTransform(), n6Global);
    EXPECT_PRED2(compareTransforms, n6.getLocalTransform(), n6Local);
}

TEST_F(Scene_TransformTest, LookAt) {
    SceneNode n0 = SceneNodes::create("n0");
    SceneNode n1 = SceneNodes::create("n1");
    n0.setGlobalTransform(Transform::identity());
    n1.setGlobalTransform(Transform(Vector3f(3, 5, 9)));

    Transform t1 = n1.getGlobalTransform();
    t1.lookAt(n0.getGlobalTransform().mTranslation);
    n1.setGlobalTransform(t1);

    Transform t0 = n0.getGlobalTransform();
    t0.lookAt(n1.getGlobalTransform().mTranslation);
    n0.setGlobalTransform(t0);

    { // Test that both transforms are 'facing' each other.
        Vector3f n0ToN1Direction = normalize(n1.getGlobalTransform().mTranslation - n0.getGlobalTransform().mTranslation);

        Vector3f n0Forward = n0.getGlobalTransform().mRotation * Vector3f::forward();
        EXPECT_PRED2(compareVectors, n0Forward, n0ToN1Direction);

        Vector3f n1Forward = n1.getGlobalTransform().mRotation * Vector3f::forward();
        EXPECT_PRED2(compareVectors, n1Forward, -n0ToN1Direction);
    }
}

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_TRANSFORM_TEST_H_