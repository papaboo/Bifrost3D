// Test Cogwheel Scene Node Transforms.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_TRANSFORM_TEST_H_
#define _COGWHEEL_SCENE_TRANSFORM_TEST_H_

#include <Cogwheel/Scene/SceneNode.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Scene {

using namespace Math;

class Scene_Transform : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        SceneNodes::allocate(8u);
    }
    virtual void TearDown() {
        SceneNodes::deallocate();
    }

    static bool compare_vector(Vector3f lhs, Vector3f rhs) {
        return almost_equal(lhs, rhs, 10);
    }

    static bool compare_quaternions(Quaternionf lhs, Quaternionf rhs) {
        return almost_equal(lhs, rhs);
    }

    static bool compare_transforms(Transform lhs, Transform rhs) {
        return almost_equal(lhs.translation, rhs.translation)
            && almost_equal(lhs.rotation, rhs.rotation)
            && almost_equal(lhs.scale, rhs.scale);
    }
};

TEST_F(Scene_Transform, identity_as_default) {
    SceneNode foo = SceneNodes::create("Foo");
    const Transform global_trans = foo.get_global_transform();
    const Transform local_trans = foo.get_local_transform();

    const Transform identity = Transform::identity();

    EXPECT_EQ(local_trans, identity);
    EXPECT_EQ(global_trans, identity);
}

TEST_F(Scene_Transform, set_transform) {
    SceneNode foo = SceneNodes::create("Foo");
    const Transform new_trans = Transform(Vector3f(3, 5, 9), normalize(Quaternionf(1, 2, 3, 4)), 3);
    foo.set_global_transform(new_trans);

    EXPECT_EQ(foo.get_global_transform(), new_trans);
}

TEST_F(Scene_Transform, hierarchical_translation) {
    SceneNode foo = SceneNodes::create("Foo");
    SceneNode bar = SceneNodes::create("Bar");
    bar.set_parent(foo);

    const Vector3f translation = Vector3f(3, 5, 9);
    const Transform new_trans = Transform(translation);
    foo.set_global_transform(new_trans);
    bar.set_global_transform(Transform::identity());

    EXPECT_EQ(foo.get_global_transform().translation, translation);
    EXPECT_EQ(bar.get_local_transform().translation, -translation);
    EXPECT_EQ(bar.get_global_transform().translation, Vector3f::zero());
}

TEST_F(Scene_Transform, hierarchical_rotation) {
    SceneNode foo = SceneNodes::create("Foo");
    SceneNode bar = SceneNodes::create("Bar");
    bar.set_parent(foo);

    const Quaternionf rotation = normalize(Quaternionf(1, 2, 3, 4));
    const Transform new_trans = Transform(Vector3f::zero(), rotation);
    foo.set_global_transform(new_trans);
    bar.set_global_transform(Transform::identity());

    EXPECT_EQ(foo.get_global_transform().rotation, rotation);
    EXPECT_EQ(bar.get_local_transform().rotation, conjugate(rotation));
    EXPECT_EQ(bar.get_global_transform().rotation, Quaternionf::identity());
}

TEST_F(Scene_Transform, local_transform) {
    SceneNode n0 = SceneNodes::create("n0");
    SceneNode n1 = SceneNodes::create("n1");
    n0.set_global_transform(Transform(Vector3f(1, 2, 3), Quaternionf::from_angle_axis(degrees_to_radians(45.0f), Vector3f(0, 1, 0))));
    n1.set_global_transform(Transform(Vector3f(1, 2, 3), Quaternionf::from_angle_axis(degrees_to_radians(-45.0f), Vector3f(0, 1, 0))));
    n1.set_parent(n0);

    // Test position of transformed child node.
    EXPECT_EQ(n1.get_local_transform().translation, Vector3f::zero());

    // Test rotation of transformed child node.
    Quaternionf expected_local_rot = Quaternionf::from_angle_axis(degrees_to_radians(-90.0f), Vector3f(0, 1, 0));
    EXPECT_PRED2(compare_quaternions, n1.get_local_transform().rotation, expected_local_rot);

    // Verify that applying n0's global transform to n1's local transform yields n1's global transform.
    Transform computedN1Global = n0.get_global_transform() * n1.get_local_transform();
    EXPECT_PRED2(compare_transforms, computedN1Global, n1.get_global_transform());
}

TEST_F(Scene_Transform, preserve_local_transform_on_parent_transformation) {
    SceneNode n0 = SceneNodes::create("n0");
    SceneNode n1 = SceneNodes::create("n1");

    Transform n1_local_trans = Transform(Vector3f(1, 2, 3), Quaternionf::from_angle_axis(degrees_to_radians(-45.0f), Vector3f::up()));
    n1.set_global_transform(n1_local_trans);

    n1.set_parent(n0);

    n0.set_global_transform(Transform(Vector3f(4, 2, 0), Quaternionf::from_angle_axis(degrees_to_radians(30.0f), Vector3f::forward())));

    EXPECT_PRED2(compare_transforms, n1_local_trans, n1.get_local_transform());
}

TEST_F(Scene_Transform, complex_hierachy) {
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

    n1.set_parent(n0);
    n2.set_parent(n0);
    n3.set_parent(n2);
    n4.set_parent(n2);
    n5.set_parent(n0);
    n6.set_parent(n5);

    n0.set_global_transform(Transform(Vector3f(3, 2, 1), Quaternionf::identity()));
    n1.set_global_transform(Transform::identity());

    n2.set_global_transform(Transform(Vector3f(1, 2, 3), Quaternionf::from_angle_axis(degrees_to_radians(45.0f), Vector3f(0, 1, 0))));
    n3.set_global_transform(Transform::identity());
    n4.set_global_transform(Transform(Vector3f(1, 2, 3), Quaternionf::from_angle_axis(degrees_to_radians(-45.0f), Vector3f(0, 1, 0))));

    n5.set_global_transform(Transform(Vector3f(4, 4, 4), Quaternionf::identity(), 0.5f));
    n6.set_global_transform(Transform(Vector3f(2, 2, 2), Quaternionf::from_angle_axis(degrees_to_radians(-45.0f), Vector3f(1, 0, 0)), 0.5f));

    // n0
    //     global: position: (3, 2, 1), rotation: (1, [0, 0, 0]), scale: 1
    //     local: position: (3, 2, 1), rotation: (1, [0, 0, 0]), scale: 1
    Transform n0_transform = Transform(Vector3f(3, 2, 1), Quaternionf::identity(), 1.0f);
    EXPECT_PRED2(compare_transforms, n0.get_global_transform(), n0_transform);
    EXPECT_PRED2(compare_transforms, n0.get_local_transform(), n0_transform);

    // n1
    //     global: position: (0, 0, 0), rotation: (1, [0, 0, 0]), scale: 1
    //     local: position: (-3, -2, -1), rotation: (1, [0, 0, 0]), scale: 1
    Transform n1_global = Transform(Vector3f(0, 0, 0), Quaternionf::identity(), 1.0f);
    Transform n1_local = Transform(Vector3f(-3, -2, -1), Quaternionf::identity(), 1.0f);
    EXPECT_PRED2(compare_transforms, n1.get_global_transform(), n1_global);
    EXPECT_PRED2(compare_transforms, n1.get_local_transform(), n1_local);

    // n2
    //     global: position: (1, 2, 3), rotation: (0.92388, [0, 0.382683, 0]), scale: 1
    //     local: position: (-2, 0, 2), rotation: (0.92388, [0, 0.382683, 0]), scale: 1
    Transform n2_global = Transform(Vector3f(1, 2, 3), Quaternionf(0.0f, 0.382683456f, 0.0f, 0.923879504f), 1.0f);
    Transform n2_local = Transform(Vector3f(-2, 0, 2), Quaternionf(0.0f, 0.382683456f, 0.0f, 0.923879504f), 1.0f);
    EXPECT_PRED2(compare_transforms, n2.get_global_transform(), n2_global);
    EXPECT_PRED2(compare_transforms, n2.get_local_transform(), n2_local);

    // n3
    //     global: position: (0, 0, 0), rotation: (1, [0, 0, 0]), scale: 1
    //     local: position: (1.41421, -2, -2.82843), rotation: (0.92388, [0, -0.38268, 0]), scale: 1
    Transform n3_global = Transform::identity();
    Transform n3_local = Transform(Vector3f(1.41421342f, -2.0f, -2.82842708f), Quaternionf(0.0f, -0.382683456f, 0.0f, 0.923879504f), 1.0f);
    EXPECT_PRED2(compare_transforms, n3.get_global_transform(), n3_global);
    EXPECT_PRED2(compare_transforms, n3.get_local_transform(), n3_local);

    // n4
    //     global: position: (1, 2, 3), rotation: (0.92388, [0, -0.38268, 0]), scale: 0.9999999
    //     local: position: (0, 0, 0), rotation: (0.70711, [0, -0.70711, 0]), scale: 1
    Transform n4_global = Transform(Vector3f(1, 2, 3), Quaternionf(0.0f, -0.382683456f, 0.0f, 0.923879504f), 1.0f);
    Transform n4_local = Transform(Vector3f::zero(), Quaternionf(0.0f, -0.707106829f, 0.0f, 0.707106709f), 1.0f);
    EXPECT_PRED2(compare_transforms, n4.get_global_transform(), n4_global);
    EXPECT_PRED2(compare_transforms, n4.get_local_transform(), n4_local);

    // n5
    //     global: position: (4, 4, 4), rotation: (1, [0, 0, 0]), scale: 0.5
    //     local: position: (1, 2, 3), rotation: (1, [0, 0, 0]), scale: 0.5
    Transform n5_global = Transform(Vector3f(4, 4, 4), Quaternionf::identity(), 0.5f);
    Transform n5_local = Transform(Vector3f(1, 2, 3), Quaternionf::identity(), 0.5f);
    EXPECT_PRED2(compare_transforms, n5.get_global_transform(), n5_global);
    EXPECT_PRED2(compare_transforms, n5.get_local_transform(), n5_local);

    // n6
    //     global: position: (2, 2, 2), rotation: (0.92388, [-0.38268, 0, 0]), scale: 0.5
    //     local: position: (-4, -4, -4), rotation: (0.92388, [-0.38268, 0, 0]), scale: 1
    Transform n6_global = Transform(Vector3f(2, 2, 2), Quaternionf(-0.382683456f, 0.0f, 0.0f, 0.923879504f), 0.5f);
    Transform n6_local = Transform(Vector3f(-4, -4, -4), Quaternionf(-0.382683456f, 0.0f, 0.0f, 0.923879504f), 1.0f);
    EXPECT_PRED2(compare_transforms, n6.get_global_transform(), n6_global);
    EXPECT_PRED2(compare_transforms, n6.get_local_transform(), n6_local);
}

TEST_F(Scene_Transform, look_at) {
    SceneNode n0 = SceneNodes::create("n0");
    SceneNode n1 = SceneNodes::create("n1");
    n0.set_global_transform(Transform::identity());
    n1.set_global_transform(Transform(Vector3f(3, 5, 9)));

    Transform t1 = n1.get_global_transform();
    t1.look_at(n0.get_global_transform().translation);
    n1.set_global_transform(t1);

    Transform t0 = n0.get_global_transform();
    t0.look_at(n1.get_global_transform().translation);
    n0.set_global_transform(t0);

    { // Test that both transforms are 'facing' each other.
        Vector3f n0_to_n1_direction = normalize(n1.get_global_transform().translation - n0.get_global_transform().translation);

        Vector3f n0_forward = n0.get_global_transform().rotation * Vector3f::forward();
        EXPECT_PRED2(compare_vector, n0_forward, n0_to_n1_direction);

        Vector3f n1_forward = n1.get_global_transform().rotation * Vector3f::forward();
        EXPECT_PRED2(compare_vector, n1_forward, -n0_to_n1_direction);
    }
}

} // NS Core
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_TRANSFORM_TEST_H_