// Test Cogwheel Cameras.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_CAMERA_TEST_H_
#define _COGWHEEL_SCENE_CAMERA_TEST_H_

#include <Cogwheel/Math/Utils.h>
#include <Cogwheel/Scene/Camera.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Scene {

class Scene_Camera : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        SceneNodes::allocate(1u);
        SceneRoots::allocate(1u);
        Cameras::allocate(1u);
    }
    virtual void TearDown() {
        Cameras::deallocate();
        SceneRoots::deallocate();
        SceneNodes::deallocate();
    }

    static bool compare_matrix4x4f(Math::Matrix4x4f lhs, Math::Matrix4x4f rhs, unsigned short max_ulps) {
        return almost_equal(lhs, rhs, max_ulps);
    }

    static bool compare_vector3f(Math::Vector3f lhs, Math::Vector3f rhs, unsigned short max_ulps) {
        return almost_equal(lhs, rhs, max_ulps);
    }
};

TEST_F(Scene_Camera, resizing) {
    Cameras::deallocate();

    Cameras::allocate(8u);
    EXPECT_GE(Cameras::capacity(), 8u);

    // Test that capacity can be increased.
    unsigned int largerCapacity = Cameras::capacity() + 4u;
    Cameras::reserve(largerCapacity);
    EXPECT_GE(Cameras::capacity(), largerCapacity);

    // Test that capacity won't be decreased.
    Cameras::reserve(5u);
    EXPECT_GE(Cameras::capacity(), largerCapacity);

    Cameras::deallocate();
    EXPECT_LT(Cameras::capacity(), largerCapacity);
}

TEST_F(Scene_Camera, perspective_matrices) {
    Math::Matrix4x4f perspective_matrix, inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(1, 1000, Math::PI<float>() / 4.0f, 8.0f / 6.0f,
                                                perspective_matrix, inverse_perspective_matrix);

    // Ground truth matrix computed in Unity.
    // Has the third 'column' flipped, to switch from OpenGL representation to DX, +Z forward.
    Math::Matrix4x4f qed = { 1.81066f, 0.00000f, 0.00000f,  0.00000f,
                             0.00000f, 2.41421f, 0.00000f,  0.00000f,
                             0.00000f, 0.00000f, 1.00200f, -2.00200f,
                             0.00000f, 0.00000f, 1.00000f,  0.00000f };

    EXPECT_PRED3(compare_matrix4x4f, perspective_matrix, qed, 25);
    EXPECT_EQ(invert(perspective_matrix), inverse_perspective_matrix);
}

TEST_F(Scene_Camera, sentinel_camera) {

    Cameras::UID sentinel_ID = Cameras::UID::invalid_UID();

    EXPECT_FALSE(Cameras::has(sentinel_ID));

    EXPECT_EQ(Cameras::get_scene_ID(sentinel_ID), SceneRoots::UID::invalid_UID());
    EXPECT_EQ(Cameras::get_viewport(sentinel_ID), Math::Rectf(0.0f, 0.0f, 0.0f, 0.0f));
}

TEST_F(Scene_Camera, create) {

    Math::Matrix4x4f perspective_matrix, inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(1, 1000, Math::PI<float>() / 4.0f, 8.0f / 6.0f,
                                                perspective_matrix, inverse_perspective_matrix);

    SceneRoots::UID scene_ID = SceneRoots::create("Root", Math::RGB::white());
    Cameras::UID cam_ID = Cameras::create("Test cam", scene_ID,
                                          perspective_matrix, inverse_perspective_matrix);
    EXPECT_TRUE(Cameras::has(cam_ID));

    EXPECT_EQ(Cameras::get_name(cam_ID), "Test cam");
    EXPECT_EQ(Cameras::get_z_index(cam_ID), 0u);
    EXPECT_EQ(Cameras::get_projection_matrix(cam_ID), perspective_matrix);
    EXPECT_EQ(Cameras::get_inverse_projection_matrix(cam_ID), inverse_perspective_matrix);
    EXPECT_EQ(Cameras::get_viewport(cam_ID), Math::Rectf(0.0f, 0.0f, 1.0f, 1.0f));
}

TEST_F(Scene_Camera, set_new_matrices) {
    // Create initial projection matrices.
    Math::Matrix4x4f initial_perspective_matrix, initial_inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(1, 1000, Math::PI<float>() / 4.0f, 8.0f / 6.0f,
        initial_perspective_matrix, initial_inverse_perspective_matrix);

    SceneRoots::UID scene_ID = SceneRoots::create("Root", Math::RGB::white());
    Cameras::UID cam_ID = Cameras::create("Test cam", scene_ID,
                                          initial_perspective_matrix, initial_inverse_perspective_matrix);
    EXPECT_TRUE(Cameras::has(cam_ID));

    EXPECT_EQ(Cameras::get_projection_matrix(cam_ID), initial_perspective_matrix);
    EXPECT_EQ(Cameras::get_inverse_projection_matrix(cam_ID), initial_inverse_perspective_matrix);

    // Create and set new projection matrices.
    Math::Matrix4x4f new_perspective_matrix, new_inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(0.3f, 100, Math::PI<float>() / 3.0f, 8.0f / 6.0f,
        new_perspective_matrix, new_inverse_perspective_matrix);

    Cameras::set_projection_matrices(cam_ID, new_perspective_matrix, new_inverse_perspective_matrix);
    EXPECT_EQ(Cameras::get_projection_matrix(cam_ID), new_perspective_matrix);
    EXPECT_EQ(Cameras::get_inverse_projection_matrix(cam_ID), new_inverse_perspective_matrix);
}

TEST_F(Scene_Camera, z_sorting) {
    SceneRoots::UID scene_ID = SceneRoots::create("Root", Math::RGB::white());
    
    Cameras::UID high_z_cam_ID = Cameras::create("high z cam", scene_ID, Math::Matrix4x4f::identity(), Math::Matrix4x4f::identity());
    Cameras::set_z_index(high_z_cam_ID, 10);

    Cameras::UID low_z_cam_ID = Cameras::create("Low z cam", scene_ID, Math::Matrix4x4f::identity(), Math::Matrix4x4f::identity());
    Cameras::set_z_index(low_z_cam_ID, 0);

    auto sorted_IDs = Cameras::get_z_sorted_IDs();
    EXPECT_EQ(low_z_cam_ID, sorted_IDs[0]);
    EXPECT_EQ(high_z_cam_ID, sorted_IDs[1]);

    Cameras::UID medium_z_cam_ID = Cameras::create("Medium z cam", scene_ID, Math::Matrix4x4f::identity(), Math::Matrix4x4f::identity());
    Cameras::set_z_index(medium_z_cam_ID, 5);

    sorted_IDs = Cameras::get_z_sorted_IDs();
    EXPECT_EQ(low_z_cam_ID, sorted_IDs[0]);
    EXPECT_EQ(medium_z_cam_ID, sorted_IDs[1]);
    EXPECT_EQ(high_z_cam_ID, sorted_IDs[2]);
}

TEST_F(Scene_Camera, ray_projection) {
    using namespace Cogwheel::Math;

    // Create initial projection matrices.
    Matrix4x4f initial_perspective_matrix, initial_inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(1, 1000, PI<float>() / 4.0f, 8.0f / 6.0f,
        initial_perspective_matrix, initial_inverse_perspective_matrix);

    SceneRoots::UID scene_ID = SceneRoots::create("Root", RGB::white());
    Cameras::UID cam_ID = Cameras::create("Test cam", scene_ID,
                                          initial_perspective_matrix, initial_inverse_perspective_matrix);
    EXPECT_TRUE(Cameras::has(cam_ID));

    const float maximally_allowed_cos_angle = cos(degrees_to_radians(0.5f));

    { // Forward should be +Z when the transform is identity.
        Ray ray = CameraUtils::ray_from_viewport_point(cam_ID, Vector2f(0.5f, 0.5f));
        EXPECT_EQ(ray.direction, Vector3f::forward());

        // Unity QED ray: Origin : (-0.55228, -0.41421, 1.00000), Dir : (-0.45450, -0.34087, 0.82294)
        Ray ray_periferi = CameraUtils::ray_from_viewport_point(cam_ID, Vector2f(0.0f, 0.0f));
        float cos_angle_between_rays = dot(ray_periferi.direction, Vector3f(-0.45450f, -0.34087f, 0.82294f));
        EXPECT_GT(cos_angle_between_rays, maximally_allowed_cos_angle);
    }

    { // Translation shouldn't change forward.
        Transform cam_transform = Transform::identity();
        cam_transform.translation = Vector3f(100, 10, -30);
        Cameras::set_transform(cam_ID, cam_transform);

        {
            Ray ray = CameraUtils::ray_from_viewport_point(cam_ID, Vector2f(0.5f, 0.5f));
            float cos_angle_between_rays = dot(ray.direction, Vector3f::forward());
            EXPECT_GT(cos_angle_between_rays, maximally_allowed_cos_angle);
        }

        { // Unity QED ray: Origin : (99.44772, 9.58579, -29.00000), Dir : (-0.45450, -0.34087, 0.82294)
            Ray ray_periferi = CameraUtils::ray_from_viewport_point(cam_ID, Vector2f(0.0f, 0.0f));
            float cos_angle_between_rays = dot(ray_periferi.direction, Vector3f(-0.45450f, -0.34087f, 0.82294f));
            EXPECT_GT(cos_angle_between_rays, maximally_allowed_cos_angle);
        }
    }

    { // The rays direction and the transform applied to the forward direction should be similar after rotation.
        Transform cam_transform = Transform::identity();
        cam_transform.rotation = Quaternionf::from_angle_axis(degrees_to_radians(30.0f), normalize(Vector3f(1, 2, 3)));
        Cameras::set_transform(cam_ID, cam_transform);

        {
            Ray ray = CameraUtils::ray_from_viewport_point(cam_ID, Vector2f(0.5f, 0.5f));
            float cos_angle_between_rays = dot(ray.direction, cam_transform * Vector3f::forward());
            EXPECT_GT(cos_angle_between_rays, maximally_allowed_cos_angle);
        }

        { // Unity QED ray: Origin : (-0.02948, -0.68276, 1.00477), Dir: (-0.02426, -0.56188, 0.82687)
            Ray ray_periferi = CameraUtils::ray_from_viewport_point(cam_ID, Vector2f(0.0f, 0.0f));
            float cos_angle_between_rays = dot(ray_periferi.direction, Vector3f(-0.02426f, -0.56188f, 0.82687f));
            EXPECT_GT(cos_angle_between_rays, maximally_allowed_cos_angle);
        }
    }

    { // Rotation and translation.
        Transform cam_transform = Transform::identity();
        cam_transform.translation = Vector3f(100, 10, -30);
        cam_transform.rotation = Quaternionf::from_angle_axis(degrees_to_radians(30.0f), normalize(Vector3f(1, 2, 3)));
        Cameras::set_transform(cam_ID, cam_transform);

        {
            Ray ray = CameraUtils::ray_from_viewport_point(cam_ID, Vector2f(0.5f, 0.5f));
            float cos_angle_between_rays = dot(ray.direction, cam_transform * Vector3f::forward());
            EXPECT_GT(cos_angle_between_rays, maximally_allowed_cos_angle);
        }

        { // Unity QED ray: Origin : (-0.02948, -0.68276, 1.00477), Dir: (-0.02426, -0.56188, 0.82687)
            Ray ray_periferi = CameraUtils::ray_from_viewport_point(cam_ID, Vector2f(0.0f, 0.0f));
            float cos_angle_between_rays = dot(ray_periferi.direction, Vector3f(-0.02426f, -0.56188f, 0.82687f));
            EXPECT_GT(cos_angle_between_rays, maximally_allowed_cos_angle);
        }
    }
}

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_CAMERA_TEST_H_