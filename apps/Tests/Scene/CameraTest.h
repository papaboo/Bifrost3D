// Test Cogwheel Cameras.
// ---------------------------------------------------------------------------
// Copyright (C) 2015, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _COGWHEEL_SCENE_CAMERA_TEST_H_
#define _COGWHEEL_SCENE_CAMERA_TEST_H_

#include <Scene/Camera.h>

#include <gtest/gtest.h>

namespace Cogwheel {
namespace Scene {

class Scene_Camera : public ::testing::Test {
protected:
    // Per-test set-up and tear-down logic.
    virtual void SetUp() {
        SceneNodes::allocate(8u);
    }
    virtual void TearDown() {
        SceneNodes::deallocate();
    }

    static bool compare_matrix4x4f(Math::Matrix4x4f lhs, Math::Matrix4x4f rhs, unsigned short maxUlps) {
        return almostEqual(lhs, rhs, maxUlps);
    }
};

TEST_F(Scene_Camera, resizing) {
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
    Math::Matrix4x4f qed = { 1.81066f, 0.0000f, 0.00000f, 0.00000f,
                             0.00000f, 2.41421f, 0.00000f, 0.00000f,
                             0.00000f, 0.00000f, -1.00200f, -1.00000f,
                             0.00000f, 0.00000f, -2.00200f, 0.00000f };

    EXPECT_PRED3(compare_matrix4x4f, perspective_matrix, qed, 25);
    EXPECT_EQ(invert(perspective_matrix), inverse_perspective_matrix);
}

TEST_F(Scene_Camera, sentinel_camera) {
    Cameras::allocate(2u);
    
    Cameras::UID sentinel_id = Cameras::UID::InvalidUID();

    EXPECT_FALSE(Cameras::has(sentinel_id));

    EXPECT_EQ(Cameras::get_parent_ID(sentinel_id), SceneNodes::UID::InvalidUID());
    EXPECT_EQ(Cameras::get_viewport(sentinel_id), Math::Rectf(0.0f, 0.0f, 0.0f, 0.0f));

    Cameras::deallocate();
}

TEST_F(Scene_Camera, creating) {
    Cameras::allocate(2u);

    SceneNodes::UID camNode = SceneNodes::create("Cam");

    Math::Matrix4x4f perspective_matrix, inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(1, 1000, Math::PI<float>() / 4.0f, 8.0f / 6.0f,
                                                perspective_matrix, inverse_perspective_matrix);

    Cameras::allocate(2u);
    Cameras::UID cam_id = Cameras::create(camNode, perspective_matrix, inverse_perspective_matrix);
    EXPECT_TRUE(Cameras::has(cam_id));
    
    EXPECT_EQ(Cameras::get_parent_ID(cam_id), camNode);
    EXPECT_EQ(Cameras::get_render_index(cam_id), 0u);
    EXPECT_EQ(Cameras::get_projection_matrix(cam_id), perspective_matrix);
    EXPECT_EQ(Cameras::get_inverse_projection_matrix(cam_id), inverse_perspective_matrix);
    EXPECT_EQ(Cameras::get_viewport(cam_id), Math::Rectf(0.0f, 0.0f, 1.0f, 1.0f));

    Cameras::deallocate();
}

TEST_F(Scene_Camera, set_new_matrices) {
    Cameras::allocate(2u);

    // Create initial camera and projection matrices.
    SceneNodes::UID camNode = SceneNodes::create("Cam");

    Math::Matrix4x4f initial_perspective_matrix, initial_inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(1, 1000, Math::PI<float>() / 4.0f, 8.0f / 6.0f,
        initial_perspective_matrix, initial_inverse_perspective_matrix);

    Cameras::UID cam_id = Cameras::create(camNode, initial_perspective_matrix, initial_inverse_perspective_matrix);
    EXPECT_TRUE(Cameras::has(cam_id));

    EXPECT_EQ(Cameras::get_projection_matrix(cam_id), initial_perspective_matrix);
    EXPECT_EQ(Cameras::get_inverse_projection_matrix(cam_id), initial_inverse_perspective_matrix);

    // Create and set new projection matrices.
    Math::Matrix4x4f new_perspective_matrix, new_inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(0.3f, 100, Math::PI<float>() / 3.0f, 8.0f / 6.0f,
        new_perspective_matrix, new_inverse_perspective_matrix);

    Cameras::set_projection_matrices(cam_id, new_perspective_matrix, new_inverse_perspective_matrix);
    EXPECT_EQ(Cameras::get_projection_matrix(cam_id), new_perspective_matrix);
    EXPECT_EQ(Cameras::get_inverse_projection_matrix(cam_id), new_inverse_perspective_matrix);

    Cameras::deallocate();
}

} // NS Scene
} // NS Cogwheel

#endif // _COGWHEEL_SCENE_CAMERA_TEST_H_