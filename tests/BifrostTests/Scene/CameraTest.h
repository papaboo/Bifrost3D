// Test Bifrost Cameras.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _BIFROST_SCENE_CAMERA_TEST_H_
#define _BIFROST_SCENE_CAMERA_TEST_H_

#include <Bifrost/Math/Utils.h>
#include <Bifrost/Scene/Camera.h>

#include <gtest/gtest.h>

namespace Bifrost {
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

    // Test camera created notification.
    Core::Iterable<Cameras::ChangedIterator> camera_changes = Cameras::get_changed_cameras();
    EXPECT_EQ(1, camera_changes.end() - camera_changes.begin());
    EXPECT_EQ(cam_ID, *camera_changes.begin());
    EXPECT_EQ(Cameras::Change::Created, Cameras::get_changes(cam_ID));
}

TEST_F(Scene_Camera, create_and_destroy_notifications) {
    SceneRoots::UID scene_ID = SceneRoots::create("Root", Math::RGB::white());
    Math::Matrix4x4f perspective_matrix = Math::Matrix4x4f::identity(), inverse_perspective_matrix = Math::Matrix4x4f::identity();

    Cameras::UID cam_ID0 = Cameras::create("Cam0", scene_ID, perspective_matrix, inverse_perspective_matrix);
    Cameras::UID cam_ID1 = Cameras::create("Cam1", scene_ID, perspective_matrix, inverse_perspective_matrix);
    EXPECT_TRUE(Cameras::has(cam_ID0));
    EXPECT_TRUE(Cameras::has(cam_ID1));

    { // Test camera create notifications.
        Core::Iterable<Cameras::ChangedIterator> changed_cameras = Cameras::get_changed_cameras();
        EXPECT_EQ(changed_cameras.end() - changed_cameras.begin(), 2);

        bool cam0_created = false;
        bool cam1_created = false;
        for (const Cameras::UID scene_ID : changed_cameras) {
            bool scene_created = Cameras::get_changes(scene_ID) == Cameras::Change::Created;
            cam0_created |= scene_ID == cam_ID0 && scene_created;
            cam1_created |= scene_ID == cam_ID1 && scene_created;
        }

        EXPECT_TRUE(cam0_created);
        EXPECT_TRUE(cam1_created);
    }

    Cameras::reset_change_notifications();

    { // Test destroy.
        Cameras::destroy(cam_ID0);
        EXPECT_FALSE(Cameras::has(cam_ID0));

        Core::Iterable<Cameras::ChangedIterator> changed_cameras = Cameras::get_changed_cameras();
        EXPECT_EQ(changed_cameras.end() - changed_cameras.begin(), 1);

        Cameras::UID changed_cam_ID = *changed_cameras.begin();
        bool cam0_destroyed = changed_cam_ID == cam_ID0 && Cameras::get_changes(changed_cam_ID) == Cameras::Change::Destroyed;
        EXPECT_TRUE(cam0_destroyed);
    }

    Cameras::reset_change_notifications();

    { // Test that destroyed camera cannot be destroyed again.
        EXPECT_FALSE(Cameras::has(cam_ID0));

        Cameras::destroy(cam_ID0);
        EXPECT_FALSE(Cameras::has(cam_ID0));
        EXPECT_TRUE(Cameras::get_changed_cameras().is_empty());
    }
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

TEST_F(Scene_Camera, screenshots) {
    using namespace Bifrost::Assets;
    using namespace Bifrost::Math;

    // Stub method can fill a screenshot with an image with 2 iterations.
    auto screenshot_filler = [](Cameras::RequestedContent content_requested, unsigned int minimum_iteration_count) -> std::vector<Screenshot> {
        unsigned int current_iteration_count = 2;
        if (current_iteration_count < minimum_iteration_count)
            return std::vector<Screenshot>();

        std::vector<Screenshot> screenshots;
        int width = 2; int height = 2;
        if (content_requested.is_set(Screenshot::Content::ColorLDR)) {
            auto* pixels = new unsigned char[width * height * 4];
            screenshots.emplace_back(width, height, Screenshot::Content::ColorLDR, PixelFormat::RGBA32, pixels);
        } else if (content_requested.is_set(Screenshot::Content::ColorHDR)) {
            auto* pixels = new float[width * height * 4];
            screenshots.emplace_back(width, height, Screenshot::Content::ColorLDR, PixelFormat::RGBA_Float, pixels);
        }

        return screenshots;
    };

    Images::allocate(1);

    Matrix4x4f initial_perspective_matrix = Matrix4x4f::identity(), initial_inverse_perspective_matrix = Matrix4x4f::identity();
    SceneRoots::UID scene_ID = SceneRoots::create("Root", RGB::white());
    Cameras::UID cam_ID = Cameras::create("Test cam", scene_ID, initial_perspective_matrix, initial_inverse_perspective_matrix);
    EXPECT_FALSE(Cameras::is_screenshot_requested(cam_ID));

    { // Request LDR image with at least 1 iteration. Filler has two iterations so it should be successful.
        unsigned int minimal_iteration_count = 1;
        Cameras::request_screenshot(cam_ID, Screenshot::Content::ColorLDR, minimal_iteration_count);
        EXPECT_TRUE(Cameras::is_screenshot_requested(cam_ID));

        Cameras::fill_screenshot(cam_ID, screenshot_filler);
        EXPECT_FALSE(Cameras::is_screenshot_requested(cam_ID));

        Image image = Cameras::resolve_screenshot(cam_ID, Screenshot::Content::ColorLDR, "Test image");
        EXPECT_TRUE(image.exists());
        EXPECT_EQ(image.get_width(), 2);
        EXPECT_EQ(image.get_height(), 2);
        EXPECT_EQ(image.get_pixel_format(), PixelFormat::RGBA32);
    }

    { // Request LDR image with at least 3 iteration. Filler has two iterations so it should fail.
        unsigned int minimal_iteration_count = 3;
        Cameras::request_screenshot(cam_ID, Screenshot::Content::ColorLDR, minimal_iteration_count);
        EXPECT_TRUE(Cameras::is_screenshot_requested(cam_ID));

        Cameras::fill_screenshot(cam_ID, screenshot_filler);
        Image image = Cameras::resolve_screenshot(cam_ID, Screenshot::Content::ColorLDR, "Test image");
        EXPECT_FALSE(image.exists());
    }

    { // Request and cancel screenshot.
        Cameras::request_screenshot(cam_ID, Screenshot::Content::ColorLDR, 1);
        EXPECT_TRUE(Cameras::is_screenshot_requested(cam_ID));
        Cameras::cancel_screenshot(cam_ID);
        EXPECT_FALSE(Cameras::is_screenshot_requested(cam_ID));
    }

    Images::deallocate();
}

TEST_F(Scene_Camera, ray_projection) {
    using namespace Bifrost::Math;

    // Create initial projection matrices.
    Matrix4x4f initial_perspective_matrix, initial_inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(1, 1000, PI<float>() / 4.0f, 8.0f / 6.0f,
        initial_perspective_matrix, initial_inverse_perspective_matrix);

    SceneRoots::UID scene_ID = SceneRoots::create("Root", RGB::white());
    Cameras::UID cam_ID = Cameras::create("Test cam", scene_ID, initial_perspective_matrix, initial_inverse_perspective_matrix);
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
} // NS Bifrost

#endif // _BIFROST_SCENE_CAMERA_TEST_H_
