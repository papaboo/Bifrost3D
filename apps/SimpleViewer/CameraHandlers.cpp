// Camera navigation.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "CameraHandlers.h"

#include <Bifrost/Input/Keyboard.h>
#include <Bifrost/Input/Mouse.h>

using namespace Bifrost::Core;
using namespace Bifrost::Input;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

// ------------------------------------------------------------------------------------------------
// Camera navigation
// ------------------------------------------------------------------------------------------------
CameraNavigation::CameraNavigation(CameraID camera_ID, float velocity, Vector3f camera_translation, float camera_vertical_rotation, float camera_horizontal_rotation)
    : m_camera_ID(camera_ID)
    , m_velocity(velocity)
{
    Transform transform = Cameras::get_transform(m_camera_ID);

    Vector3f forward = transform.rotation.forward();
    m_vertical_rotation = !isnan(camera_vertical_rotation) ? camera_vertical_rotation : std::atan2(forward.x, forward.z);
    m_horizontal_rotation = !isnan(camera_horizontal_rotation) ? camera_horizontal_rotation : std::asin(forward.y);

    if (!isnan(camera_translation.x))
        transform.translation = camera_translation;
    transform.rotation = Quaternionf::from_angle_axis(m_vertical_rotation, Vector3f::up()) * Quaternionf::from_angle_axis(m_horizontal_rotation, -Vector3f::right());

    Cameras::set_transform(m_camera_ID, transform);
}

void CameraNavigation::navigate(Engine& engine) {
    const Keyboard* keyboard = engine.get_keyboard();
    const Mouse* mouse = engine.get_mouse();

    Transform camera_transform = Cameras::get_transform(m_camera_ID);

    { // Translation
        float strafing = 0.0f;
        if (keyboard->is_pressed(Keyboard::Key::D) || keyboard->is_pressed(Keyboard::Key::Right))
            strafing = 1.0f;
        if (keyboard->is_pressed(Keyboard::Key::A) || keyboard->is_pressed(Keyboard::Key::Left))
            strafing -= 1.0f;

        float forward = 0.0f;
        if (keyboard->is_pressed(Keyboard::Key::W) || keyboard->is_pressed(Keyboard::Key::Up))
            forward = 1.0f;
        if (keyboard->is_pressed(Keyboard::Key::S) || keyboard->is_pressed(Keyboard::Key::Down))
            forward -= 1.0f;

        float velocity = m_velocity;
        if (keyboard->is_pressed(Keyboard::Key::LeftShift) || keyboard->is_pressed(Keyboard::Key::RightShift))
            velocity *= 5.0f;

        if (strafing != 0.0f || forward != 0.0f) {
            Vector3f translation_offset = camera_transform.rotation * Vector3f(strafing, 0.0f, forward);
            float dt = engine.get_time().is_paused() ? engine.get_time().get_raw_delta_time() : engine.get_time().get_smooth_delta_time();
            camera_transform.translation += normalize(translation_offset) * velocity * dt;
        }
    }

    { // Rotation
        if (mouse->is_pressed(Mouse::Button::Left) && mouse->get_delta() != Vector2i::zero()) {

            float vertical_angle_delta = degrees_to_radians(float(mouse->get_delta().x));
            float horizontal_angle_delta = degrees_to_radians(float(mouse->get_delta().y));

            bool rotate_scene = keyboard->is_pressed(Keyboard::Key::LeftControl) || keyboard->is_pressed(Keyboard::Key::RightControl);
            if (rotate_scene) {
                // Rotate the scene root relative to the camera axis.
                Vector3f vertical_axis = camera_transform.rotation.up();
                Quaternionf vertical_rotation = Quaternionf::from_angle_axis(-vertical_angle_delta, vertical_axis);

                Vector3f horizontal_axis = camera_transform.rotation.right();
                Quaternionf horizontal_rotation = Quaternionf::from_angle_axis(-horizontal_angle_delta, horizontal_axis);

                SceneRoot scene_root = Cameras::get_scene_ID(m_camera_ID);
                Transform scene_transform = scene_root.get_root_node().get_local_transform();
                scene_transform.rotation = horizontal_rotation * vertical_rotation * scene_transform.rotation;
                scene_root.get_root_node().set_local_transform(scene_transform);
            } else {
                // Clamp horizontal rotation to -89 and 89 degrees to avoid turning the camera on it's head and the singularities of cross products at the poles.
                m_horizontal_rotation = clamp(m_horizontal_rotation - horizontal_angle_delta, -PI<float>() * 0.49f, PI<float>() * 0.49f);
                m_vertical_rotation += vertical_angle_delta;

                camera_transform.rotation = Quaternionf::from_angle_axis(m_vertical_rotation, Vector3f::up()) * Quaternionf::from_angle_axis(m_horizontal_rotation, -Vector3f::right());
            }
        }
    }

    if (camera_transform != Cameras::get_transform(m_camera_ID))
        Cameras::set_transform(m_camera_ID, camera_transform);

    if (keyboard->was_pressed(Keyboard::Key::Space)) {
        float new_time_scale = engine.get_time().is_paused() ? 1.0f : 0.0f;
        engine.get_time().set_time_scale(new_time_scale);
    }
}

// ------------------------------------------------------------------------------------------------
// Camera viewport handling
// ------------------------------------------------------------------------------------------------
CameraViewportHandler::CameraViewportHandler(CameraID camera_ID, float aspect_ratio, float near, float far)
        : m_camera_ID(camera_ID), m_aspect_ratio(aspect_ratio), m_FOV(PI<float>() / 4.0f)
    , m_near(near), m_far(far) {
    Matrix4x4f perspective_matrix, inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(m_near, m_far, m_FOV, m_aspect_ratio,
        perspective_matrix, inverse_perspective_matrix);

    Cameras::set_projection_matrices(m_camera_ID, perspective_matrix, inverse_perspective_matrix);
}

void CameraViewportHandler::handle(const Engine& engine) {

    const Mouse* mouse = engine.get_mouse();
    float new_FOV = m_FOV - mouse->get_scroll_delta() * engine.get_time().get_smooth_delta_time(); // TODO Non-linear increased / decrease. Especially that it can become negative is an issue.

    float window_aspect_ratio = engine.get_window().get_aspect_ratio();
    if (window_aspect_ratio != m_aspect_ratio || new_FOV != m_FOV) {
        Matrix4x4f perspective_matrix, inverse_perspective_matrix;
        CameraUtils::compute_perspective_projection(m_near, m_far, new_FOV, window_aspect_ratio,
            perspective_matrix, inverse_perspective_matrix);

        Cameras::set_projection_matrices(m_camera_ID, perspective_matrix, inverse_perspective_matrix);
        m_aspect_ratio = window_aspect_ratio;
        m_FOV = new_FOV;
    }
}

void CameraViewportHandler::set_near_and_far(float near, float far) {
    m_near = near;
    m_far = far;

    Matrix4x4f perspective_matrix, inverse_perspective_matrix;
    CameraUtils::compute_perspective_projection(m_near, m_far, m_FOV, m_aspect_ratio,
        perspective_matrix, inverse_perspective_matrix);

    Cameras::set_projection_matrices(m_camera_ID, perspective_matrix, inverse_perspective_matrix);
}
