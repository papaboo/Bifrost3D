// Camera navigation.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_CAMERA_NAVIGATION_H_
#define _SIMPLEVIEWER_CAMERA_NAVIGATION_H_

#include <Bifrost/Core/Engine.h>
#include <Bifrost/Scene/Camera.h>

// ------------------------------------------------------------------------------------------------
// Camera navigation
// ------------------------------------------------------------------------------------------------
class CameraNavigation final {
public:

    CameraNavigation(Bifrost::Scene::CameraID camera_ID, float velocity, Bifrost::Math::Vector3f camera_translation,
        float camera_vertical_rotation, float camera_horizontal_rotation);

    void navigate(Bifrost::Core::Engine& engine);

    inline Bifrost::Scene::CameraID get_camera_ID() const { return m_camera_ID; }
    inline float get_velocity() const { return m_velocity; }
    inline void set_velocity(float velocity) { m_velocity = velocity; }

private:
    Bifrost::Scene::CameraID m_camera_ID;
    float m_vertical_rotation;
    float m_horizontal_rotation;
    float m_velocity;
};

// ------------------------------------------------------------------------------------------------
// Camera viewport handling
// ------------------------------------------------------------------------------------------------
class CameraViewportHandler final {
public:
    CameraViewportHandler(Bifrost::Scene::CameraID camera_ID, float aspect_ratio, float near, float far);

    void handle(const Bifrost::Core::Engine& engine);

    void set_near_and_far(float near, float far);

private:
    Bifrost::Scene::CameraID m_camera_ID;
    float m_aspect_ratio;
    float m_FOV;
    float m_near, m_far;
};

#endif // _SIMPLEVIEWER_CAMERA_NAVIGATION_H_