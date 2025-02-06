// SimpleViewer shpere light scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_SPHERE_LIGHT_SCENE_H_
#define _SIMPLEVIEWER_SPHERE_LIGHT_SCENE_H_

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Math/Constants.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>

using namespace Bifrost;

namespace Scenes {
namespace SphereLightScene {

class MovingLight final {
public:
    MovingLight(Scene::SceneNode root_node) {
        Math::Transform light_transform = Math::Transform(Math::Vector3f(0.0f, 1.0f, -10.0f));
        m_light_node = Scene::SceneNode("light", light_transform);
        m_light_node.set_parent(root_node);
        Scene::SphereLight(m_light_node, Math::RGB(300.0f), 2.0f);
    }

    void move(Core::Engine& engine) {
        if (engine.get_time().is_paused())
            return;

        float light_position_delta = fmodf(float(engine.get_time().get_total_time()) * 0.25f, 1.0f);

        float phi = 2.0f * Math::PI<float>() * light_position_delta;
        Math::Vector3f position = Math::Vector3f(cos(phi), sin(phi), 0) * 10.0f;
        m_light_node.set_global_transform(Math::Transform(position));
    }

    static inline void move_callback(Core::Engine& engine, void* state) {
        static_cast<MovingLight*>(state)->move(engine);
    }

private:
    Scene::SceneNode m_light_node;
};

void create(Core::Engine& engine, Scene::CameraID camera_ID, Scene::SceneRoot scene) {
    using namespace Bifrost::Assets;
    using namespace Bifrost::Math;
    using namespace Bifrost::Scene;

    SceneNode root_node = scene.get_root_node();

    { // Setup camera transform.
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 3.0f, -17.0f);
        cam_transform.look_at(Vector3f(0, 1.0f, 0.0f));
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Remove environment light
        if (!scene.get_environment_map().exists())
            scene.set_environment_tint(RGB::black());
    }

    { // Add sphere lights.
        Vector3f positions[] = { Vector3f(5.0f, 0.25f, -10.0f), Math::Vector3f(-10.0f, 2.0f, -2.0f), Math::Vector3f(-2.0f, 10.0f, 2.0f) };
        for (int i = 0; i < 3; ++i) {
            SceneNode light_node = SceneNode("light", Transform(positions[i]));
            light_node.set_parent(root_node);
            RGB light_power = RGB::black();
            light_power[i] = 400.0f;
            SphereLight(light_node, light_power, 4.0f / powf(2, float(i)));
        }
    }

    { // Create checkered floor.
        SceneNode floor_node = create_checkered_floor(400, 1);
        floor_node.set_global_transform(Transform(Vector3f(0, -1.0f, 0)));
        floor_node.set_parent(root_node);
    }

    { // Create material models.
        Materials::Data material0_data = Materials::Data::create_dielectric(RGB(0.02f, 0.27f, 0.33f), 1, 0.02f);
        Materials::Data material1_data = Materials::Data::create_metal(gold_tint, 0.02f);

        Mesh sphere_mesh = MeshCreation::revolved_sphere(32, 16);
        Transform sphere_transform = Transform(Vector3f(0.0f, 1.0f, 0.0f), Quaternionf::identity(), 1.5f);

        for (int m = 0; m < 9; ++m) {
            float lerp_t = m / 8.0f;
            Materials::Data material_data = {};
            material_data.flags = MaterialFlag::None;
            material_data.tint = lerp(material0_data.tint, material1_data.tint, lerp_t);
            material_data.roughness = lerp(material0_data.roughness, material1_data.roughness, lerp_t);
            material_data.specularity = lerp(material0_data.specularity, material1_data.specularity, lerp_t);
            material_data.metallic = lerp(material0_data.metallic, material1_data.metallic, lerp_t);
            material_data.coverage = lerp(material0_data.coverage, material1_data.coverage, lerp_t);
            Material material = Material("Lerped material", material_data);

            Transform transform = Transform(Vector3f(float(m - 4) * 1.25f, 0.0, 0.0f));
            SceneNode node = SceneNode("Model", transform);
            MeshModel(node, sphere_mesh, material);
            node.set_parent(root_node);
        }
    }
}

} // NS SphereLightScene
} // NS Scenes

#endif // _SIMPLEVIEWER_SPHERE_LIGHT_SCENE_H_