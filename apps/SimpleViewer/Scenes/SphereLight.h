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
        m_light_node = Scene::SceneNodes::create("light", light_transform);
        m_light_node.set_parent(root_node);
        Scene::LightSources::create_sphere_light(m_light_node.get_ID(), Math::RGB(300.0f), 2.0f);
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

void create(Core::Engine& engine, Scene::CameraID camera_ID, Scene::SceneRootID scene_ID) {
    using namespace Bifrost::Assets;
    using namespace Bifrost::Math;
    using namespace Bifrost::Scene;

    SceneNode root_node = SceneRoots::get_root_node(scene_ID);

    { // Setup camera transform.
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 3.0f, -17.0f);
        cam_transform.look_at(Vector3f(0, 1.0f, 0.0f));
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Remove environment light
        SceneRoot scene = scene_ID;
        if (!Textures::has(scene.get_environment_map()))
            scene.set_environment_tint(RGB::black());
    }

    { // Add sphere lights.
        Vector3f positions[] = { Vector3f(5.0f, 0.25f, -10.0f), Math::Vector3f(-10.0f, 2.0f, -2.0f), Math::Vector3f(-2.0f, 10.0f, 2.0f) };
        for (int i = 0; i < 3; ++i) {
            SceneNode light_node = SceneNodes::create("light", Transform(positions[i]));
            light_node.set_parent(root_node);
            RGB light_power = RGB::black();
            light_power[i] = 400.0f;
            LightSources::create_sphere_light(light_node.get_ID(), light_power, 4.0f / powf(2, float(i)));
        }
    }

    { // Create checkered floor.
        unsigned int size = 41;
        RGBA32 white_tile = { unsigned char(127), unsigned char(127), unsigned char(127), unsigned char(51) };
        RGBA32 black_tile = { unsigned char(1), unsigned char(1), unsigned char(1), unsigned char(5) };
        ImageID tint_roughness_image_ID = Images::create2D("Floor image", PixelFormat::RGBA32, 2.2f, Vector2ui(size, size));
        Images::set_mipmapable(tint_roughness_image_ID, true);
        RGBA32* tint_roughness_pixels = Images::get_pixels<RGBA32>(tint_roughness_image_ID);
        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                bool is_black = (x & 1) != (y & 1);
                RGBA32* tile = tint_roughness_pixels + (x + y * size);
                *tile = is_black ? black_tile : white_tile;
            }
        }

        Materials::Data material_data = Materials::Data::create_dielectric(RGB::white(), 1, 0.04f);
        material_data.tint_roughness_texture_ID = Textures::create2D(tint_roughness_image_ID, MagnificationFilter::None, MinificationFilter::Trilinear);
        material_data.flags = MaterialFlag::ThinWalled;
        MaterialID material_ID = Materials::create("Floor", material_data);

        SceneNode plane_node = SceneNodes::create("Floor", Transform(Vector3f(0.0, -1.0, 0.0), Quaternionf::identity(), float(size)));
        MeshID plane_mesh_ID = MeshCreation::plane(1, { MeshFlag::Position, MeshFlag::Texcoord });
        MeshModels::create(plane_node.get_ID(), plane_mesh_ID, material_ID);
        plane_node.set_parent(root_node);
    }

    { // Create material models.
        Materials::Data material0_data = Materials::Data::create_dielectric(RGB(0.02f, 0.27f, 0.33f), 1, 0.02f);
        Materials::Data material1_data = Materials::Data::create_metal(gold_tint, 0.02f);

        MeshID sphere_mesh_ID = MeshCreation::revolved_sphere(32, 16);
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
            MaterialID material_ID = Materials::create("Lerped material", material_data);

            Transform transform = Transform(Vector3f(float(m - 4) * 1.25f, 0.0, 0.0f));
            SceneNode node = SceneNodes::create("Model", transform);
            MeshModels::create(node.get_ID(), sphere_mesh_ID, material_ID);
            node.set_parent(root_node);
        }
    }
}

} // NS SphereLightScene
} // NS Scenes

#endif // _SIMPLEVIEWER_SPHERE_LIGHT_SCENE_H_