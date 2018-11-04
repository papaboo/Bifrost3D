// SimpleViewer test scene.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_TEST_SCENE_H_
#define _SIMPLEVIEWER_TEST_SCENE_H_

#include <Cogwheel/Assets/Image.h>
#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshCreation.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Assets/Texture.h>
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Input/Mouse.h>
#include <Cogwheel/Math/Transform.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/LightSource.h>
#include <Cogwheel/Scene/SceneNode.h>

using namespace Cogwheel;

namespace Scenes {

class LocalRotator final {
public:
    LocalRotator(Scene::SceneNodes::UID node_ID, float rotation_stength)
        : m_node_ID(node_ID), m_rotation_stength(rotation_stength){
    }

    void rotate(Core::Engine& engine) {
        if (!engine.get_time().is_paused()) {
            Math::Quaternionf rotation = Math::Quaternionf::from_angle_axis(float(engine.get_time().get_smooth_delta_time()) * m_rotation_stength, Math::Vector3f::right());
            Math::Transform delta_transform = Math::Transform(Math::Vector3f::zero(), rotation);
            Scene::SceneNodes::apply_delta_transform(m_node_ID, delta_transform);
        }
    }

private:
    Scene::SceneNodes::UID m_node_ID;
    float m_rotation_stength;
};

class BlinkingLight final {
public:
    BlinkingLight() 
        : m_light_ID(Scene::LightSources::UID::invalid_UID()) {
        Math::Transform transform = Math::Transform(Math::Vector3f(1,5,2));
        m_node_ID = Scene::SceneNodes::create("BlinkingLight", transform);
    }

    void blink(Core::Engine& engine) {
        using namespace Scene;

        if (engine.get_time().is_paused())
            return;

        // Blink every two seconds.
        int light_should_be_enabled = int(engine.get_time().get_total_time() / 2) & 1;
        
        // Create light source.
        if (light_should_be_enabled && m_light_ID == LightSources::UID::invalid_UID())
            m_light_ID = LightSources::create_sphere_light(m_node_ID, Math::RGB(800.0f, 600.0f, 600.0f), 1.0f);
        
        // Destroy light source.
        if (light_should_be_enabled != 1 && m_light_ID != LightSources::UID::invalid_UID()) {
            LightSources::destroy(m_light_ID);
            m_light_ID = LightSources::UID::invalid_UID();
        }
    }

private:
    Scene::SceneNodes::UID m_node_ID;
    Scene::LightSources::UID m_light_ID;
};

class BoxGun final {
public:
    BoxGun(Scene::Cameras::UID shooter_node_ID)
        : m_shooter_node_ID(shooter_node_ID)
        , m_cube_mesh_ID(Assets::MeshCreation::cube(1))
        , m_model_ID(Assets::MeshModels::UID::invalid_UID())
        , m_existed_time(0.0f) {
    }

    void update(Core::Engine& engine) {
        using namespace Cogwheel::Assets;
        using namespace Cogwheel::Input;
        using namespace Cogwheel::Scene;

        if (engine.get_time().is_paused())
            return;

        if (engine.get_mouse()->was_released(Mouse::Button::Right) && m_model_ID == MeshModels::UID::invalid_UID()) {
            Math::Transform transform = Cameras::get_transform(m_shooter_node_ID);
            transform.scale = 0.1f;
            transform.translation -= transform.rotation.up() * transform.scale;
            SceneNodes::UID cube_node_ID = SceneNodes::create("Cube", transform);
            m_model_ID = MeshModels::create(cube_node_ID, m_cube_mesh_ID, *Materials::begin()); // Just grab the first material, whatever that is.
            m_existed_time = 0.0f;
        }

        if (m_model_ID != MeshModels::UID::invalid_UID()) {
            float delta_time = engine.get_time().get_smooth_delta_time();
            m_existed_time += delta_time;
            SceneNode cube_node = MeshModels::get_scene_node_ID(m_model_ID);
            Math::Transform transform = cube_node.get_global_transform();
            transform.translation += transform.rotation.forward() * 3.0f * delta_time;
            cube_node.set_global_transform(transform);

            // Cube bullets should only exist for 2 seconds.
            if (m_existed_time > 2.0f) {
                SceneNodes::destroy(cube_node.get_ID());
                MeshModels::destroy(m_model_ID);
                m_model_ID = MeshModels::UID::invalid_UID();
            }
        }
    }

private:
    Scene::Cameras::UID m_shooter_node_ID;
    Assets::Meshes::UID m_cube_mesh_ID;
    Assets::MeshModels::UID m_model_ID;

    float m_existed_time;
};

void create_test_scene(Core::Engine& engine, Scene::Cameras::UID camera_ID, Scene::SceneNode root_node) {
    using namespace Cogwheel::Assets;
    using namespace Cogwheel::Math;
    using namespace Cogwheel::Scene;

    { // Setup camera transform.
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 1, -6);
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Create floor.
        // A checker pattern texture would be really nice on the floor.
        unsigned int width = 16, height = 16;
        Images::UID image_ID = Images::create2D("Checker", PixelFormat::RGBA32, 2.2f, Vector2ui(width, height));
        unsigned char* pixels = (unsigned char*)Images::get_pixels(image_ID);
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                unsigned char* pixel = pixels + (x + y * width) * 4u;
                unsigned char intensity = ((x & 1) == (y & 1)) ? 2 : 255;
                pixel[0] = intensity; pixel[1] = intensity; pixel[2] = intensity; pixel[3] = 255;
            }
        }

        Materials::Data material_data = Materials::Data::create_dielectric(RGB(0.02f, 0.27f, 0.33f), 0.3f, 0.25f);
        material_data.tint_texture_ID = Textures::create2D(image_ID, MagnificationFilter::None, MinificationFilter::None);
        Materials::UID material_ID = Materials::create("Floor", material_data);

        SceneNode plane_node = SceneNodes::create("Floor", Transform(Vector3f(0, -0.0005f, 0), Quaternionf::identity(), 10));
        Meshes::UID plane_mesh_ID = MeshCreation::plane(1, { MeshFlag::Position, MeshFlag::Texcoord });
        MeshModels::create(plane_node.get_ID(), plane_mesh_ID, material_ID);
        plane_node.set_parent(root_node);
    }

    { // Create rotating rings.
        Materials::Data material_data = Materials::Data::create_metal(RGB(1.0f, 0.766f, 0.336f), 0.02f, 0.0f);
        Materials::UID material_ID = Materials::create("Gold", material_data);

        unsigned int torus_detail = 64;
        float minor_radius = 0.02f;
        Mesh ring_mesh = MeshCreation::torus(torus_detail, torus_detail, minor_radius, { MeshFlag::Position, MeshFlag::Normal });

        { // Ringify
            Vector3f* positions = ring_mesh.get_positions();
            Vector3f* normals = ring_mesh.get_normals();
            for (unsigned int i = 0; i < ring_mesh.get_vertex_count(); ++i) {
                Vector3f ring_center = positions[i] - normals[i] * minor_radius;
                Vector3f dir = normalize(ring_center);
                Vector3f tangent = cross(dir, Vector3f::up());
                positions[i] += Vector3f::up() * dot(Vector3f::up(), normals[i]) * 2.5f * minor_radius;
                positions[i] -= dir * (1.0f - abs(dot(dir, normals[i]))) * 0.5f * minor_radius;
            }
            MeshUtils::compute_normals(ring_mesh.get_ID());
            
            // Fix normals at the discontinuity where the ring mesh starts and ends.
            Vector3f* end_normals = normals + ring_mesh.get_vertex_count() - torus_detail - 2;
            for (unsigned int v = 0; v < torus_detail + 1; ++v) {
                Vector3f begin_normal = normals[v];
                Vector3f end_normal = end_normals[v];
                normals[v] = end_normals[v] = normalize(begin_normal + end_normal);
            }
        }

        // Create the rings.
        float scale = 1.0f - minor_radius * 2.0f;
        SceneNode parent_node = root_node;
        for (int i = 0; i < 3; ++i) {
            Quaternionf local_rot = Quaternionf::from_angle_axis(0.5f * Math::PI<float>(), Vector3f::right()) *
                                    Quaternionf::from_angle_axis(0.5f * Math::PI<float>(), Vector3f::forward());
            Transform transform = parent_node.get_global_transform() * Transform(Vector3f::zero(), local_rot, scale);
            transform.translation = Vector3f(0, 0.5f, 0);

            SceneNode ring_node = SceneNodes::create("Rotating ring", transform);
            MeshModels::create(ring_node.get_ID(), ring_mesh.get_ID(), material_ID);
            ring_node.set_parent(parent_node);
            parent_node = ring_node;

            LocalRotator* simple_rotator = new LocalRotator(ring_node.get_ID(), i * 0.31415f * 0.5f + 0.2f);
            engine.add_mutating_callback([=, &engine] { simple_rotator->rotate(engine); });
        }
    }

    { // Destroyable cylinder. TODO Implement destruction of the mesh, model and scene node.
        Materials::Data material_data = Materials::Data::create_metal(RGB(0.56f, 0.57f, 0.58f), 0.4f, 0.75f);
        Materials::UID material_ID = Materials::create("Iron", material_data);

        Transform transform = Transform(Vector3f(-1.5f, 0.5f, 0.0f));
        SceneNode cylinder_node = SceneNodes::create("Destroyed Cylinder", transform);
        Meshes::UID cylinder_mesh_ID = MeshCreation::cylinder(4, 16, { MeshFlag::Position, MeshFlag::Normal });
        MeshModels::create(cylinder_node.get_ID(), cylinder_mesh_ID, material_ID);
        cylinder_node.set_parent(root_node);
    }

    { // Sphere for the hell of it.
        Materials::Data material_data = Materials::Data::create_dielectric(RGB(0.001f, 0.001f, 0.001f), 0.75f, 0.5f);
        Materials::UID material_ID = Materials::create("Dark rubber", material_data);

        Transform transform = Transform(Vector3f(1.5f, 0.5f, 0.0f));
        SceneNode sphere_node = SceneNodes::create("Sphere", transform);
        Meshes::UID sphere_mesh_ID = MeshCreation::revolved_sphere(128, 64, { MeshFlag::Position, MeshFlag::Normal });
        MeshModels::create(sphere_node.get_ID(), sphere_mesh_ID, material_ID);
        sphere_node.set_parent(root_node);
    }

    { // Partial coverage copper torus.
        unsigned int width = 17, height = 17;
        Images::UID image_ID = Images::create2D("Grid", PixelFormat::I8, 1.0f, Vector2ui(width, height));
        unsigned char* pixels = Images::get_pixels<unsigned char>(image_ID);
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                unsigned char* pixel = pixels + (x + y * width);
                unsigned char intensity = ((x & 1) == 0 || (y & 1) == 0) ? 255 : 0;
                pixel[0] = intensity;
            }
        }

        Materials::Data material_data = Materials::Data::create_dielectric(RGB(0.005f, 0.01f, 0.25f), 0.05f, 0.5f);
        material_data.coverage_texture_ID = Textures::create2D(image_ID, MagnificationFilter::None, MinificationFilter::None);
        material_data.flags = MaterialFlag::Cutout;
        Materials::UID material_ID = Materials::create("Plastic", material_data);

        Transform transform = Transform(Vector3f(3.0f, 0.35f, 0.0f), Quaternionf::identity(), 0.5f);
        SceneNode torus_node = SceneNodes::create("Swizz torus", transform);
        Meshes::UID torus_mesh_ID = MeshCreation::torus(64, 64, 0.7f);
        MeshModels::create(torus_node.get_ID(), torus_mesh_ID, material_ID);
        torus_node.set_parent(root_node);
    }

    { // GUN!
        Cameras::UID cam_ID = *Cameras::begin();
        BoxGun* boxgun = new BoxGun(cam_ID);
        engine.add_mutating_callback([=, &engine] { boxgun->update(engine); });
    }

    Vector3f light_position = Vector3f(100.0f, 20.0f, 100.0f);
    Transform light_transform = Transform(light_position);
    SceneNodes::UID light_node_ID = SceneNodes::create("Light", light_transform);
    LightSources::UID light_ID = LightSources::create_sphere_light(light_node_ID, RGB(1000000.0f), 0.0f);
    SceneNodes::set_parent(light_node_ID, root_node.get_ID());

    BlinkingLight* blinking_light = new BlinkingLight();
    engine.add_mutating_callback([=, &engine] { blinking_light->blink(engine); });
}

} // NS Scenes

#endif // _SIMPLEVIEWER_TEST_SCENE_H_