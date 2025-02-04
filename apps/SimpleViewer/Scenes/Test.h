// SimpleViewer test scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_TEST_SCENE_H_
#define _SIMPLEVIEWER_TEST_SCENE_H_

#include <Scenes/Utils.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Assets/Texture.h>
#include <Bifrost/Core/Engine.h>
#include <Bifrost/Input/Mouse.h>
#include <Bifrost/Math/Transform.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>

#include <StbImageLoader/StbImageLoader.h>

using namespace Bifrost;

namespace Scenes {

class LocalRotator final {
public:
    LocalRotator(Scene::SceneNodeID node_ID, float rotation_stength, Math::Vector3f axis = Math::Vector3f::right())
        : m_node_ID(node_ID), m_rotation_stength(rotation_stength), m_axis(axis) { }

    void rotate(Core::Engine& engine) {
        if (!engine.get_time().is_paused()) {
            Math::Quaternionf rotation = Math::Quaternionf::from_angle_axis(float(engine.get_time().get_smooth_delta_time()) * m_rotation_stength, m_axis);
            Math::Transform delta_transform = Math::Transform(Math::Vector3f::zero(), rotation);
            Scene::SceneNodes::apply_delta_transform(m_node_ID, delta_transform);
        }
    }

private:
    Scene::SceneNodeID m_node_ID;
    float m_rotation_stength;
    Math::Vector3f m_axis;
};

class BlinkingLight final {
public:
    BlinkingLight() 
        : m_light_ID(Scene::LightSourceID::invalid_UID()) {
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
        if (light_should_be_enabled && m_light_ID == LightSourceID::invalid_UID())
            m_light_ID = LightSources::create_sphere_light(m_node_ID, Math::RGB(800.0f, 600.0f, 600.0f), 1.0f);
        
        // Destroy light source.
        if (light_should_be_enabled != 1 && m_light_ID != LightSourceID::invalid_UID()) {
            LightSources::destroy(m_light_ID);
            m_light_ID = LightSourceID::invalid_UID();
        }
    }

private:
    Scene::SceneNodeID m_node_ID;
    Scene::LightSourceID m_light_ID;
};

class BoxGun final {
public:
    BoxGun(Scene::CameraID shooter_node_ID)
        : m_shooter_node_ID(shooter_node_ID)
        , m_cube_mesh_ID(Assets::MeshCreation::cube(1))
        , m_model_ID(Assets::MeshModelID::invalid_UID())
        , m_existed_time(0.0f) {
    }

    void update(Core::Engine& engine) {
        using namespace Bifrost::Assets;
        using namespace Bifrost::Input;
        using namespace Bifrost::Scene;

        if (engine.get_time().is_paused())
            return;

        if (engine.get_mouse()->was_released(Mouse::Button::Right) && m_model_ID == MeshModelID::invalid_UID()) {
            Math::Transform transform = Cameras::get_transform(m_shooter_node_ID);
            transform.scale = 0.1f;
            transform.translation -= transform.rotation.up() * transform.scale;
            SceneNodeID cube_node_ID = SceneNodes::create("Cube", transform);
            m_model_ID = MeshModels::create(cube_node_ID, m_cube_mesh_ID, *Materials::begin()); // Just grab the first material, whatever that is.
            m_existed_time = 0.0f;
        }

        if (m_model_ID != MeshModelID::invalid_UID()) {
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
                m_model_ID = MeshModelID::invalid_UID();
            }
        }
    }

private:
    Scene::CameraID m_shooter_node_ID;
    Assets::MeshID m_cube_mesh_ID;
    Assets::MeshModelID m_model_ID;

    float m_existed_time;
};

void create_test_scene(Core::Engine& engine, Scene::CameraID camera_ID, Scene::SceneNode root_node, const std::filesystem::path& resource_directory) {
    using namespace Bifrost::Assets;
    using namespace Bifrost::Math;
    using namespace Bifrost::Scene;

    { // Setup camera transform.
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 1, -6);
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Create floor.
        SceneNode floor_node = create_checkered_floor(400, 0.66f);
        floor_node.set_global_transform(Transform(Vector3f(0, -0.0005f, 0)));
        floor_node.set_parent(root_node);

        MeshModel floor_model = MeshModels::get_attached_mesh_model(floor_node.get_ID());
        Material floor_material = floor_model.get_material();
        floor_material.set_tint(RGB(0.02f, 0.27f, 0.33f));
        floor_material.set_roughness(0.3f);
    }

    { // Create rotating rings.
        Materials::Data material_data = Materials::Data::create_metal(gold_tint, 0.02f);
        MaterialID material_ID = Materials::create("Gold", material_data);

        unsigned int torus_detail = 64;
        float minor_radius = 0.02f;
        Mesh discontinuous_ring_mesh = MeshCreation::torus(torus_detail, torus_detail, minor_radius, { MeshFlag::Position, MeshFlag::Normal });
        Mesh ring_mesh = MeshUtils::merge_duplicate_vertices(discontinuous_ring_mesh, { MeshFlag::Position, MeshFlag::Normal });

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

        { // Glass ball inside the rings
            Materials::Data glass_material_data = Materials::Data::create_transmissive(RGB(0.95f), 1.0f, glass_specularity);
            Material glass_material = Materials::create("Glass", glass_material_data);

            auto world_mask_path = resource_directory / "WorldMask.png";
            Image world_mask = StbImageLoader::load(world_mask_path.generic_string());
            world_mask.change_format(PixelFormat::Roughness8, 1.0f);
            glass_material.set_tint_roughness_texture(Textures::create2D(world_mask.get_ID()));

            Mesh sphere_mesh = MeshCreation::revolved_sphere(64, 32);
            Vector2f* uvs = sphere_mesh.get_texcoords();
            for (unsigned int v = 0; v < sphere_mesh.get_vertex_count(); ++v)
                uvs[v].y = 1.0f - uvs[v].y;
            
            SceneNode sphere_node = SceneNodes::create("Glass sphere", Transform(Vector3f(0, 0.5f, 0), Quaternionf::identity(), 0.7f));
            MeshModels::create(sphere_node.get_ID(), sphere_mesh.get_ID(), glass_material.get_ID());
            sphere_node.set_parent(root_node);
        }
    }

    { // Destroyable cylinder. TODO Implement destruction of the mesh, model and scene node.
        // Checkered roughness texture.
        const int size = 32;
        Image roughness = Images::create2D("Cylinder roughness", PixelFormat::Roughness8, 1.0f, Vector2ui(size));
        unsigned char* roughness_pixels = roughness.get_pixels<unsigned char>();
        for (unsigned int y = 0; y < size; ++y)
            for (unsigned int x = 0; x < size; ++x)
                roughness_pixels[x + y * size] = ((x & 1) == (y & 1)) ? 63 : 127;

        Materials::Data material_data = Materials::Data::create_metal(iron_tint, 1.0f);
        material_data.tint_roughness_texture_ID = Textures::create2D(roughness.get_ID());
        MaterialID material_ID = Materials::create("Iron", material_data);

        Transform transform = Transform(Vector3f(-1.5f, 0.5f, 0.0f));
        SceneNode cylinder_node = SceneNodes::create("Destroyed Cylinder", transform);
        MeshID cylinder_mesh_ID = MeshCreation::cylinder(4, 16);
        MeshModels::create(cylinder_node.get_ID(), cylinder_mesh_ID, material_ID);
        cylinder_node.set_parent(root_node);
    }

    { // Copper / rubber sphere.
        struct TintRoughness {
            unsigned char r, g, b, roughness;
        };

        const TintRoughness rubber_tint = { 3, 3, 3, 192 };
        const TintRoughness metal_tint = { 255, 255, 255, 0 };

        const int size = 1024;
        const int metal_streak_count = 5;
        Image tint_roughness = Images::create2D("Sphere tint", PixelFormat::RGBA32, 2.2f, Vector2ui(size));
        TintRoughness* tint_roughness_pixels = tint_roughness.get_pixels<TintRoughness>();
        std::fill_n(tint_roughness_pixels, size* size, rubber_tint);
        Image metalness = Images::create2D("Sphere metalness", PixelFormat::Alpha8, 1.0f, Vector2ui(size));
        unsigned char* metalness_pixels = metalness.get_pixels<unsigned char>();
        std::fill_n(metalness_pixels, size* size, (unsigned char)0u);
        #pragma omp parallel for 
        for (int s = 0; s < metal_streak_count; ++s) {
            int streak_begin = size * s / metal_streak_count + 40;
            int streak_end = size * (s + 1) / metal_streak_count - 40;
            for (int y = 0; y < size; ++y) {
                for (int x = streak_begin; x < streak_end; ++x) {
                    int _x = x % size;
                    metalness_pixels[_x + y * size] = 255u;
                    tint_roughness_pixels[_x + y * size] = metal_tint;
                    float t = inverse_lerp<float>(float(streak_begin), streak_end - 1.0f, float(x));
                    float roughness = sin(t * PI<float>());
                    tint_roughness_pixels[_x + y * size].roughness = 255u - unsigned char(roughness * 255u);
                }

                streak_begin += 1;
                streak_end += 1;
            }
        }

        Materials::Data material_data = Materials::Data::create_dielectric(copper_tint, 0.75f, 0.04f);
        material_data.tint_roughness_texture_ID = Textures::create2D(tint_roughness.get_ID());
        material_data.metallic = 1.0f;
        material_data.metallic_texture_ID = Textures::create2D(metalness.get_ID());
        MaterialID material_ID = Materials::create("Copper/rubber", material_data);

        Transform transform = Transform(Vector3f(1.5f, 0.5f, 0.0f));
        SceneNode sphere_node = SceneNodes::create("Sphere", transform);
        MeshID sphere_mesh_ID = MeshCreation::revolved_sphere(128, 64);
        MeshModels::create(sphere_node.get_ID(), sphere_mesh_ID, material_ID);
        sphere_node.set_parent(root_node);

        LocalRotator* simple_rotator = new LocalRotator(sphere_node.get_ID(), 0.2f, Vector3f::up());
        engine.add_mutating_callback([=, &engine] { simple_rotator->rotate(engine); });
    }

    { // Partial coverage plastic torus.
        unsigned int width = 16, height = 16;
        ImageID coverage_image_ID = Images::create2D("Grid", PixelFormat::Alpha8, 1.0f, Vector2ui(width, height));
        unsigned char* pixels = Images::get_pixels<unsigned char>(coverage_image_ID);
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                unsigned char* pixel = pixels + (x + y * width);
                unsigned char intensity = ((x & 1) == 0 || (y & 1) == 0) ? 255 : 0;
                pixel[0] = intensity;
            }
        }

        Materials::Data material_data = Materials::Data::create_dielectric(RGB(0.005f, 0.01f, 0.25f), 0.05f, 0.04f);
        material_data.coverage = 0.5f;
        material_data.coverage_texture_ID = Textures::create2D(coverage_image_ID, MagnificationFilter::None, MinificationFilter::None);
        material_data.flags = { MaterialFlag::Cutout, MaterialFlag::ThinWalled };
        MaterialID material_ID = Materials::create("Plastic", material_data);

        Transform transform = Transform(Vector3f(3.0f, 0.35f, 0.0f), Quaternionf::identity(), 0.7f);
        SceneNode torus_node = SceneNodes::create("Swizz torus", transform);
        MeshID torus_mesh_ID = MeshCreation::torus(64, 64, 0.49f);

        MeshModels::create(torus_node.get_ID(), torus_mesh_ID, material_ID);
        torus_node.set_parent(root_node);
    }

    { // GUN!
        CameraID cam_ID = *Cameras::begin();
        BoxGun* boxgun = new BoxGun(cam_ID);
        engine.add_mutating_callback([=, &engine] { boxgun->update(engine); });
    }

    Vector3f light_position = Vector3f(100.0f, 20.0f, 100.0f);
    Transform light_transform = Transform(light_position);
    SceneNodeID light_node_ID = SceneNodes::create("Light", light_transform);
    LightSourceID light_ID = LightSources::create_sphere_light(light_node_ID, RGB(1000000.0f), 0.0f);
    SceneNodes::set_parent(light_node_ID, root_node.get_ID());

    BlinkingLight* blinking_light = new BlinkingLight();
    engine.add_mutating_callback([=, &engine] { blinking_light->blink(engine); });
}

} // NS Scenes

#endif // _SIMPLEVIEWER_TEST_SCENE_H_