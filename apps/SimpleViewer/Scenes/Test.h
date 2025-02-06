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
    LocalRotator(Scene::SceneNode node, float rotation_stength, Math::Vector3f axis = Math::Vector3f::right())
        : m_node(node), m_rotation_stength(rotation_stength), m_axis(axis) { }

    void rotate(Core::Engine& engine) {
        if (!engine.get_time().is_paused()) {
            Math::Quaternionf rotation = Math::Quaternionf::from_angle_axis(float(engine.get_time().get_smooth_delta_time()) * m_rotation_stength, m_axis);
            Math::Transform delta_transform = Math::Transform(Math::Vector3f::zero(), rotation);
            m_node.apply_delta_transform(delta_transform);
        }
    }

private:
    Scene::SceneNode m_node;
    float m_rotation_stength;
    Math::Vector3f m_axis;
};

class BlinkingLight final {
public:
    BlinkingLight() 
        : m_light(Scene::SphereLight::invalid()) {
        Math::Transform transform = Math::Transform(Math::Vector3f(1,5,2));
        m_node = Scene::SceneNode("BlinkingLight", transform);
    }

    void blink(Core::Engine& engine) {
        using namespace Scene;

        if (engine.get_time().is_paused())
            return;

        // Blink every two seconds.
        int light_should_be_enabled = int(engine.get_time().get_total_time() / 2) & 1;
        
        // Create light source.
        if (light_should_be_enabled && !m_light.exists())
            m_light = SphereLight(m_node, Math::RGB(800.0f, 600.0f, 600.0f), 1.0f);
        
        // Destroy light source.
        if (light_should_be_enabled != 1 && m_light.exists()) {
            m_light.destroy();
            m_light = SphereLight::invalid();
        }
    }

private:
    Scene::SceneNode m_node;
    Scene::SphereLight m_light;
};

class BoxGun final {
public:
    BoxGun(Scene::CameraID shooter_node_ID)
        : m_shooter_node_ID(shooter_node_ID)
        , m_box_mesh(Assets::MeshCreation::box(1))
        , m_model(Assets::MeshModel::invalid())
        , m_existed_time(0.0f) {
    }

    void update(Core::Engine& engine) {
        using namespace Bifrost::Assets;
        using namespace Bifrost::Input;
        using namespace Bifrost::Scene;

        if (engine.get_time().is_paused())
            return;

        if (engine.get_mouse()->was_released(Mouse::Button::Right) && !m_model.exists()) {
            Math::Transform transform = Cameras::get_transform(m_shooter_node_ID);
            transform.scale = 0.1f;
            transform.translation -= transform.rotation.up() * transform.scale;
            SceneNode box_node = SceneNode("Box", transform);
            m_model = MeshModel(box_node, m_box_mesh, *Materials::begin()); // Just grab the first material, whatever that is.
            m_existed_time = 0.0f;
        }

        if (m_model.exists()) {
            float delta_time = engine.get_time().get_smooth_delta_time();
            m_existed_time += delta_time;
            SceneNode box_node = m_model.get_scene_node();
            Math::Transform transform = box_node.get_global_transform();
            transform.translation += transform.rotation.forward() * 3.0f * delta_time;
            box_node.set_global_transform(transform);

            // box bullets should only exist for 2 seconds.
            if (m_existed_time > 2.0f) {
                box_node.destroy();
                m_model.destroy();
                m_model = MeshModel::invalid();
            }
        }
    }

private:
    Scene::CameraID m_shooter_node_ID;
    Assets::Mesh m_box_mesh;
    Assets::MeshModel m_model;

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
        Material material = Material::create_metal("Gold", gold_tint, 0.02f);

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
            MeshUtils::compute_normals(ring_mesh);
        }

        // Create the rings.
        float scale = 1.0f - minor_radius * 2.0f;
        SceneNode parent_node = root_node;
        for (int i = 0; i < 3; ++i) {
            Quaternionf local_rot = Quaternionf::from_angle_axis(0.5f * Math::PI<float>(), Vector3f::right()) *
                                    Quaternionf::from_angle_axis(0.5f * Math::PI<float>(), Vector3f::forward());
            Transform transform = parent_node.get_global_transform() * Transform(Vector3f::zero(), local_rot, scale);
            transform.translation = Vector3f(0, 0.5f, 0);

            SceneNode ring_node = SceneNode("Rotating ring", transform);
            MeshModel(ring_node, ring_mesh, material);
            ring_node.set_parent(parent_node);
            parent_node = ring_node;

            LocalRotator* simple_rotator = new LocalRotator(ring_node, i * 0.31415f * 0.5f + 0.2f);
            engine.add_mutating_callback([=, &engine] { simple_rotator->rotate(engine); });
        }

        { // Glass ball inside the rings
            Material glass_material = Material::create_transmissive("Glass", RGB(0.95f), 1.0f, glass_specularity);

            auto world_mask_path = resource_directory / "WorldMask.png";
            Image world_mask = StbImageLoader::load(world_mask_path.generic_string());
            world_mask.change_format(PixelFormat::Roughness8, 1.0f);
            glass_material.set_tint_roughness_texture(Texture::create2D(world_mask));

            Mesh sphere_mesh = MeshCreation::revolved_sphere(64, 32);
            Vector2f* uvs = sphere_mesh.get_texcoords();
            for (unsigned int v = 0; v < sphere_mesh.get_vertex_count(); ++v)
                uvs[v].y = 1.0f - uvs[v].y;
            
            SceneNode sphere_node = SceneNode("Glass sphere", Transform(Vector3f(0, 0.5f, 0), Quaternionf::identity(), 0.7f));
            MeshModel(sphere_node, sphere_mesh, glass_material);
            sphere_node.set_parent(root_node);
        }
    }

    { // Destroyable cylinder. TODO Implement destruction of the mesh, model and scene node.
        // Checkered roughness texture.
        const int size = 32;
        Image roughness = Image::create2D("Cylinder roughness", PixelFormat::Roughness8, 1.0f, Vector2ui(size));
        unsigned char* roughness_pixels = roughness.get_pixels<unsigned char>();
        for (unsigned int y = 0; y < size; ++y)
            for (unsigned int x = 0; x < size; ++x)
                roughness_pixels[x + y * size] = ((x & 1) == (y & 1)) ? 63 : 127;

        Materials::Data material_data = Materials::Data::create_metal(iron_tint, 1.0f);
        material_data.tint_roughness_texture_ID = Texture::create2D(roughness).get_ID();
        Material material = Material("Iron", material_data);

        Transform transform = Transform(Vector3f(-1.5f, 0.5f, 0.0f));
        SceneNode cylinder_node = SceneNode("Destroyed Cylinder", transform);
        Mesh cylinder_mesh = MeshCreation::cylinder(4, 16);
        MeshModel(cylinder_node, cylinder_mesh, material);
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
        Image tint_roughness = Image::create2D("Sphere tint", PixelFormat::RGBA32, 2.2f, Vector2ui(size));
        TintRoughness* tint_roughness_pixels = tint_roughness.get_pixels<TintRoughness>();
        std::fill_n(tint_roughness_pixels, size* size, rubber_tint);
        Image metalness = Image::create2D("Sphere metalness", PixelFormat::Alpha8, 1.0f, Vector2ui(size));
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

        Materials::Data material_data = Materials::Data::create_dielectric(copper_tint, 0.75f);
        material_data.tint_roughness_texture_ID = Texture::create2D(tint_roughness).get_ID();
        material_data.metallic = 1.0f;
        material_data.metallic_texture_ID = Texture::create2D(metalness).get_ID();
        Material material = Material("Copper/rubber", material_data);

        Transform transform = Transform(Vector3f(1.5f, 0.5f, 0.0f));
        SceneNode sphere_node = SceneNode("Sphere", transform);
        Mesh sphere_mesh = MeshCreation::revolved_sphere(128, 64);
        MeshModel(sphere_node, sphere_mesh, material);
        sphere_node.set_parent(root_node);

        LocalRotator* simple_rotator = new LocalRotator(sphere_node.get_ID(), 0.2f, Vector3f::up());
        engine.add_mutating_callback([=, &engine] { simple_rotator->rotate(engine); });
    }

    { // Partial coverage plastic torus.
        unsigned int width = 16, height = 16;
        Image coverage_image = Image::create2D("Grid", PixelFormat::Alpha8, 1.0f, Vector2ui(width, height));
        unsigned char* pixels = coverage_image.get_pixels<unsigned char>();
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                unsigned char* pixel = pixels + (x + y * width);
                unsigned char intensity = ((x & 1) == 0 || (y & 1) == 0) ? 255 : 0;
                pixel[0] = intensity;
            }
        }

        Materials::Data material_data = Materials::Data::create_dielectric(RGB(0.005f, 0.01f, 0.25f), 0.05f);
        material_data.coverage = 0.5f;
        material_data.coverage_texture_ID = Texture::create2D(coverage_image, MagnificationFilter::None, MinificationFilter::None).get_ID();
        material_data.flags = { MaterialFlag::Cutout, MaterialFlag::ThinWalled };
        Material material = Material("Plastic", material_data);

        Transform transform = Transform(Vector3f(3.0f, 0.35f, 0.0f), Quaternionf::identity(), 0.7f);
        SceneNode torus_node = SceneNode("Swizz torus", transform);
        Mesh torus_mesh = MeshCreation::torus(64, 64, 0.49f);

        MeshModel(torus_node, torus_mesh, material);
        torus_node.set_parent(root_node);
    }

    { // GUN!
        CameraID cam_ID = *Cameras::begin();
        BoxGun* boxgun = new BoxGun(cam_ID);
        engine.add_mutating_callback([=, &engine] { boxgun->update(engine); });
    }

    Vector3f light_position = Vector3f(100.0f, 20.0f, 100.0f);
    Transform light_transform = Transform(light_position);
    SceneNode light_node = SceneNode("Light", light_transform);
    light_node.set_parent(root_node);
    SphereLight light = SphereLight(light_node, RGB(1000000.0f), 0.0f);

    BlinkingLight* blinking_light = new BlinkingLight();
    engine.add_mutating_callback([=, &engine] { blinking_light->blink(engine); });
}

} // NS Scenes

#endif // _SIMPLEVIEWER_TEST_SCENE_H_