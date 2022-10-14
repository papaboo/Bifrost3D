// SimpleViewer sphere scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Scenes/Sphere.h>

#include <Bifrost/Assets/Image.h>
#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Math/Transform.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace Scenes {

void create_sphere_scene(Bifrost::Scene::CameraID camera_ID, Bifrost::Scene::SceneNode root_node) {
    const int sphere_model_count = 5;

    { // Setup camera transform.
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 3.0f, -17.0f);
        cam_transform.look_at(Vector3f(0, 1.0f, 0.0f));
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Create floor.
      // A checker pattern texture would be really nice on the floor.
        unsigned int size = 81;
        ImageID tint_roughness_image_ID = Images::create2D("Floor color", PixelFormat::RGBA32, 2.2f, Vector2ui(size, size));
        Images::set_mipmapable(tint_roughness_image_ID, true);
        unsigned char* tint_roughness_pixels = (unsigned char*)Images::get_pixels(tint_roughness_image_ID);
        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                bool is_black = (x & 1) != (y & 1);
                unsigned char* pixel = tint_roughness_pixels + (x + y * size) * 4u;
                unsigned char intensity = is_black ? 1 : 255;
                pixel[0] = pixel[1] = pixel[2] = intensity;
                pixel[3] = is_black ? 6 : 102;
            }
        }

        Materials::Data material_data = Materials::Data::create_dielectric(RGB::white(), 1, 0.04f);
        material_data.tint_roughness_texture_ID = Textures::create2D(tint_roughness_image_ID, MagnificationFilter::None, MinificationFilter::Trilinear);
        material_data.flags = MaterialFlag::ThinWalled;
        MaterialID material_ID = Materials::create("Floor", material_data);

        SceneNode plane_node = SceneNodes::create("Floor", Transform(Vector3f(0.5, -1.0, 0.5), Quaternionf::identity(), float(size)));
        MeshID plane_mesh_ID = MeshCreation::plane(1, { MeshFlag::Position, MeshFlag::Texcoord });
        MeshModels::create(plane_node.get_ID(), plane_mesh_ID, material_ID);
        plane_node.set_parent(root_node);
    }

    { // Cubed spheres
        Materials::Data material_data = Materials::Data::create_metal(silver_tint, 0.04f);
        MaterialID material_ID = Materials::create("Spheres", material_data);

        const float sphere_scale = 1.5f;
        const float sphere_spacing = 2.0f;
        const float x_offset = -sphere_spacing * sphere_model_count / 2.0f;
        const float y_offset = sphere_scale * 0.5f;

        for (int i = 0; i < sphere_model_count; ++i) {
            auto add_mesh_to_scene = [=](MeshID mesh_ID, Transform transform) {
                SceneNode sphere_node = SceneNodes::create("Sphere", transform);
                MeshModels::create(sphere_node.get_ID(), mesh_ID, material_ID);
                sphere_node.set_parent(root_node);
            };

            Vector3f translation = Vector3f(x_offset + i * sphere_spacing, y_offset, 0.0f);
            Transform sphere_transform = Transform(translation, Quaternionf::identity(), sphere_scale);

            MeshID sphere_mesh_ID = MeshCreation::revolved_sphere(1 << (i + 2), 1 << (i + 1));
            add_mesh_to_scene(sphere_mesh_ID, sphere_transform);

            sphere_transform.translation.z += 1.5f * sphere_spacing;
            sphere_mesh_ID = MeshCreation::spherical_cube(1 << i);
            add_mesh_to_scene(sphere_mesh_ID, sphere_transform);
        }
    }

    { // Add a directional light.
        Transform light_transform = Transform(Vector3f(20.0f, 20.0f, -20.0f));
        light_transform.look_at(Vector3f::zero());
        SceneNode light_node = SceneNodes::create("light", light_transform);
        light_node.set_parent(root_node);
        LightSources::create_directional_light(light_node.get_ID(), RGB(3.0f, 2.9f, 2.5f));
    }
}

} // NS Scenes