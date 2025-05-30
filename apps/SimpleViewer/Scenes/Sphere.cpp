// SimpleViewer sphere scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Scenes/Sphere.h>
#include <Scenes/Utils.h>

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
        SceneNode floor_node = create_checkered_floor(400, 1);
        floor_node.set_global_transform(Transform(Vector3f(0, -1.0f, 0)));
        floor_node.set_parent(root_node);
    }

    { // Cubed spheres
        Material material = Material::create_metal("Spheres", silver_tint, 0.04f);

        const float sphere_scale = 1.5f;
        const float sphere_spacing = 2.0f;
        const float x_offset = -sphere_spacing * sphere_model_count / 2.0f;
        const float y_offset = sphere_scale * 0.5f;

        for (int i = 0; i < sphere_model_count; ++i) {
            auto add_mesh_to_scene = [=](Mesh mesh, Transform transform) {
                SceneNode sphere_node = SceneNode("Sphere", transform);
                MeshModel(sphere_node, mesh, material);
                sphere_node.set_parent(root_node);
            };

            Vector3f translation = Vector3f(x_offset + i * sphere_spacing, y_offset, 0.0f);
            Transform sphere_transform = Transform(translation, Quaternionf::identity(), sphere_scale);

            Mesh sphere_mesh = MeshCreation::revolved_sphere(1 << (i + 2), 1 << (i + 1));
            add_mesh_to_scene(sphere_mesh, sphere_transform);

            sphere_transform.translation.z += 1.5f * sphere_spacing;
            sphere_mesh = MeshCreation::spherical_box(1 << i);
            add_mesh_to_scene(sphere_mesh, sphere_transform);
        }
    }

    { // Add a directional light.
        Transform light_transform = Transform(Vector3f(20.0f, 20.0f, -20.0f));
        light_transform.look_at(Vector3f::zero());
        SceneNode light_node = SceneNode("light", light_transform);
        light_node.set_parent(root_node);
        DirectionalLight(light_node, RGB(3.0f, 2.9f, 2.5f));
    }
}

} // NS Scenes