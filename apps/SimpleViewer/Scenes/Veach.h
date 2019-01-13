// SimpleViewer veach scene.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2015-2017, Bifrost. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_VEACH_SCENE_H_
#define _SIMPLEVIEWER_VEACH_SCENE_H_

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Core/Engine.h>
#include <Bifrost/Input/Keyboard.h>
#include <Bifrost/Math/Intersect.h>
#include <Bifrost/Math/Plane.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/SceneNode.h>
#include <Bifrost/Scene/SceneRoot.h>

using namespace Bifrost;

namespace Scenes {

void create_veach_scene(Core::Engine& engine, Scene::Cameras::UID camera_ID, Scene::SceneRoots::UID scene_ID) {
    using namespace Bifrost::Assets;
    using namespace Bifrost::Math;
    using namespace Bifrost::Scene;

    SceneRoot scene = scene_ID;

    // Remove background lighting
    SceneRoots::set_environment_tint(scene_ID, RGB::black());

    { // Setup camera transform. Look downwards by 22.5 degress
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0.0f, 2.0f, 0.0f);
        cam_transform.rotation = Quaternionf::from_angle_axis(PI<float>() / 8.0f, Vector3f::right());
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Create floor.
        Materials::Data material_data = Materials::Data::create_dielectric(RGB(0.4f, 0.4f, 0.4f), 0.9f, 0.5f);
        Materials::UID material_ID = Materials::create("Floor", material_data);

        SceneNode plane_node = SceneNodes::create("Floor", Transform(Vector3f(0, 0.0f, 0), Quaternionf::identity(), 50));
        Meshes::UID plane_mesh_ID = MeshCreation::plane(1, { MeshFlag::Position, MeshFlag::Texcoord });
        MeshModels::create(plane_node.get_ID(), plane_mesh_ID, material_ID);
        plane_node.set_parent(scene.get_root_node());
    }

    Vector3f block_0_position;
    Plane light_plane;
    { // Initialize by computing the position of the first block and the light source plane.

        // Compute position of nearest block.
        Ray ray = CameraUtils::ray_from_viewport_point(camera_ID, Vector2f(0.5f, 0.2f));
        float distance_to_block_0 = -(ray.origin.y - 0.05f) / ray.direction.y;
        block_0_position = ray.origin + distance_to_block_0 * ray.direction;

        // Compute the plane intersecting all four lights.
        ray.direction.y *= -1.0f;
        Vector3f light_plane_normal = cross(ray.direction, Vector3f::right());
        light_plane = Plane::from_point_direction(block_0_position, light_plane_normal);
    }

    float light_distance;
    Vector3f light_mean_position = Vector3f::zero();
    { // Create lights.

        const int light_count = 4;
        const RGB light_colors[] = { RGB(1, 1, 0.1f), RGB(0.55f, 1.0f, 0.55f), RGB(0.55f, 0.55f, 1.0f), RGB(1.0f, 0.6f, 0.5f) };
        
        // Compute light positions.
        Vector3f light_positions[light_count];
        for (int i = 0; i < light_count; ++i) {
            Ray ray_to_light = CameraUtils::ray_from_viewport_point(camera_ID, Vector2f(0.2f + 0.2f * i, 0.9f));
            float distance_to_light = intersect(ray_to_light, light_plane);
            light_positions[i] = ray_to_light.position_at(distance_to_light);
            light_mean_position += light_positions[i];
        }
        light_distance = light_positions[light_count-1].x - light_positions[0].x;
        light_mean_position /= float(light_count);

        // Create lights
        float interlight_distance = light_distance / (light_count - 1);
        float light_radius = interlight_distance / 4.0f;
        for (int i = light_count - 1; i >= 0; --i) {
            Transform light_transform = Transform(light_positions[i]);
            SceneNode light_node = SceneNodes::create("Light", light_transform);
            LightSources::UID light_ID =  LightSources::create_sphere_light(light_node.get_ID(), light_colors[i] * 10, light_radius);
            light_node.set_parent(scene.get_root_node());
            light_radius *= 0.333f;
        }
    }

    { // Place blocks
        float block_depth = light_distance * 0.23f;
        float block_offset = block_depth * 1.1f;
        float base_roughness = 0.3f;
        int block_count = 4;

        // Create the block mesh. It's size is dependent on the distance between the lights.
        Mesh block_mesh = MeshCreation::cube(1, Vector3f::one(), MeshFlag::Position);
        Vector3f* positions = block_mesh.get_positions();
        for (unsigned int v = 0; v < block_mesh.get_vertex_count(); ++v)
            positions[v] *= Vector3f(light_distance * 1.1f, 0.01f, block_depth);

        // Place the first block.
        Materials::Data material_data = Materials::Data::create_metal(RGB(0.7f, 0.7f, 0.7f), base_roughness);
        Materials::UID material_ID = Materials::create("Block0", material_data);
        SceneNode block_node = SceneNodes::create("Block0", block_0_position);
        MeshModels::create(block_node.get_ID(), block_mesh.get_ID(), material_ID);
        block_node.set_parent(scene.get_root_node());

        Transform previous_transform = block_node.get_global_transform();

        for (int b = 1; b < block_count; ++b) {
            Transform block_transform = previous_transform;
            block_transform.translation += previous_transform.rotation.forward() * block_offset;

            for (int i = 0; i < 2; ++i) {
                // Rotate such that lights reflects of the block.
                Vector3f dir_to_light = light_mean_position - block_transform.translation;
                Vector3f dir_to_camera = Cameras::get_transform(camera_ID).translation - block_transform.translation;
                Vector3f block_normal = normalize(normalize(dir_to_light) + normalize(dir_to_camera));
                block_transform.rotation = Quaternionf::look_in(cross(-block_normal, Vector3f::right()));

                // Adjust block position
                Vector3f prev_block_normal = previous_transform.rotation.up();
                Vector3f inter_plane_normal = normalize(prev_block_normal + block_normal);
                Vector3f inter_plane_forward = cross(-inter_plane_normal, Vector3f::right());
                block_transform.translation = previous_transform.translation + inter_plane_forward * block_offset;
            }

            Vector3f dir_to_light = light_mean_position - block_transform.translation;
            Vector3f dir_to_camera = Cameras::get_transform(camera_ID).translation - block_transform.translation;
            Vector3f block_normal = normalize(normalize(dir_to_light) + normalize(dir_to_camera));
            block_transform.rotation = Quaternionf::look_in(cross(-block_normal, Vector3f::right()));

            float roughness_scale = (block_count - 1 - b) / float(block_count - 1);
            Materials::Data material_data = Materials::Data::create_metal(RGB(0.7f, 0.7f, 0.7f), base_roughness * roughness_scale);
            Materials::UID material_ID = Materials::create("Block", material_data);
            SceneNode block_node = SceneNodes::create("Block", block_transform);
            MeshModels::create(block_node.get_ID(), block_mesh.get_ID(), material_ID);
            block_node.set_parent(scene.get_root_node());

            previous_transform = block_transform;
        }
    }
}

} // NS Scenes

#endif // _SIMPLEVIEWER_VEACH_SCENE_H_