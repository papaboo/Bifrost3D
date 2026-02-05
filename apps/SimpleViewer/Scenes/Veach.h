// SimpleViewer veach scene.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
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

void create_veach_scene(Core::Engine& engine, Scene::CameraID camera_ID, Scene::SceneRoot scene, bool use_polygonal_lights) {
    using namespace Bifrost::Assets;
    using namespace Bifrost::Math;
    using namespace Bifrost::Scene;

    // Black background if no environment map is specified.
    if (!scene.get_environment_map().exists())
        scene.set_environment_tint(RGB(0.02f));

    { // Setup camera transform. Look downwards by 22.5 degress
        Vector3f camera_translation = Vector3f(0.0f, 2.0f, 0.0f);
        Quaternionf camera_rotation = Quaternionf::from_angle_axis(PI<float>() / 8.0f, Vector3f::right());
        Cameras::set_transform(camera_ID, Transform(camera_translation, camera_rotation));
    }

    { // Create floor.
        Materials::Data material_data = Materials::Data::create_dielectric(RGB(0.4f, 0.4f, 0.4f), 0.9f, 0.04f);
        material_data.flags = MaterialFlag::ThinWalled;
        Material material = Material("Floor", material_data);

        SceneNode plane_node = SceneNode("Floor", Transform(Vector3f(0, 0.0f, 0), Quaternionf::identity(), 50));
        Mesh plane_mesh = MeshCreation::plane(1, { MeshFlag::Position, MeshFlag::Texcoord });
        MeshModel(plane_node, plane_mesh, material);
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
        const RGB light_colors[light_count] = { RGB(1, 1, 0.1f), RGB(0.55f, 1.0f, 0.55f), RGB(0.55f, 0.55f, 1.0f), RGB(1.0f, 0.6f, 0.5f) };

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

        if (!use_polygonal_lights) {
            float light_radius = interlight_distance / 4.0f;
            for (int i = light_count - 1; i >= 0; --i) {
                Transform light_transform = Transform(light_positions[i]);
                SceneNode light_node = SceneNode("Light" + std::to_string(i), light_transform);
                SphereLight(light_node, light_colors[i] * 10, light_radius);
                light_node.set_parent(scene.get_root_node());
                light_radius *= 0.333f;
            }
        } else {

            Material emissive_material = Material::create_emissive("Emissive", RGB::white());
            emissive_material.set_flags(MaterialFlag::ThinWalled);
            Quaternionf quad_rotation = Quaternionf::from_angle_axis(-0.5f * PI<float>(), Vector3f::right());

            { // Small quad light
                Mesh small_quad_mesh = MeshCreation::plane(1, { MeshFlag::Position });
                MeshUtils::scale_mesh(small_quad_mesh, interlight_distance * 0.2f);
                Transform small_quad_light_transform = Transform(light_positions[0], quad_rotation);
                SceneNode small_quad_light_node = SceneNode("Small quad light", small_quad_light_transform);
                MeshModel(small_quad_light_node, small_quad_mesh, emissive_material);
                small_quad_light_node.set_parent(scene.get_root_node());
            }

            { // Large quad light
                Mesh large_quad_mesh = MeshCreation::plane(1, { MeshFlag::Position, MeshFlag::Emissive });
                RGB* emission = large_quad_mesh.get_emission();
                for (int i = 0; i < 4; ++i)
                    emission[i] = light_colors[i];

                MeshUtils::scale_mesh(large_quad_mesh, interlight_distance * 0.8f);
                Transform large_quad_light_transform = Transform(light_positions[1], quad_rotation);
                SceneNode large_quad_light_node = SceneNode("Large quad light", large_quad_light_transform);
                MeshModel(large_quad_light_node, large_quad_mesh, emissive_material);
                large_quad_light_node.set_parent(scene.get_root_node());
            }

            { // Line light
                Mesh line_mesh = MeshCreation::plane(1, { MeshFlag::Position });
                MeshUtils::scale_mesh(line_mesh, Vector3f(interlight_distance * 0.1f, 1.0f, interlight_distance * 0.8f));
                Transform line_light_transform = Transform(light_positions[2], quad_rotation);
                SceneNode line_light_node = SceneNode("Line light", line_light_transform);
                MeshModel(line_light_node, line_mesh, emissive_material);
                line_light_node.set_parent(scene.get_root_node());
            }

            { // Star light
                //       v0
                //       /\
                //      /  \
                // v8--v9--v1--v2
                //  \          /
                //   \v7 v5 v3/
                //   /   /\   \
                //  / /      \ \
                //  v6        v4

                Mesh star_mesh = Mesh("Star", 4, 10, { MeshFlag::Position });

                // Create vertex positions by rotating the star in increments of 5.
                // The outer vertices are called point and the innermost are named shoulders below.
                float point_distance = interlight_distance * 0.4f;
                float shoulder_distance = interlight_distance * 0.152786f; // Distance scale constant computed by intersecting the lines [V0, V4] and [V8, V2]
                Vector3f* positions = star_mesh.get_positions();
                for (int i = 0; i < 5; ++i) {
                    float point_angle = 2 * PI<float>() * i / 5.0f;
                    positions[2 * i] = point_distance * Vector3f(cosf(point_angle), 0.0f, sinf(point_angle));

                    float shoulder_angle = 2 * PI<float>() * (i + 0.5f) / 5.0f;
                    positions[2 * i + 1] = shoulder_distance * Vector3f(cosf(shoulder_angle), 0.0f, sinf(shoulder_angle));
                }
                star_mesh.compute_bounds();

                // Create triangles
                Vector3ui* triangles = star_mesh.get_primitives();
                triangles[0] = Vector3ui(0, 9, 1);
                triangles[1] = Vector3ui(8, 5, 2);
                triangles[2] = Vector3ui(3, 5, 4);
                triangles[3] = Vector3ui(5, 7, 6);

                Material star_material = Material::create_emissive("Emissive star", RGB(1, 0.5f, 0));
                star_material.set_flags(MaterialFlag::ThinWalled);
                Transform star_light_transform = Transform(light_positions[3], quad_rotation);
                SceneNode star_light_node = SceneNode("Star light", star_light_transform);
                MeshModel(star_light_node, star_mesh, star_material);
                star_light_node.set_parent(scene.get_root_node());
            }
        }
    }

    { // Place blocks
        float block_depth = light_distance * 0.23f;
        float block_offset = block_depth * 1.1f;
        float base_roughness = 0.3f;
        int block_count = 4;

        // Create the block mesh. It's size is dependent on the distance between the lights.
        Mesh block_mesh = MeshCreation::box(1, Vector3f::one(), MeshFlag::Position);
        Vector3f* positions = block_mesh.get_positions();
        for (unsigned int v = 0; v < block_mesh.get_vertex_count(); ++v)
            positions[v] *= Vector3f(light_distance * 1.1f, 0.01f, block_depth);

        // Place the first block.
        Material material = Material::create_metal("Block0", RGB(0.7f), base_roughness);
        SceneNode block_node = SceneNode("Block0", block_0_position);
        MeshModel(block_node, block_mesh, material);
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
            Material material = Material::create_metal("Block" + std::to_string(b), RGB(0.7f), base_roughness * roughness_scale);
            SceneNode block_node = SceneNode("Block" + std::to_string(b), block_transform);
            MeshModel(block_node, block_mesh, material);
            block_node.set_parent(scene.get_root_node());

            previous_transform = block_transform;
        }
    }
}

} // NS Scenes

#endif // _SIMPLEVIEWER_VEACH_SCENE_H_