// SimpleViewer material scene.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_MATERIAL_SCENE_H_
#define _SIMPLEVIEWER_MATERIAL_SCENE_H_

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshCreation.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/LightSource.h>
#include <Cogwheel/Scene/SceneNode.h>

using namespace Cogwheel;

void create_material_scene(Scene::Cameras::UID camera_ID, Scene::SceneNode root_node) {
    using namespace Cogwheel::Assets;
    using namespace Cogwheel::Math;
    using namespace Cogwheel::Scene;

    { // Setup camera transform.
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 3.0f, -17.0f);
        cam_transform.look_at(Vector3f(0, 1.0f, 0.0f));
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Add a directional light.
        Math::Transform light_transform = Math::Transform(Math::Vector3f(20.0f, 20.0f, -20.0f));
        light_transform.look_at(Vector3f::zero());
        SceneNode light_node = SceneNodes::create("light", light_transform);
        light_node.set_parent(root_node);
        LightSources::create_directional_light(light_node.get_ID(), Math::RGB(3.0f, 2.9f, 2.5f));
    }

    { // Create floor.
        const int tile_count_pr_side = 41;
        const int vertices_pr_side = tile_count_pr_side + 1;

        { // White tiles.
            Transform transform = Transform(Vector3f(0.0f, -1.0f, 0.0f));
            SceneNode node = SceneNodes::create("White tiles", transform);
            int white_tile_count = tile_count_pr_side * tile_count_pr_side * 3;
            int vertex_count = (tile_count_pr_side + 1) * (tile_count_pr_side + 1);
            Meshes::UID tiles_mesh_ID = Meshes::create("White tiles", white_tile_count, vertex_count, MeshFlags::Position);

            Vector3f* positions = Meshes::get_positions(tiles_mesh_ID);
            for (int y = 0; y < vertices_pr_side; ++y)
                for (int x = 0; x < vertices_pr_side; ++x)
                    positions[x + y * vertices_pr_side] = Vector3f(x - 20.0f, 0.0f, y - 20.0f);

            Vector3ui* indices = Meshes::get_indices(tiles_mesh_ID);
            for (int y = 0; y < tile_count_pr_side; ++y)
                for (int x = 0; x < tile_count_pr_side; ++x) {
                    if ((x & 1) != (y & 1))
                        continue; // Ignore every other tile.

                    unsigned int base_index = x + y * vertices_pr_side;
                    *indices = Vector3ui(base_index, base_index + vertices_pr_side, base_index + 1);
                    ++indices;
                    *indices = Vector3ui(base_index + 1, base_index + vertices_pr_side, base_index + vertices_pr_side + 1);
                    ++indices;
                }

            Meshes::compute_bounds(tiles_mesh_ID);

            Materials::Data white_tile_data = Materials::Data::create_dielectric(RGB(0.5f), 0.4f, 0.25f);
            Materials::UID white_tile_material_ID = Materials::create("White tile", white_tile_data);

            MeshModels::create(node.get_ID(), tiles_mesh_ID, white_tile_material_ID);
            node.set_parent(root_node);
        }

        { // Black tiles.
            Transform transform = Transform(Vector3f(0.0f, -1.0f, 0.0f));
            SceneNode node = SceneNodes::create("Black tiles", transform);
            int white_tile_count = tile_count_pr_side * tile_count_pr_side * 3;
            int vertex_count = (tile_count_pr_side + 1) * (tile_count_pr_side + 1);
            Meshes::UID tiles_mesh_ID = Meshes::create("Black tiles", white_tile_count, vertex_count, MeshFlags::Position);

            Vector3f* positions = Meshes::get_positions(tiles_mesh_ID);
            for (int y = 0; y < vertices_pr_side; ++y)
                for (int x = 0; x < vertices_pr_side; ++x)
                    positions[x + y * vertices_pr_side] = Vector3f(x - 20.0f, 0.0f, y - 20.0f);

            Vector3ui* indices = Meshes::get_indices(tiles_mesh_ID);
            for (int y = 0; y < tile_count_pr_side; ++y)
                for (int x = 0; x < tile_count_pr_side; ++x) {
                    if ((x & 1) == (y & 1))
                        continue; // Ignore every other tile.

                    unsigned int base_index = x + y * vertices_pr_side;
                    *indices = Vector3ui(base_index, base_index + vertices_pr_side, base_index + 1);
                    ++indices;
                    *indices = Vector3ui(base_index + 1, base_index + vertices_pr_side, base_index + vertices_pr_side + 1);
                    ++indices;
                }

            Meshes::compute_bounds(tiles_mesh_ID);

            Materials::Data black_tile_data = Materials::Data::create_dielectric(RGB(0.001f), 0.02f, 0.5f);
            Materials::UID black_tile_material_ID = Materials::create("Black tile", black_tile_data);

            MeshModels::create(node.get_ID(), tiles_mesh_ID, black_tile_material_ID);
            node.set_parent(root_node);
        }
    }

    { // Create material models.
        Materials::Data material0_data = {};
        material0_data.tint = RGB(0.02f, 0.27f, 0.33f);
        material0_data.roughness = 1.0f;
        material0_data.specularity = 0.25f;
        material0_data.metallic = 0.0f;
        material0_data.coverage = 1.0f;

        Materials::Data material1_data = {};
        material1_data.tint = RGB(1.0f, 0.766f, 0.336f);
        material1_data.roughness = 0.02f;
        material1_data.specularity = 0.25f;
        material1_data.metallic = 1.0f;
        material1_data.coverage = 1.0f;

        Meshes::UID cube_mesh_ID = MeshCreation::cube(1);
        Transform cube_transform = Transform(Vector3f(0.0f, -0.25f, 0.0f), Quaternionf::identity(), 1.5f);
        Meshes::UID sphere_mesh_ID = MeshCreation::revolved_sphere(32, 16);
        Transform sphere_transform = Transform(Vector3f(0.0f, 1.0f, 0.0f), Quaternionf::identity(), 1.5f);

        // Mesh combine models.
        Meshes::UID mesh_ID = MeshUtils::combine("MaterialMesh", cube_mesh_ID, cube_transform, sphere_mesh_ID, sphere_transform);
        Meshes::destroy(cube_mesh_ID);
        Meshes::destroy(sphere_mesh_ID);

        for (int m = 0; m < 9; ++m) {
            float lerp_t = m / 8.0f;
            Materials::Data material_data;
            material_data.tint = lerp(material0_data.tint, material1_data.tint, lerp_t);
            material_data.roughness = lerp(material0_data.roughness, material1_data.roughness, lerp_t);
            material_data.specularity = lerp(material0_data.specularity, material1_data.specularity, lerp_t);
            material_data.metallic = lerp(material0_data.metallic, material1_data.metallic, lerp_t);
            material_data.coverage = lerp(material0_data.coverage, material1_data.coverage, lerp_t);
            Materials::UID material_ID = Materials::create("Lerped material", material_data);

            Transform transform = Transform(Vector3f(float(m * 2 - 8), 0.0, 0.0f));
            SceneNode node = SceneNodes::create("Model", transform);
            MeshModels::create(node.get_ID(), mesh_ID, material_ID);
            node.set_parent(root_node);
        }
    }
}

#endif // _SIMPLEVIEWER_MATERIAL_SCENE_H_