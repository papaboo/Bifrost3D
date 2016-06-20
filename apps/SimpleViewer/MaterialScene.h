// SimpleViewer cornell box scene.
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

Scene::SceneNodes::UID create_material_scene(Scene::Cameras::UID camera_ID) {
    using namespace Cogwheel::Assets;
    using namespace Cogwheel::Math;
    using namespace Cogwheel::Scene;

    SceneNode root_node = SceneNodes::create("Root");

    { // Setup camera transform.
        SceneNodes::UID cam_node_ID = Cameras::get_node_ID(camera_ID);
        Transform cam_transform = SceneNodes::get_global_transform(cam_node_ID);
        cam_transform.translation = Vector3f(0, 3.0f, -17.0f);
        cam_transform.look_at(Vector3f(0, 1.0f, 0.0f));
        SceneNodes::set_global_transform(cam_node_ID, cam_transform);
    }

    { // Add a directional light.
        Math::Transform light_transform = Math::Transform(Math::Vector3f(20.0f, 20.0f, -20.0f));
        light_transform.look_at(Vector3f::zero());
        SceneNode light_node = SceneNodes::create("light", light_transform);
        light_node.set_parent(root_node);
        LightSources::create_directional_light(light_node.get_ID(), Math::RGB(3.0f, 2.9f, 2.5f));
    }

    { // Create floor.
        Materials::Data black_tile_data;
        black_tile_data.base_tint = RGB(0.001f, 0.001f, 0.001f);
        black_tile_data.base_roughness = 0.02f;
        black_tile_data.specularity = 0.5f;
        black_tile_data.metallic = 0.0f;
        Materials::UID black_tile_material_ID = Materials::create("Black tile", black_tile_data);

        Materials::Data white_tile_data;
        white_tile_data.base_tint = RGB(0.5f, 0.5f, 0.5f);
        white_tile_data.base_roughness = 0.4f;
        white_tile_data.specularity = 0.25f;
        white_tile_data.metallic = 0.0f;
        Materials::UID white_tile_material_ID = Materials::create("white tile", white_tile_data);

        Meshes::UID plane_mesh_ID = MeshCreation::plane(1);
        
        // TODO Meshcombine tiles.
        for (int x = 0; x < 41; ++x) {
            for (int y = 0; y < 41; ++y) {
                Transform tile_transform = Transform(Vector3f(x - 20.5f, -1.0f, y - 20.5f));
                SceneNode tile_node = SceneNodes::create("Tile", tile_transform);
                Materials::UID tile_material_ID = (x & 1) == (y & 1) ? white_tile_material_ID : black_tile_material_ID;
                MeshModels::create(tile_node.get_ID(), plane_mesh_ID, tile_material_ID);
                tile_node.set_parent(root_node);
            }
        }
    }

    { // Create material models.
        Materials::Data material0_data;
        material0_data.base_tint = RGB(0.02f, 0.27f, 0.33f);
        material0_data.base_roughness = 1.0f;
        material0_data.specularity = 0.25f;
        material0_data.metallic = 0.0f;

        Materials::Data material1_data;
        material1_data.base_tint = RGB(1.0f, 0.766f, 0.336f);
        material1_data.base_roughness = 0.02f;
        material1_data.specularity = 0.25f;
        material1_data.metallic = 1.0f;

        Meshes::UID cube_mesh_ID = MeshCreation::cube(1);
        Meshes::UID sphere_mesh_ID = MeshCreation::revolved_sphere(32, 16);

        for (int m = 0; m < 9; ++m) {
            float lerp_t = m / 8.0f;
            Materials::Data material_data;
            material_data.base_tint = lerp(material0_data.base_tint, material1_data.base_tint, lerp_t);
            material_data.base_roughness = lerp(material0_data.base_roughness, material1_data.base_roughness, lerp_t);;
            material_data.specularity = lerp(material0_data.specularity, material1_data.specularity, lerp_t);;
            material_data.metallic = lerp(material0_data.metallic, material1_data.metallic, lerp_t);;
            Materials::UID material_ID = Materials::create("Lerped material", material_data);

            Transform cube_transform = Transform(Vector3f(float(m * 2 - 8), -0.25f, 0.0f), Quaternionf::identity(), 1.5f);
            SceneNode cube_node = SceneNodes::create("Cube", cube_transform);
            MeshModels::create(cube_node.get_ID(), cube_mesh_ID, material_ID);
            cube_node.set_parent(root_node);

            Transform sphere_transform = Transform(Vector3f(float(m * 2 - 8), 1.0f, 0.0f), Quaternionf::identity(), 1.5f);
            SceneNode sphere_node = SceneNodes::create("Sphere", sphere_transform);
            MeshModels::create(sphere_node.get_ID(), sphere_mesh_ID, material_ID);
            sphere_node.set_parent(root_node);
        }
    }

    return root_node.get_ID();
}

#endif // _SIMPLEVIEWER_CORNELL_BOX_SCENE_H_