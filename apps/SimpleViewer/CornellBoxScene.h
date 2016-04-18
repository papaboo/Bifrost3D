// SimpleViewer cornell box scene.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_CORNELL_BOX_SCENE_H_
#define _SIMPLEVIEWER_CORNELL_BOX_SCENE_H_

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshCreation.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/LightSource.h>
#include <Cogwheel/Scene/SceneNode.h>

using namespace Cogwheel;


Scene::SceneNodes::UID create_cornell_box_scene() {
    using namespace Cogwheel::Assets;
    using namespace Cogwheel::Math;
    using namespace Cogwheel::Scene;

    SceneNode root_node = SceneNodes::create("Root");

    Materials::Data white_material_data;
    white_material_data.base_tint = RGB(0.98f);
    white_material_data.base_roughness = 1.0f;
    white_material_data.specularity = 0.25f;
    white_material_data.metallic = 0.0f;
    Materials::UID white_material_ID = Materials::create("White", white_material_data);

    Materials::Data red_material_data;
    red_material_data.base_tint = RGB(0.98f, 0.02f, 0.02f);
    red_material_data.base_roughness = 1.0f;
    red_material_data.specularity = 0.25f;
    red_material_data.metallic = 0.0f;
    Materials::UID red_material_ID = Materials::create("Red", red_material_data);

    Materials::Data green_material_data;
    green_material_data.base_tint = RGB(0.02f, 0.98f, 0.02f);
    green_material_data.base_roughness = 1.0f;
    green_material_data.specularity = 0.25f;
    green_material_data.metallic = 0.0f;
    Materials::UID green_material_ID = Materials::create("Green", green_material_data);

    Materials::Data iron_material_data;
    iron_material_data.base_tint = RGB(0.56f, 0.57f, 0.58f);
    iron_material_data.base_roughness = 0.4f;
    iron_material_data.specularity = 0.0f;
    iron_material_data.metallic = 1.0f;
    Materials::UID iron_material_ID = Materials::create("Iron", iron_material_data);

    Materials::Data copper_material_data;
    copper_material_data.base_tint = RGB(0.8f, 0.4f, 0.3f);
    copper_material_data.base_roughness = 0.02f;
    copper_material_data.specularity = 0.0f;
    copper_material_data.metallic = 1.0f;
    Materials::UID copper_material_ID = Materials::create("Copper", copper_material_data);

    { // Add camera
        Cameras::allocate(1u);
        SceneNodes::UID cam_node_ID = SceneNodes::create("Cam");

        Transform cam_transform = SceneNodes::get_global_transform(cam_node_ID);
        cam_transform.translation = Vector3f(0, 0.0f, -1.5);
        SceneNodes::set_global_transform(cam_node_ID, cam_transform);

        Matrix4x4f perspective_matrix, inverse_perspective_matrix;
        CameraUtils::compute_perspective_projection(0.1f, 100.0f, PI<float>() / 4.0f, 8.0f / 6.0f,
            perspective_matrix, inverse_perspective_matrix);

        Cameras::UID cam_ID = Cameras::create(cam_node_ID, perspective_matrix, inverse_perspective_matrix);
    }

    { // Add light source.
        Vector3f light_position = Vector3f(0.0f, 0.45f, 0.0f);
        Transform light_transform = Transform(light_position);
        SceneNode light_node = SceneNodes::create("Light", light_transform);
        light_node.set_parent(root_node);
        LightSources::UID light_ID = LightSources::create_sphere_light(light_node.get_ID(), RGB(2.0f), 0.05f);
    }

    { // Create room.
        Meshes::UID plane_mesh_ID = MeshCreation::plane(1);
        float PI_half = PI<float>() * 0.5f;

        { // Floor
            Transform floor_transform = Transform(Vector3f(0.0f, -0.5f, 0.0f));
            SceneNode floor_node = SceneNodes::create("Floor", floor_transform);
            MeshModels::UID floor_model_ID = MeshModels::create(floor_node.get_ID(), plane_mesh_ID, white_material_ID);
            floor_node.set_parent(root_node);
        }

        { // Roof
            Transform roof_transform = Transform(Vector3f(0.0f, 0.5f, 0.0f));
            SceneNode roof_node = SceneNodes::create("Roof", roof_transform);
            MeshModels::UID roof_model_ID = MeshModels::create(roof_node.get_ID(), plane_mesh_ID, white_material_ID);
            roof_node.set_parent(root_node);
        }

        { // Back
            Transform back_transform = Transform(Vector3f(0.0f, 0.0f, 0.5f), Quaternionf::from_angle_axis(PI_half, Vector3f::right()));
            SceneNode back_node = SceneNodes::create("Back", back_transform);
            MeshModels::UID back_model_ID = MeshModels::create(back_node.get_ID(), plane_mesh_ID, white_material_ID);
            back_node.set_parent(root_node);
        }

        { // Left
            Transform left_transform = Transform(Vector3f(-0.5f, 0.0f, 0.0f), Quaternionf::from_angle_axis(PI_half, Vector3f::forward()));
            SceneNode left_node = SceneNodes::create("Left", left_transform);
            MeshModels::UID left_model_ID = MeshModels::create(left_node.get_ID(), plane_mesh_ID, red_material_ID);
            left_node.set_parent(root_node);
        }

        { // Right
            Transform right_transform = Transform(Vector3f(0.5f, 0.0f, 0.0f), Quaternionf::from_angle_axis(-PI_half, Vector3f::forward()));
            SceneNode right_node = SceneNodes::create("Right", right_transform);
            MeshModels::create(right_node.get_ID(), plane_mesh_ID, green_material_ID);
            right_node.set_parent(root_node);
        }
    }

    { // Create small box.
        Meshes::UID box_mesh_ID = MeshCreation::cube(1);

        Transform transform = Transform(Vector3f(0.2f, -0.35f, -0.2f), 
                                        Quaternionf::from_angle_axis(PI<float>() / 6.0f, Vector3f::up()),
                                        0.3f);
        SceneNode node = SceneNodes::create("Small box", transform);
        MeshModels::UID model_ID = MeshModels::create(node.get_ID(), box_mesh_ID, iron_material_ID);
        node.set_parent(root_node);
    }

    { // Create big box.
        Meshes::UID box_mesh_ID = MeshCreation::cube(1);

        Mesh& mesh = Meshes::get_mesh(box_mesh_ID);
        for (unsigned int v = 0; v < mesh.vertex_count; ++v) {
            mesh.positions[v].y *= 2.0f;
        }

        Transform transform = Transform(Vector3f(-0.2f, -0.2f, 0.2f),
            Quaternionf::from_angle_axis(-PI<float>() / 6.0f, Vector3f::up()),
            0.3f);
        SceneNode node = SceneNodes::create("Big box", transform);
        MeshModels::UID model_ID = MeshModels::create(node.get_ID(), box_mesh_ID, copper_material_ID);
        node.set_parent(root_node);
    }

    return root_node.get_ID();
}

#endif // _SIMPLEVIEWER_CORNELL_BOX_SCENE_H_