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

void create_cornell_box_scene(Scene::Cameras::UID camera_ID, Scene::SceneNode root_node) {
    using namespace Cogwheel::Assets;
    using namespace Cogwheel::Math;
    using namespace Cogwheel::Scene;

    Materials::Data white_material_data = Materials::Data::create_dielectric(RGB(0.98f), 1.0f, 0.25f);
    Materials::UID white_material_ID = Materials::create("White", white_material_data);

    Materials::Data red_material_data = Materials::Data::create_dielectric(RGB(0.98f, 0.02f, 0.02f), 1.0f, 0.25f);
    Materials::UID red_material_ID = Materials::create("Red", red_material_data);

    Materials::Data green_material_data = Materials::Data::create_dielectric(RGB(0.02f, 0.98f, 0.02f), 1.0f, 0.25f);
    Materials::UID green_material_ID = Materials::create("Green", green_material_data);

    Materials::Data iron_material_data = Materials::Data::create_metal(RGB(0.56f, 0.57f, 0.58f), 0.4f, 0.0f);
    Materials::UID iron_material_ID = Materials::create("Iron", iron_material_data);

    Materials::Data copper_material_data = Materials::Data::create_metal(RGB(0.8f, 0.4f, 0.3f), 0.02f, 0.0f);
    Materials::UID copper_material_ID = Materials::create("Copper", copper_material_data);

    { // Set camera position
        SceneNodes::UID cam_node_ID = Cameras::get_node_ID(camera_ID);
        Transform cam_transform = SceneNodes::get_global_transform(cam_node_ID);
        cam_transform.translation = Vector3f(0, 0.0f, -1.5);
        SceneNodes::set_global_transform(cam_node_ID, cam_transform);
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
            MeshModels::create(floor_node.get_ID(), plane_mesh_ID, white_material_ID);
            floor_node.set_parent(root_node);
        }

        { // Roof
            Transform roof_transform = Transform(Vector3f(0.0f, 0.5f, 0.0f));
            SceneNode roof_node = SceneNodes::create("Roof", roof_transform);
            MeshModels::create(roof_node.get_ID(), plane_mesh_ID, white_material_ID);
            roof_node.set_parent(root_node);
        }

        { // Back
            Transform back_transform = Transform(Vector3f(0.0f, 0.0f, 0.5f), Quaternionf::from_angle_axis(PI_half, Vector3f::right()));
            SceneNode back_node = SceneNodes::create("Back", back_transform);
            MeshModels::create(back_node.get_ID(), plane_mesh_ID, white_material_ID);
            back_node.set_parent(root_node);
        }

        { // Left
            Transform left_transform = Transform(Vector3f(-0.5f, 0.0f, 0.0f), Quaternionf::from_angle_axis(PI_half, Vector3f::forward()));
            SceneNode left_node = SceneNodes::create("Left", left_transform);
            MeshModels::create(left_node.get_ID(), plane_mesh_ID, red_material_ID);
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
        MeshModels::create(node.get_ID(), box_mesh_ID, iron_material_ID);
        node.set_parent(root_node);
    }

    { // Create big box.
        Meshes::UID box_mesh_ID = MeshCreation::cube(1);

        Vector3f* positions = Meshes::get_positions(box_mesh_ID);
        for (unsigned int v = 0; v < Meshes::get_vertex_count(box_mesh_ID); ++v) {
            positions[v].y *= 2.0f;
        }

        Transform transform = Transform(Vector3f(-0.2f, -0.2f, 0.2f),
            Quaternionf::from_angle_axis(-PI<float>() / 6.0f, Vector3f::up()),
            0.3f);
        SceneNode node = SceneNodes::create("Big box", transform);
        MeshModels::create(node.get_ID(), box_mesh_ID, copper_material_ID);
        node.set_parent(root_node);
    }
}

#endif // _SIMPLEVIEWER_CORNELL_BOX_SCENE_H_