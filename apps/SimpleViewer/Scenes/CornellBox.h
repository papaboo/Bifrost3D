// SimpleViewer cornell box scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_CORNELL_BOX_SCENE_H_
#define _SIMPLEVIEWER_CORNELL_BOX_SCENE_H_

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>

using namespace Bifrost;

namespace Scenes {

void create_cornell_box(Scene::CameraID camera_ID, Scene::SceneNode root_node) {
    using namespace Bifrost::Assets;
    using namespace Bifrost::Math;
    using namespace Bifrost::Scene;

    Materials::Data white_material_data = Materials::Data::create_dielectric(RGB(0.98f), 1.0f, 0.02f);
    white_material_data.flags = MaterialFlag::ThinWalled;
    MaterialID white_material_ID = Materials::create("White", white_material_data);

    Materials::Data red_material_data = Materials::Data::create_dielectric(RGB(0.98f, 0.02f, 0.02f), 1.0f, 0.02f);
    red_material_data.flags = MaterialFlag::ThinWalled;
    MaterialID red_material_ID = Materials::create("Red", red_material_data);

    Materials::Data green_material_data = Materials::Data::create_dielectric(RGB(0.02f, 0.98f, 0.02f), 1.0f, 0.02f);
    green_material_data.flags = MaterialFlag::ThinWalled;
    MaterialID green_material_ID = Materials::create("Green", green_material_data);

    Materials::Data iron_material_data = Materials::Data::create_metal(iron_tint, 0.4f);
    MaterialID iron_material_ID = Materials::create("Iron", iron_material_data);

    Materials::Data copper_material_data = Materials::Data::create_metal(copper_tint, 0.02f);
    MaterialID copper_material_ID = Materials::create("Copper", copper_material_data);

    { // Set camera position
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 0.0f, -1.5);
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Add light source.
        Vector3f light_position = Vector3f(0.0f, 0.45f, 0.0f);
        Transform light_transform = Transform(light_position);
        SceneNode light_node = SceneNodes::create("Light", light_transform);
        light_node.set_parent(root_node);
        LightSourceID light_ID = LightSources::create_sphere_light(light_node.get_ID(), RGB(2.0f), 0.05f);
    }

    { // Create room.
        MeshID plane_mesh_ID = MeshCreation::plane(1);
        float PI_half = PI<float>() * 0.5f;

        { // Floor
            Transform floor_transform = Transform(Vector3f(0.0f, -0.5f, 0.0f));
            SceneNode floor_node = SceneNodes::create("Floor", floor_transform);
            MeshModels::create(floor_node.get_ID(), plane_mesh_ID, white_material_ID);
            floor_node.set_parent(root_node);
        }

        { // Roof
            Transform roof_transform = Transform(Vector3f(0.0f, 0.5f, 0.0f), Quaternionf::from_angle_axis(Math::PI<float>(), Vector3f::forward()));
            SceneNode roof_node = SceneNodes::create("Roof", roof_transform);
            MeshModels::create(roof_node.get_ID(), plane_mesh_ID, white_material_ID);
            roof_node.set_parent(root_node);
        }

        { // Back
            Transform back_transform = Transform(Vector3f(0.0f, 0.0f, 0.5f), Quaternionf::from_angle_axis(-PI_half, Vector3f::right()));
            SceneNode back_node = SceneNodes::create("Back", back_transform);
            MeshModels::create(back_node.get_ID(), plane_mesh_ID, white_material_ID);
            back_node.set_parent(root_node);
        }

        { // Left
            Transform left_transform = Transform(Vector3f(-0.5f, 0.0f, 0.0f), Quaternionf::from_angle_axis(-PI_half, Vector3f::forward()));
            SceneNode left_node = SceneNodes::create("Left", left_transform);
            MeshModels::create(left_node.get_ID(), plane_mesh_ID, red_material_ID);
            left_node.set_parent(root_node);
        }

        { // Right
            Transform right_transform = Transform(Vector3f(0.5f, 0.0f, 0.0f), Quaternionf::from_angle_axis(PI_half, Vector3f::forward()));
            SceneNode right_node = SceneNodes::create("Right", right_transform);
            MeshModels::create(right_node.get_ID(), plane_mesh_ID, green_material_ID);
            right_node.set_parent(root_node);
        }
    }

    { // Create small box.
        MeshID box_mesh_ID = MeshCreation::cube(1);

        Transform transform = Transform(Vector3f(0.2f, -0.35f, -0.2f),
            Quaternionf::from_angle_axis(PI<float>() / 6.0f, Vector3f::up()),
            0.3f);
        SceneNode node = SceneNodes::create("Small box", transform);
        MeshModels::create(node.get_ID(), box_mesh_ID, iron_material_ID);
        node.set_parent(root_node);
    }

    { // Create big box.
        MeshID box_mesh_ID = MeshCreation::cube(1);

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

} // NS Scenes

#endif // _SIMPLEVIEWER_CORNELL_BOX_SCENE_H_