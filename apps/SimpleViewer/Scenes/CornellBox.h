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
    Material white_material = Materials::create("White", white_material_data);

    Materials::Data red_material_data = Materials::Data::create_dielectric(RGB(0.98f, 0.02f, 0.02f), 1.0f, 0.02f);
    red_material_data.flags = MaterialFlag::ThinWalled;
    Material red_material = Materials::create("Red", red_material_data);

    Materials::Data green_material_data = Materials::Data::create_dielectric(RGB(0.02f, 0.98f, 0.02f), 1.0f, 0.02f);
    green_material_data.flags = MaterialFlag::ThinWalled;
    Material green_material = Materials::create("Green", green_material_data);

    Material iron_material = Material::create_metal("Iron", iron_tint, 0.4f);

    Material copper_material = Material::create_metal("Copper", copper_tint, 0.02f);

    { // Set camera position
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 0.0f, -1.5);
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Add light source.
        Vector3f light_position = Vector3f(0.0f, 0.45f, 0.0f);
        Transform light_transform = Transform(light_position);
        SceneNode light_node = SceneNode("Light", light_transform);
        light_node.set_parent(root_node);
        SphereLight(light_node, RGB(2.0f), 0.05f);
    }

    { // Create room.
        Mesh plane_mesh = MeshCreation::plane(1);
        float PI_half = PI<float>() * 0.5f;

        { // Floor
            Transform floor_transform = Transform(Vector3f(0.0f, -0.5f, 0.0f));
            SceneNode floor_node = SceneNode("Floor", floor_transform);
            MeshModel(floor_node, plane_mesh, white_material);
            floor_node.set_parent(root_node);
        }

        { // Roof
            Transform roof_transform = Transform(Vector3f(0.0f, 0.5f, 0.0f), Quaternionf::from_angle_axis(Math::PI<float>(), Vector3f::forward()));
            SceneNode roof_node = SceneNode("Roof", roof_transform);
            MeshModel(roof_node, plane_mesh, white_material);
            roof_node.set_parent(root_node);
        }

        { // Back
            Transform back_transform = Transform(Vector3f(0.0f, 0.0f, 0.5f), Quaternionf::from_angle_axis(-PI_half, Vector3f::right()));
            SceneNode back_node = SceneNode("Back", back_transform);
            MeshModel(back_node, plane_mesh, white_material);
            back_node.set_parent(root_node);
        }

        { // Left
            Transform left_transform = Transform(Vector3f(-0.5f, 0.0f, 0.0f), Quaternionf::from_angle_axis(-PI_half, Vector3f::forward()));
            SceneNode left_node = SceneNode("Left", left_transform);
            MeshModel(left_node, plane_mesh, red_material);
            left_node.set_parent(root_node);
        }

        { // Right
            Transform right_transform = Transform(Vector3f(0.5f, 0.0f, 0.0f), Quaternionf::from_angle_axis(PI_half, Vector3f::forward()));
            SceneNode right_node = SceneNode("Right", right_transform);
            MeshModel(right_node, plane_mesh, green_material);
            right_node.set_parent(root_node);
        }
    }

    { // Create small box.
        Mesh box_mesh = MeshCreation::box(1);

        Transform transform = Transform(Vector3f(0.2f, -0.35f, -0.2f),
            Quaternionf::from_angle_axis(PI<float>() / 6.0f, Vector3f::up()),
            0.3f);
        SceneNode node = SceneNode("Small box", transform);
        MeshModel(node, box_mesh, iron_material);
        node.set_parent(root_node);
    }

    { // Create big box.
        Mesh box_mesh = MeshCreation::box(1);

        for (Vector3f& position : box_mesh.get_position_iterable())
            position.y *= 2.0f;

        Transform transform = Transform(Vector3f(-0.2f, -0.2f, 0.2f),
            Quaternionf::from_angle_axis(-PI<float>() / 6.0f, Vector3f::up()),
            0.3f);
        SceneNode node = SceneNode("Big box", transform);
        MeshModel(node, box_mesh, copper_material);
        node.set_parent(root_node);
    }
}

} // NS Scenes

#endif // _SIMPLEVIEWER_CORNELL_BOX_SCENE_H_