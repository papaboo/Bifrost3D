// SimpleViewer opacity scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_OPACITY_SCENE_H_
#define _SIMPLEVIEWER_OPACITY_SCENE_H_

#include <Scenes/Utils.h>

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Core/Engine.h>
#include <Bifrost/Input/Keyboard.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/SceneNode.h>
#include <Bifrost/Scene/SceneRoot.h>

using namespace Bifrost;

namespace Scenes {

void create_opacity_scene(Core::Engine& engine, Scene::CameraID camera_ID, Scene::SceneNode root_node) {
    using namespace Bifrost::Assets;
    using namespace Bifrost::Math;
    using namespace Bifrost::Scene;

    { // Setup camera transform.
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 1, -6);
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Create floor.
        SceneNode floor_node = create_checkered_floor(400, 1);
        floor_node.set_global_transform(Transform(Vector3f(0, -0.0005f, 0)));
        floor_node.set_parent(root_node);

        MeshModel floor_model = MeshModels::get_attached_mesh_model(floor_node.get_ID());
        Material floor_material = floor_model.get_material();
        floor_material.set_tint(RGB(0.02f, 0.27f, 0.33f));
        floor_material.set_roughness(0.3f);
    }

    { // Sphere light
        Transform light_transform = Transform(Vector3f(0.0f, 0.5f, 0.0f));
        SceneNode light_node = SceneNode("Light", light_transform);
        light_node.set_parent(root_node);
        SphereLight(light_node, RGB(50.0f), 0.1f);
    }

    { // Partial coverage box around the light.
        unsigned int width = 17, height = 17;
        Image image = Image::create2D("Grid", PixelFormat::Alpha8, false, Vector2ui(width, height));
        unsigned char* pixels = image.get_pixels<unsigned char>();
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                unsigned char* pixel = pixels + (x + y * width);
                unsigned char intensity = ((x & 1) == 0 || (y & 1) == 0) ? 255 : 0;
                pixel[0] = intensity;
            }
        }

        Materials::Data material_data = Materials::Data::create_dielectric(RGB(0.005f, 0.01f, 0.25f), 0.05f, 0.04f);
        material_data.coverage_texture_ID = Texture::create2D(image, MagnificationFilter::None, MinificationFilter::None).get_ID();
        material_data.flags = MaterialFlag::Cutout;
        Material material = Material("Plastic", material_data);

        Transform transform = Transform(Vector3f(0.0f, 0.5f, 0.0f));
        SceneNode box_node = SceneNode("Swizz box", transform);
        Mesh box_mesh = MeshCreation::box(1);
        MeshModel(box_node, box_mesh, material);
        box_node.set_parent(root_node);
    }

    { // Planes in front of the box.
        Materials::Data transparent_material_data = Materials::Data::create_dielectric(RGB(0.25f), 0.95f, 0.04f);
        transparent_material_data.coverage = 0.75;
        transparent_material_data.flags = MaterialFlag::ThinWalled;
        Material transparent_material = Material("Transparent", transparent_material_data);

        Mesh plane_mesh = MeshCreation::plane(1);

        Quaternionf rotation = Quaternionf::from_angle_axis(Math::PI<float>() * 0.5f, Vector3f::right());

        {
            Transform transform = Transform(Vector3f(1.0f, 1.0f, -2.0f), rotation, 2.0f);
            SceneNode plane_node = SceneNode("Plane", transform);
            MeshModel(plane_node, plane_mesh, transparent_material);
            plane_node.set_parent(root_node);
        }

        {
            Transform transform = Transform(Vector3f(0.0f, 0.25f, -3.0f), rotation, 1.0f);
            SceneNode plane_node = SceneNode("Plane", transform);
            MeshModel(plane_node, plane_mesh, transparent_material);
            plane_node.set_parent(root_node);
        }
    }
}

} // NS Scenes

#endif // _SIMPLEVIEWER_OPACITY_SCENE_H_