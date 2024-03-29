// SimpleViewer opacity scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_OPACITY_SCENE_H_
#define _SIMPLEVIEWER_OPACITY_SCENE_H_

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
      // A checker pattern texture would be really nice on the floor.
        unsigned int width = 16, height = 16;
        ImageID image_ID = Images::create2D("Checker", PixelFormat::RGBA32, 2.2f, Vector2ui(width, height));
        unsigned char* pixels = (unsigned char*)Images::get_pixels(image_ID);
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                unsigned char* pixel = pixels + (x + y * width) * 4u;
                unsigned char intensity = ((x & 1) == (y & 1)) ? 2 : 255;
                pixel[0] = pixel[1] = pixel[2] = intensity; 
                pixel[3] = 255;
            }
        }

        Materials::Data material_data = Materials::Data::create_dielectric(RGB(0.02f, 0.27f, 0.33f), 0.3f, 0.02f);
        material_data.tint_roughness_texture_ID = Textures::create2D(image_ID, MagnificationFilter::None, MinificationFilter::None);
        material_data.flags = MaterialFlag::ThinWalled;
        MaterialID material_ID = Materials::create("Floor", material_data);

        SceneNode plane_node = SceneNodes::create("Floor", Transform(Vector3f(0, -0.0005f, 0), Quaternionf::identity(), 10));
        MeshID plane_mesh_ID = MeshCreation::plane(1, { MeshFlag::Position, MeshFlag::Texcoord });
        MeshModels::create(plane_node.get_ID(), plane_mesh_ID, material_ID);
        plane_node.set_parent(root_node);
    }

    { // Sphere light
        Transform light_transform = Transform(Vector3f(0.0f, 0.5f, 0.0f));
        SceneNodeID light_node_ID = SceneNodes::create("Light", light_transform);
        LightSourceID light_ID = LightSources::create_sphere_light(light_node_ID, RGB(50.0f), 0.1f);
        SceneNodes::set_parent(light_node_ID, root_node.get_ID());
    }

    { // Partial coverage box around the light.
        unsigned int width = 17, height = 17;
        ImageID image_ID = Images::create2D("Grid", PixelFormat::Alpha8, 1.0f, Vector2ui(width, height));
        unsigned char* pixels = Images::get_pixels<unsigned char>(image_ID);
        for (unsigned int y = 0; y < height; ++y) {
            for (unsigned int x = 0; x < width; ++x) {
                unsigned char* pixel = pixels + (x + y * width);
                unsigned char intensity = ((x & 1) == 0 || (y & 1) == 0) ? 255 : 0;
                pixel[0] = intensity;
            }
        }

        Materials::Data material_data = Materials::Data::create_dielectric(RGB(0.005f, 0.01f, 0.25f), 0.05f, 0.04f);
        material_data.coverage_texture_ID = Textures::create2D(image_ID, MagnificationFilter::None, MinificationFilter::None);
        material_data.flags = MaterialFlag::Cutout;
        MaterialID material_ID = Materials::create("Plastic", material_data);

        Transform transform = Transform(Vector3f(0.0f, 0.5f, 0.0f));
        SceneNode cube_node = SceneNodes::create("Swizz box", transform);
        MeshID cube_mesh_ID = MeshCreation::cube(1);
        MeshModels::create(cube_node.get_ID(), cube_mesh_ID, material_ID);
        cube_node.set_parent(root_node);
    }

    { // Planes in front of the box.
        Materials::Data transparent_material_data = Materials::Data::create_dielectric(RGB(0.25f), 0.95f, 0.04f);
        transparent_material_data.coverage = 0.75;
        MaterialID transparent_material_ID = Materials::create("Transparent", transparent_material_data);

        MeshID plane_mesh_ID = MeshCreation::plane(1);

        Quaternionf rotation = Quaternionf::from_angle_axis(Math::PI<float>() * 0.5f, Vector3f::right());

        {
            Transform transform = Transform(Vector3f(1.0f, 1.0f, -2.0f), rotation, 2.0f);
            SceneNode plane_node = SceneNodes::create("Plane", transform);
            MeshModels::create(plane_node.get_ID(), plane_mesh_ID, transparent_material_ID);
            plane_node.set_parent(root_node);
        }

        {
            Transform transform = Transform(Vector3f(0.0f, 0.25f, -3.0f), rotation, 1.0f);
            SceneNode plane_node = SceneNodes::create("Plane", transform);
            MeshModels::create(plane_node.get_ID(), plane_mesh_ID, transparent_material_ID);
            plane_node.set_parent(root_node);
        }
    }
}

} // NS Scenes

#endif // _SIMPLEVIEWER_OPACITY_SCENE_H_