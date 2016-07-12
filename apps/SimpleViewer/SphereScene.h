// SimpleViewer sphere scene.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_SPHERE_SCENE_H_
#define _SIMPLEVIEWER_SPHERE_SCENE_H_

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshCreation.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/SceneNode.h>

using namespace Cogwheel;

void create_sphere_scene(Scene::Cameras::UID camera_ID, Scene::SceneNode root_node) {
    using namespace Cogwheel::Assets;
    using namespace Cogwheel::Math;
    using namespace Cogwheel::Scene;

    { // Setup camera transform.
        SceneNodes::UID cam_node_ID = Cameras::get_node_ID(camera_ID);
        Transform cam_transform = SceneNodes::get_global_transform(cam_node_ID);
        cam_transform.translation = Vector3f(0.0f, 0.0f, 4.0f);
        cam_transform.look_at(Vector3f(0.0f, 0.0f, 0.0f));
        SceneNodes::set_global_transform(cam_node_ID, cam_transform);
    }

    { // Create sphere.
        Materials::Data material_data;
        material_data.base_tint = RGB(1.0f, 0.766f, 0.336f);
        material_data.base_roughness = 1.0f;
        material_data.specularity = 0.25f;
        material_data.metallic = 1.0f;
        Materials::UID material_ID = Materials::create("White material", material_data);

        Meshes::UID sphere_mesh_ID = MeshCreation::revolved_sphere(128, 64);

        SceneNode node = SceneNodes::create("Sphere");
        MeshModels::create(node.get_ID(), sphere_mesh_ID, material_ID);
        node.set_parent(root_node);
    }
}

#endif // _SIMPLEVIEWER_SPHERE_SCENE_H_