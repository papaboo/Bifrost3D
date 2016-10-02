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
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0.0f, 0.0f, -2.0f);
        cam_transform.rotation = Quaternionf::identity();
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Create sphere.
        Materials::Data material_data = Materials::Data::create_dielectric(RGB(1.0f), 0.0f, 0.25);
        Materials::UID material_ID = Materials::create("White material", material_data);

        Meshes::UID sphere_mesh_ID = MeshCreation::revolved_sphere(1024, 512);

        SceneNode node = SceneNodes::create("Sphere");
        MeshModels::create(node.get_ID(), sphere_mesh_ID, material_ID);
        node.set_parent(root_node);
    }
}

#endif // _SIMPLEVIEWER_SPHERE_SCENE_H_