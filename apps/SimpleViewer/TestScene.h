// SimpleViewer test scene
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _SIMPLEVIEWER_TEST_SCENE_H_
#define _SIMPLEVIEWER_TEST_SCENE_H_

#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshCreation.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Math/Transform.h>
#include <Cogwheel/Scene/SceneNode.h>

using namespace Cogwheel;

class LocalRotator final {
public:
    LocalRotator(Scene::SceneNodes::UID node_ID)
        : m_node_ID(node_ID) {
    }

    void rotate(Core::Engine& engine) {
        Math::Transform transform = Scene::SceneNodes::get_local_transform(m_node_ID);
        transform.rotation = Math::Quaternionf::from_angle_axis(float(engine.get_time().get_total_time()) * 0.1f, Math::Vector3f::up());
        Scene::SceneNodes::set_local_transform(m_node_ID, transform);
    }

    static inline void rotate_callback(Core::Engine& engine, void* state) {
        static_cast<LocalRotator*>(state)->rotate(engine);
    }

private:
    Scene::SceneNodes::UID m_node_ID;
};

Scene::SceneNodes::UID create_test_scene(Core::Engine& engine) {
    using namespace Cogwheel::Assets;
    using namespace Cogwheel::Math;
    using namespace Cogwheel::Scene;

    SceneNode root_node = SceneNodes::create("Root");

    { // Create floor.
        SceneNode plane_node = SceneNodes::create("Floor");
        Meshes::UID plane_mesh_ID = MeshCreation::plane(10);
        MeshModels::UID plane_model_ID = MeshModels::create(plane_node.get_ID(), plane_mesh_ID);
        plane_node.set_parent(root_node);
    }

    { // Create rotating box. TODO Replace by those three cool spinning rings later.
        Transform transform = Transform(Vector3f(1.0f, 0.5f, 0.0f));
        SceneNode cube_node = SceneNodes::create("Rotating cube", transform);
        Meshes::UID cube_mesh_ID = MeshCreation::cube(3);
        MeshModels::UID cube_model_ID = MeshModels::create(cube_node.get_ID(), cube_mesh_ID);
        cube_node.set_parent(root_node);

        LocalRotator* simple_rotator = new LocalRotator(cube_node.get_ID());
        engine.add_mutating_callback(LocalRotator::rotate_callback, simple_rotator);
    }

    { // Destroyable cylinder. Test by destroying the mesh, model and scene node.
        Transform transform = Transform(Vector3f(-1.0f, 0.5f, 0.0f));
        SceneNode cylinder_node = SceneNodes::create("Destroyed Cylinder", transform);
        Meshes::UID cylinder_mesh_ID = MeshCreation::cylinder(4, 16);
        MeshModels::UID cylinder_model_ID = MeshModels::create(cylinder_node.get_ID(), cylinder_mesh_ID);
        cylinder_node.set_parent(root_node);
    }

    { // Sphere for the hell of it. 
        Transform transform = Transform(Vector3f(3.0f, 0.5f, 0.0f));
        SceneNode sphere_node = SceneNodes::create("Sphere", transform);
        Meshes::UID sphere_mesh_ID = MeshCreation::revolved_sphere(32, 16);
        MeshModels::UID sphere_model_ID = MeshModels::create(sphere_node.get_ID(), sphere_mesh_ID);
        sphere_node.set_parent(root_node);
    }

    return root_node.get_ID();
}

#endif _SIMPLEVIEWER_TEST_SCENE_H_