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
#include <Cogwheel/Core/Engine.h>
#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Scene/Camera.h>
#include <Cogwheel/Scene/SceneNode.h>
#include <Cogwheel/Scene/SceneRoot.h>

using namespace Cogwheel;

void create_sphere_scene(Core::Engine& engine, Scene::Cameras::UID camera_ID, Scene::SceneRoots::UID scene_ID) {
    using namespace Cogwheel::Assets;
    using namespace Cogwheel::Math;
    using namespace Cogwheel::Scene;

    SceneRoot scene = scene_ID;
    if (!Textures::has(scene.get_environment_map()))
        scene.set_environment_tint(RGB(0.5f));

    { // Setup camera transform.
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0.0f, 0.0f, -2.0f);
        cam_transform.rotation = Quaternionf::identity();
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Setup material swapper.
        class MaterialSwapper {
            MeshModel m_model;
            Materials::UID* m_mat_IDs = new Materials::UID[5];
        public:
            MaterialSwapper(Scene::SceneRoots::UID scene_ID) {
                Materials::Data white_mat_data = Materials::Data::create_dielectric(RGB(1.0f), 0.0f, 0.25);
                m_mat_IDs[0] = Materials::create("White", white_mat_data);

                Materials::Data plastic_mat_data = Materials::Data::create_dielectric(RGB(0.005f, 0.01f, 0.25f), 0.05f, 0.5f);
                m_mat_IDs[1] = Materials::create("Plastic", plastic_mat_data);

                Materials::Data rubber_mat_data = Materials::Data::create_dielectric(RGB(0.005f, 0.005f, 0.005f), 0.75f, 0.5f);
                m_mat_IDs[2] = Materials::create("Rubber", rubber_mat_data);

                Materials::Data gold_mat_data = Materials::Data::create_metal(RGB(1.0f, 0.766f, 0.336f), 0.02f, 0.0f);
                m_mat_IDs[3] = Materials::create("Gold", gold_mat_data);

                Materials::Data copper_mat_data = Materials::Data::create_metal(RGB(0.8f, 0.4f, 0.3f), 0.5f, 0.0f);
                m_mat_IDs[4] = Materials::create("Copper", copper_mat_data);

                { // Create sphere.
                    Materials::Data material_data = Materials::Data::create_dielectric(RGB(1.0f), 0.0f, 0.25);
                    Materials::UID material_ID = Materials::create("White material", material_data);

                    Meshes::UID sphere_mesh_ID = MeshCreation::revolved_sphere(1024, 512);

                    SceneNode node = SceneNodes::create("Sphere");
                    m_model = MeshModels::create(node.get_ID(), sphere_mesh_ID, m_mat_IDs[1]);

                    SceneNode root_node = SceneRoots::get_root_node(scene_ID);
                    node.set_parent(root_node);
                }
            }

            void update_material(int id) {
                SceneNode node = m_model.get_scene_node();
                Mesh mesh = m_model.get_mesh();

                MeshModels::destroy(m_model.get_ID());
                m_model = MeshModels::create(node.get_ID(), mesh.get_ID(), m_mat_IDs[id]);
            }

            void update(Core::Engine& engine) {
                using namespace Input;
                const Keyboard* const keyboard = engine.get_keyboard();
                if (keyboard->was_released(Keyboard::Key::Key1))
                    update_material(0);
                else if (keyboard->was_released(Keyboard::Key::Key2))
                    update_material(1);
                else if (keyboard->was_released(Keyboard::Key::Key3))
                    update_material(2);
                else if (keyboard->was_released(Keyboard::Key::Key4))
                    update_material(3);
                else if (keyboard->was_released(Keyboard::Key::Key5))
                    update_material(4);
            }
        };

        MaterialSwapper* mat_swapper = new MaterialSwapper(scene_ID);
        auto mat_swap_update = [](Core::Engine& engine, void* state) {
            ((MaterialSwapper*)state)->update(engine);
        };
        engine.add_mutating_callback(mat_swap_update, mat_swapper);

        // TODO Destroy on shutdown.
    }

}

#endif // _SIMPLEVIEWER_SPHERE_SCENE_H_