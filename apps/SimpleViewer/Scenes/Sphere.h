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

#include <ImGui/ImGuiAdaptor.h>

using namespace Cogwheel;

namespace Scenes {

void create_sphere_scene(Core::Engine& engine, Scene::Cameras::UID camera_ID, Scene::SceneRoots::UID scene_ID, ImGui::ImGuiAdaptor* imgui) {
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

    { // Create sphere.
        auto plastic_mat_data = Materials::Data::create_dielectric(RGB(0.005f, 0.01f, 0.25f), 0.05f, 0.5f);
        Materials::UID material_ID = Materials::create("Material", plastic_mat_data);

        Meshes::UID sphere_mesh_ID = MeshCreation::revolved_sphere(1024, 512);

        SceneNode node = SceneNodes::create("Sphere");
        MeshModel model = MeshModels::create(node.get_ID(), sphere_mesh_ID, material_ID);

        SceneNode root_node = SceneRoots::get_root_node(scene_ID);
        node.set_parent(root_node);
    }

    { // Setup sphere and material swapper.
        static auto swap_material = [](Core::Engine& engine) {
            using namespace Input;

            static auto update_material = [](Materials::Data data) {
                Material m_material = *Materials::get_iterable().begin();
                m_material.set_tint(data.tint);
                m_material.set_roughness(data.roughness);
                m_material.set_metallic(data.metallic);
                m_material.set_specularity(data.specularity);
            };

            const Keyboard* const keyboard = engine.get_keyboard();
            if (keyboard->was_released(Keyboard::Key::Key1)) {
                auto plastic_mat_data = Materials::Data::create_dielectric(RGB(0.005f, 0.01f, 0.25f), 0.05f, 0.5f);
                update_material(plastic_mat_data);
            } else if (keyboard->was_released(Keyboard::Key::Key2)) {
                auto white_mat_data = Materials::Data::create_dielectric(RGB::white(), 0.0f, 0.25);
                update_material(white_mat_data);
            } else if (keyboard->was_released(Keyboard::Key::Key3)) {
                auto rubber_mat_data = Materials::Data::create_dielectric(RGB::white(), 0.95f, 0.5f);
                update_material(rubber_mat_data);
            } else if (keyboard->was_released(Keyboard::Key::Key4)) {
                auto gold_mat_data = Materials::Data::create_metal(RGB(1.0f, 0.766f, 0.336f), 0.02f, 0.0f);
                update_material(gold_mat_data);
            } else if (keyboard->was_released(Keyboard::Key::Key5)) {
                auto copper_mat_data = Materials::Data::create_metal(RGB(0.8f, 0.4f, 0.3f), 0.5f, 0.0f);
                update_material(copper_mat_data);
            }
        };

        engine.add_mutating_callback([=, &engine]() { swap_material(engine); });
    }

    { // Setup GUI.
        class MaterialGUI final : public ImGui::IImGuiFrame {
        public:
            MaterialGUI(Material material) : m_material(material) { }

            void layout_frame() {
                ImGui::Begin("Material");

                RGB tint = m_material.get_tint();
                if (ImGui::ColorEdit3("Tint", &tint.r))
                    m_material.set_tint(tint);

                float roughness = m_material.get_roughness();
                if (ImGui::SliderFloat("Roughness", &roughness, 0, 1))
                    m_material.set_roughness(roughness);

                float metallic = m_material.get_metallic();
                if (ImGui::SliderFloat("Metallic", &metallic, 0, 1))
                    m_material.set_metallic(metallic);

                float specularity = m_material.get_specularity();
                if (ImGui::SliderFloat("Specularity", &specularity, 0, 1))
                    m_material.set_specularity(specularity);

                ImGui::End();
            }

        private:
            Material m_material;
        };

        imgui->add_frame(std::make_unique<MaterialGUI>(*Materials::get_iterable().begin()));
    }
}

} // NS Scenes

#endif // _SIMPLEVIEWER_SPHERE_SCENE_H_