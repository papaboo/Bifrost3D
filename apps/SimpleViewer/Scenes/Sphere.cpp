// SimpleViewer sphere scene.
// ---------------------------------------------------------------------------
// Copyright (C) 2015-2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Scenes/Sphere.h>

#include <Cogwheel/Assets/Shading/Fittings.h>
#include <Cogwheel/Assets/Material.h>
#include <Cogwheel/Assets/Mesh.h>
#include <Cogwheel/Assets/MeshCreation.h>
#include <Cogwheel/Assets/MeshModel.h>
#include <Cogwheel/Input/Keyboard.h>
#include <Cogwheel/Math/Distributions.h>
#include <Cogwheel/Scene/SceneNode.h>

#include <ImGui/ImGuiAdaptor.h>

#define NOMINMAX
#ifdef OPTIX_FOUND
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#endif

using namespace Cogwheel::Assets;
using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

namespace Scenes {

class MaterialGUI final : public ImGui::IImGuiFrame {
public:
    MaterialGUI(Material material) : m_material(material) {
        std::fill_n(m_ggx.sampled_rho, m_ggx.sample_count, 0.0f);
        m_ggx.accumulation_count = 0;

        std::fill_n(m_ggx_with_fresnel.sampled_rho, m_ggx_with_fresnel.sample_count, 0.0f);
        m_ggx_with_fresnel.accumulation_count = 0;
    }

    void layout_frame() {
        bool roughness_changed = false;
        bool specularity_changed = false;

        ImGui::Begin("Material");

        RGB tint = m_material.get_tint();
        if (ImGui::ColorEdit3("Tint", &tint.r))
            m_material.set_tint(tint);

        float roughness = m_material.get_roughness();
        if (ImGui::SliderFloat("Roughness", &roughness, 0, 1)) {
            m_material.set_roughness(roughness);
            roughness_changed = true;
        }

        float metallic = m_material.get_metallic();
        if (ImGui::SliderFloat("Metallic", &metallic, 0, 1))
            m_material.set_metallic(metallic);

        float specularity = m_material.get_specularity();
        if (ImGui::SliderFloat("Specularity", &specularity, 0, 1)) {
            m_material.set_specularity(specularity);
            specularity_changed = true;
        }


        ImGui::PoppedTreeNode("GGX", [&] {
            if (roughness_changed) {
                std::fill_n(m_ggx.sampled_rho, m_ggx.sample_count, 0.0f);
                m_ggx.accumulation_count = 0;
            }

            const float full_specularity = 1.0f;
            float ggx_rho[GGX::sample_count];
            for (int i = 0; i < GGX::sample_count; ++i) {
                float cos_theta = (i + 0.5f) / GGX::sample_count;
                ggx_rho[i] = Shading::Rho::sample_GGX(cos_theta, m_material.get_roughness());
                
                float ggx_alpha = m_material.get_roughness() * m_material.get_roughness();

                optix::float3 wo = { sqrtf(1.0f - cos_theta * cos_theta), 0.0f, cos_theta };
                Vector2f uv = RNG::sample02(m_ggx.accumulation_count, { 0,0 });
                auto ggx_sample = OptiXRenderer::Shading::BSDFs::GGX::sample(ggx_alpha, full_specularity, wo, { uv.x, uv.y });
                float new_sample_weight = OptiXRenderer::is_PDF_valid(ggx_sample.PDF) ? ggx_sample.weight.x * ggx_sample.direction.z / ggx_sample.PDF : 0.0f; // f * ||cos_theta|| / pdf
                m_ggx.sampled_rho[i] = (m_ggx.accumulation_count * m_ggx.sampled_rho[i] + new_sample_weight) / (m_ggx.accumulation_count + 1);
            }
            ++m_ggx.accumulation_count;

            ImGui::PlotData ggx_plot = { [&](int i) -> float { return ggx_rho[i]; } , GGX::sample_count, IM_COL32(0, 255, 0, 255), "Precomputed GGX" };
            ImGui::PlotData sampled_ggx_plot = { [&](int i) -> float { return m_ggx.sampled_rho[i]; } , GGX::sample_count, IM_COL32(0, 0, 255, 255), "Sampled GGX" };
            ImGui::PlotData plots[2] = { ggx_plot, sampled_ggx_plot };
            ImGui::PlotLines("", plots, 2, "", 0.0f, 1.0f, ImVec2(0, 80));

            ImGui::Text("%u samples", m_ggx.accumulation_count);
        });

        ImGui::PoppedTreeNode("GGX with Fresnel", [&] {
            if (roughness_changed || specularity_changed) {
                std::fill_n(m_ggx_with_fresnel.sampled_rho, m_ggx_with_fresnel.sample_count, 0.0f);
                m_ggx_with_fresnel.accumulation_count = 0;
            }

            float ggx_rho[GGX::sample_count];
            for (int i = 0; i < GGX::sample_count; ++i) {
                float cos_theta = (i + 0.5f) / GGX::sample_count;
                float ggx_no_fresnel_rho = Shading::Rho::sample_GGX(cos_theta, m_material.get_roughness());
                float ggx_full_fresnel_rho = Shading::Rho::sample_GGX_with_fresnel(cos_theta, m_material.get_roughness());
                float specularity_adjusted_ggx_rho = optix::lerp(ggx_full_fresnel_rho, ggx_no_fresnel_rho, specularity);
                ggx_rho[i] = specularity_adjusted_ggx_rho;

                float ggx_alpha = m_material.get_roughness() * m_material.get_roughness();

                optix::float3 wo = { sqrtf(1.0f - cos_theta * cos_theta), 0.0f, cos_theta };
                Vector2f uv = RNG::sample02(m_ggx_with_fresnel.accumulation_count, { 0,0 });
                auto ggx_sample = OptiXRenderer::Shading::BSDFs::GGX::sample(ggx_alpha, specularity, wo, { uv.x, uv.y });
                float new_sample_weight = OptiXRenderer::is_PDF_valid(ggx_sample.PDF) ? ggx_sample.weight.x * ggx_sample.direction.z / ggx_sample.PDF : 0.0f; // f * ||cos_theta|| / pdf
                m_ggx_with_fresnel.sampled_rho[i] = (m_ggx_with_fresnel.accumulation_count * m_ggx_with_fresnel.sampled_rho[i] + new_sample_weight) / (m_ggx_with_fresnel.accumulation_count + 1);
            }
            ++m_ggx_with_fresnel.accumulation_count;

            ImGui::PlotData ggx_plot = { [&](int i) -> float { return ggx_rho[i]; } , GGX::sample_count, IM_COL32(0, 255, 0, 255), "Precomputed GGX * Fresnel" };
            ImGui::PlotData sampled_ggx_plot = { [&](int i) -> float { return m_ggx_with_fresnel.sampled_rho[i]; } , GGX::sample_count, IM_COL32(0, 0, 255, 255), "Sampled GGX * Fresnel" };
            ImGui::PlotData plots[2] = { ggx_plot, sampled_ggx_plot };
            ImGui::PlotLines("", plots, 2, "", 0.0f, 1.0f, ImVec2(0, 80));

            ImGui::Text("%u samples", m_ggx_with_fresnel.accumulation_count);
        });

        ImGui::End();
    }

private:
    Material m_material;

    struct GGX {
        static const int sample_count = 64;
        float sampled_rho[sample_count];
        int accumulation_count;
    } m_ggx;

    struct GGXWithFresnel {
        static const int sample_count = 64;
        float sampled_rho[sample_count];
        int accumulation_count;
    } m_ggx_with_fresnel;
};

void create_sphere_scene(Engine& engine, Cameras::UID camera_ID, SceneRoots::UID scene_ID, ImGui::ImGuiAdaptor* imgui) {
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
        static auto swap_material = [](Engine& engine) {
            using namespace Cogwheel::Input;

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
                auto gold_mat_data = Materials::Data::create_metal(gold_tint, 0.02f);
                update_material(gold_mat_data);
            } else if (keyboard->was_released(Keyboard::Key::Key5)) {
                auto copper_mat_data = Materials::Data::create_metal(copper_tint, 0.5f);
                update_material(copper_mat_data);
            }
        };

        engine.add_mutating_callback([=, &engine]() { swap_material(engine); });
    }

    { // Setup GUI.
        imgui->add_frame(std::make_unique<MaterialGUI>(*Materials::get_iterable().begin()));
    }
}

} // NS Scenes
