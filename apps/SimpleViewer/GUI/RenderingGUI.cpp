// SimpleViewer rendering GUI.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <GUI/RenderingGUI.h>

#include <Cogwheel/Scene/Camera.h>

#include <DX11Renderer/Compositor.h>
#include <DX11Renderer/Renderer.h>

using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

namespace GUI {

void RenderingGUI::layout_frame() {
    ImGui::Begin("Rendering");

    if (ImGui::TreeNode("Compositor")) {
        if (ImGui::Button("Toggle V-sync")) {
            bool is_v_sync_enabled = m_compositor->uses_v_sync();
            m_compositor->set_v_sync(!is_v_sync_enabled);
        }

        ImGui::TreePop();
    }

    ImGui::Separator();

    if (ImGui::TreeNode("Camera effects")) {

        Cameras::UID camera_ID = *Cameras::get_iterable().begin();
        auto effects_settings = Cameras::get_effects_settings(camera_ID);
        bool has_changed = false;

        if (ImGui::TreeNode("Bloom")) {
            has_changed |= ImGui::InputFloat("Threshold", &effects_settings.bloom.threshold, 0.0f, 0.0f);
            has_changed |= ImGui::SliderFloat("Support", &effects_settings.bloom.support, 0.0f, 1.0f);
            ImGui::TreePop();
        }

        const char* exposure_modes[] = { "Fixed", "LogAverage", "Histogram" };
        int current_exposure_mode = (int)effects_settings.exposure.mode;
        has_changed |= ImGui::Combo("Exposure", &current_exposure_mode, exposure_modes, IM_ARRAYSIZE(exposure_modes));
        effects_settings.exposure.mode = (CameraEffects::ExposureMode)current_exposure_mode;

        const char* tonemapping_modes[] = { "Linear", "Filmic", "Uncharted2" };
        int current_tonemapping_mode = (int)effects_settings.tonemapping.mode;
        has_changed |= ImGui::Combo("Tonemapper", &current_tonemapping_mode, tonemapping_modes, IM_ARRAYSIZE(tonemapping_modes));
        effects_settings.tonemapping.mode = (CameraEffects::TonemappingMode)current_tonemapping_mode;

        if (has_changed)
            Cameras::set_effects_settings(camera_ID, effects_settings);

        ImGui::TreePop();
    }

    ImGui::Separator();

    if (m_renderer != nullptr) {
        if (ImGui::TreeNode("DirectX11")) {
            using namespace DX11Renderer;

            { // Settings
                auto settings = m_renderer->get_settings();
                bool has_changed = false;

                has_changed |= ImGui::SliderFloat("G-buffer band scale", &settings.g_buffer_guard_band_scale, 0.0f, 0.99f, "%.2f");

                if (ImGui::TreeNode("SSAO")) {
                    auto& ssao_settings = settings.ssao.settings;
                    has_changed |= ImGui::Checkbox("SSAO", &settings.ssao.enabled);
                    has_changed |= ImGui::InputFloat("World radius", &ssao_settings.world_radius, 0.05f, 0.25f, "%.2f");
                    ssao_settings.world_radius = max(0.0f, ssao_settings.world_radius);
                    has_changed |= ImGui::InputFloat("Bias", &ssao_settings.bias, 0.001f, 0.01f, "%.3f");
                    has_changed |= ImGui::InputFloat("Intensity", &ssao_settings.intensity_scale, 0.001f, 0.01f, "%.3f");
                    has_changed |= ImGui::InputFloat("Falloff", &ssao_settings.falloff, 0.001f, 0.01f, "%.3f");
                    has_changed |= ImGui::InputInt("Sample count", &ssao_settings.sample_count, 1, 5);
                    ssao_settings.sample_count = max(0, ssao_settings.sample_count);
                    has_changed |= ImGui::SliderFloat("Depth filtering %", &ssao_settings.depth_filtering_percentage, 0.0f, 1.0f, "%.2f");
                    has_changed |= ImGui::InputInt("Filter support", &ssao_settings.filter_support, 1, 5);
                    ssao_settings.filter_support = max(0, ssao_settings.filter_support);
                    has_changed |= ImGui::InputFloat("Normal std dev", &ssao_settings.normal_std_dev, 0.0f, 0.0f);
                    has_changed |= ImGui::InputFloat("Plane std dev", &ssao_settings.plane_std_dev, 0.0f, 0.0f);
                    ImGui::TreePop();
                }

                if (has_changed)
                    m_renderer->set_settings(settings);
            }

            // Debug settings
            if (ImGui::TreeNode("Debug")) {
                auto settings = m_renderer->get_debug_settings();
                bool has_changed = false;

                const char* display_modes[] = { "Color", "Normals", "Depth", "Ambient occlusion" };
                int current_display_mode = (int)settings.display_mode;
                has_changed |= ImGui::Combo("Tonemapper", &current_display_mode, display_modes, IM_ARRAYSIZE(display_modes));
                settings.display_mode = (Renderer::DebugSettings::DisplayMode)current_display_mode;

                if (has_changed)
                    m_renderer->set_debug_settings(settings);

                ImGui::TreePop();
            }

            ImGui::TreePop();
        }
    }

    ImGui::End();
}

} // NS GUI
