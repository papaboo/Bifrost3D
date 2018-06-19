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

    ImGui::Text("Compositor");
    if (ImGui::Button("Toggle V-sync")) {
        bool is_v_sync_enabled = m_compositor->uses_v_sync();
        m_compositor->set_v_sync(!is_v_sync_enabled);
    }

    ImGui::Separator();

    { // Camera efects
        Cameras::UID camera_ID = *Cameras::get_iterable().begin();
        auto effects_settings = Cameras::get_effects_settings(camera_ID);
        bool has_changed = false;

        ImGui::Text("Camera effects");

        if (ImGui::TreeNode("Bloom")) {
            has_changed |= ImGui::InputFloat("Threshold", &effects_settings.bloom.threshold, 0.0f, 0.0f);
            has_changed |= ImGui::SliderFloat("Bandwidth", &effects_settings.bloom.bandwidth, 0.0f, 1.0f);
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
    }

    if (m_renderer != nullptr) {
        if (ImGui::TreeNode("DirectX11")) {
            auto settings = m_renderer->get_settings();
            bool has_changed = false;

            has_changed |= ImGui::Checkbox("SSAO", &settings.ssao_enabled);

            if (has_changed)
                m_renderer->set_settings(settings);

            ImGui::TreePop();
        }
    }

    ImGui::End();
}

} // NS GUI
