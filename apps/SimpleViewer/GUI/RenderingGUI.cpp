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

#include <StbImageWriter/StbImageWriter.h>

using namespace Cogwheel::Assets;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

// ------------------------------------------------------------------------------------------------
// ImGui utility functions.
// ------------------------------------------------------------------------------------------------
namespace ImGui {

    bool InputUint(const char* label, unsigned int* v, unsigned int step = 1u, unsigned int step_fast = 100u, ImGuiInputTextFlags extra_flags = 0) {
        // Hexadecimal input provided as a convenience but the flag name is awkward. Typically you'd use InputText() to parse your own data, if you want to handle prefixes.
        const char* format = (extra_flags & ImGuiInputTextFlags_CharsHexadecimal) ? "%08X" : "%u";
        return InputScalar(label, ImGuiDataType_U32, (void*)v, (void*)(step > 0 ? &step : NULL), (void*)(step_fast > 0 ? &step_fast : NULL), format, extra_flags);
    }
}

namespace GUI {


struct RenderingGUI::State {
    struct {
        Cogwheel::Math::CameraEffects::FilmicSettings filmic;
        Cogwheel::Math::CameraEffects::Uncharted2Settings uncharted2;
    } tonemapping;
};

RenderingGUI::RenderingGUI(DX11Renderer::Compositor* compositor, DX11Renderer::Renderer* renderer)
    : m_compositor(compositor), m_renderer(renderer), m_state(new State()) {
    strcpy_s(m_screenshot.path, m_screenshot.max_path_length, "c:\\temp\\ss.png");

    // Tonemapping parameters
    m_state->tonemapping.filmic = CameraEffects::FilmicSettings::default();
    m_state->tonemapping.uncharted2 = CameraEffects::Uncharted2Settings::default();
}

RenderingGUI::~RenderingGUI() { 
    delete m_state; 
}

void RenderingGUI::layout_frame() {
    ImGui::Begin("Rendering");

    { // Screenshotting
        { // Resolve existing screenshots
            for (auto cam_ID : Cameras::get_iterable()) {
                Images::UID image_ID = Cameras::resolve_screenshot(cam_ID, "ss");
                if (Images::has(image_ID)) {
                    if (!StbImageWriter::write(image_ID, std::string(m_screenshot.path)))
                        printf("Failed to output screenshot to '%s'\n", m_screenshot.path);
                    Images::destroy(image_ID);
                }
            }
        }

        if (ImGui::TreeNode("Screenshot")) {
            bool take_screenshot = ImGui::Button("Take screenshots");
            ImGui::InputText("Path", m_screenshot.path, m_screenshot.max_path_length);
            ImGui::InputUint("Iterations", &m_screenshot.iterations);
            ImGui::Checkbox("HDR", &m_screenshot.is_HDR);

            if (take_screenshot) {
                auto first_cam_ID = *Cameras::get_iterable().begin();
                Cameras::request_screenshot(first_cam_ID, m_screenshot.is_HDR, m_screenshot.iterations);
            }

            ImGui::TreePop();
        }
    }

    ImGui::Separator();

    if (ImGui::TreeNode("Compositor")) {
        if (ImGui::Button("Toggle V-sync")) {
            bool is_v_sync_enabled = m_compositor->uses_v_sync();
            m_compositor->set_v_sync(!is_v_sync_enabled);
        }

        ImGui::TreePop();
    }

    ImGui::Separator();

    if (ImGui::TreeNode("Camera effects")) {
        using namespace Cogwheel::Math::CameraEffects;

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
        effects_settings.exposure.mode = (ExposureMode)current_exposure_mode;

        // Save tonemapping settings
        switch (effects_settings.tonemapping.mode) {
        case TonemappingMode::Filmic:
            m_state->tonemapping.filmic = effects_settings.tonemapping.filmic; break;
        case TonemappingMode::Uncharted2:
            m_state->tonemapping.uncharted2 = effects_settings.tonemapping.uncharted2; break;
        }

        const char* tonemapping_modes[] = { "Linear", "Filmic", "Uncharted2" };
        int current_tonemapping_mode = (int)effects_settings.tonemapping.mode;
        has_changed |= ImGui::Combo("Tonemapper", &current_tonemapping_mode, tonemapping_modes, IM_ARRAYSIZE(tonemapping_modes));
        effects_settings.tonemapping.mode = (TonemappingMode)current_tonemapping_mode;

        // Restore tonemapping settings
        switch (effects_settings.tonemapping.mode) {
        case TonemappingMode::Filmic:
            effects_settings.tonemapping.filmic = m_state->tonemapping.filmic; break;
        case TonemappingMode::Uncharted2:
            effects_settings.tonemapping.uncharted2 = m_state->tonemapping.uncharted2; break;
        }

        if (effects_settings.tonemapping.mode == TonemappingMode::Filmic) {
            auto& filmic = effects_settings.tonemapping.filmic;
            has_changed |= ImGui::SliderFloat("Black clip", &filmic.black_clip, 0.0f, 1.0f);
            has_changed |= ImGui::SliderFloat("Toe", &filmic.toe, 0.0f, 1.0f);
            has_changed |= ImGui::SliderFloat("Slope", &filmic.slope, 0.0f, 1.0f);
            has_changed |= ImGui::SliderFloat("Shoulder", &filmic.shoulder, 0.0f, 1.0f);
            has_changed |= ImGui::SliderFloat("White clip", &filmic.white_clip, 0.0f, 1.0f);

            const char* filmic_presets[] = { "Select preset", "ACES", "Uncharted2", "HP", "Legacy" };
            int current_preset = 0;
            has_changed |= ImGui::Combo("Preset", &current_preset, filmic_presets, IM_ARRAYSIZE(filmic_presets));
            switch (current_preset) {
            case 1:
                filmic = FilmicSettings::ACES(); break;
            case 2:
                filmic = FilmicSettings::uncharted2(); break;
            case 3:
                filmic = FilmicSettings::HP(); break;
            case 4:
                filmic = FilmicSettings::legacy(); break;
            }

        }
        if (effects_settings.tonemapping.mode == TonemappingMode::Uncharted2) {
            auto& uncharted2 = effects_settings.tonemapping.uncharted2;
            has_changed |= ImGui::SliderFloat("Shoulder strength", &uncharted2.shoulder_strength, 0.0f, 1.0f);
            has_changed |= ImGui::SliderFloat("Linear strength", &uncharted2.linear_strength, 0.0f, 1.0f);
            has_changed |= ImGui::SliderFloat("Linear angle", &uncharted2.linear_angle, 0.0f, 1.0f);
            has_changed |= ImGui::SliderFloat("Toe strength", &uncharted2.toe_strength, 0.0f, 1.0f);
            has_changed |= ImGui::SliderFloat("Toe numerator", &uncharted2.toe_numerator, 0.0f, 1.0f);
            has_changed |= ImGui::SliderFloat("Toe denominator", &uncharted2.toe_denominator, 0.0f, 1.0f);
            // uncharted2.linear_white

            if (ImGui::Button("Reset")) {
                has_changed = true;
                uncharted2 = Uncharted2Settings::default();
            }
        }

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
                    has_changed |= ImGui::InputUint("Sample count", &ssao_settings.sample_count, 1u, 5u);
                    has_changed |= ImGui::SliderFloat("Depth filtering %", &ssao_settings.depth_filtering_percentage, 0.0f, 1.0f, "%.2f");

                    const char* filter_types[] = { "Cross", "Box" };
                    int current_filter_type = (int)ssao_settings.filter_type;
                    has_changed |= ImGui::Combo("Filter type", &current_filter_type, filter_types, IM_ARRAYSIZE(filter_types));
                    ssao_settings.filter_type = (SsaoFilter)current_filter_type;
                    if (ssao_settings.filter_type != SsaoFilter::Box) {
                        has_changed |= ImGui::InputInt("Filter support", &ssao_settings.filter_support, 1, 5);
                        ssao_settings.filter_support = max(0, ssao_settings.filter_support);
                    }
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

                const char* display_modes[] = { "Color", "Normals", "Depth", "Scene size", "Ambient occlusion" };
                int current_display_mode = (int)settings.display_mode;
                has_changed |= ImGui::Combo("Mode", &current_display_mode, display_modes, IM_ARRAYSIZE(display_modes));
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
