// SimpleViewer rendering GUI.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <GUI/RenderingGUI.h>

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Scene/Camera.h>

#include <DX11Renderer/Compositor.h>
#include <DX11Renderer/Renderer.h>

#include <StbImageWriter/StbImageWriter.h>

#ifdef OPTIX_FOUND
#include <OptiXRenderer/Renderer.h>
#endif

using namespace Bifrost::Assets;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace ImGui {

template <typename E>
inline bool CheckboxFlags(const char* label, Bifrost::Core::Bitmask<E>* flags, E flags_value) {
    unsigned int raw_flags = flags->raw();
    bool result = CheckboxFlags(label, &raw_flags, (unsigned int)flags_value);
    *flags = Bifrost::Core::Bitmask<E>(raw_flags);
    return result;
}

} // NS ImGui

namespace GUI {

struct RenderingGUI::State {
    struct {
        Bifrost::Math::CameraEffects::FilmicSettings filmic;
        Bifrost::Math::CameraEffects::Uncharted2Settings uncharted2;
    } tonemapping;

    struct {
        bool use_path_regularization = true;
        OptiXRenderer::PathRegularizationSettings path_regularization_settings;
    } optix;
};

RenderingGUI::RenderingGUI(DX11Renderer::Compositor* compositor, DX11Renderer::Renderer* dx_renderer, OptiXRenderer::Renderer* optix_renderer)
    : m_compositor(compositor), m_dx_renderer(dx_renderer), m_optix_renderer(optix_renderer), m_state(new State()) {
    strcpy_s(m_screenshot.path, m_screenshot.max_path_length, "c:\\temp\\ss");

    // Tonemapping parameters
    m_state->tonemapping.filmic = CameraEffects::FilmicSettings::default();
    m_state->tonemapping.uncharted2 = CameraEffects::Uncharted2Settings::default();
}

RenderingGUI::~RenderingGUI() { 
    delete m_state; 
}

void RenderingGUI::layout_frame() {
    // ImGui::ShowDemoWindow();

    ImGui::Begin("Rendering");
    ImGui::PushItemWidth(180);

    { // Screenshotting
        { // Resolve existing screenshots
            for (auto cam_ID : Cameras::get_iterable()) {
                auto output_screenshot = [&](Screenshot::Content content, char* file_extension) {
                    auto image_ID = Cameras::resolve_screenshot(cam_ID, content, "ss");
                    if (Images::has(image_ID)) {
                        std::string path = std::string(m_screenshot.path) + file_extension;
                        if (!StbImageWriter::write(image_ID, path))
                            printf("Failed to output screenshot to '%s'\n", path.c_str());
                        Images::destroy(image_ID);
                    }
                };
                output_screenshot(Screenshot::Content::ColorLDR, ".png");
                output_screenshot(Screenshot::Content::ColorHDR, ".hdr");
                output_screenshot(Screenshot::Content::Depth, "_depth.hdr");
                output_screenshot(Screenshot::Content::Albedo, "_albedo.png");
                output_screenshot(Screenshot::Content::Tint, "_tint.png");
                output_screenshot(Screenshot::Content::Roughness, "_roughness.png");
            }
        }

        ImGui::PoppedTreeNode("Screenshot", [&]() {
            bool take_screenshot = ImGui::Button("Take screenshots");
            ImGui::InputText("Path without extension", m_screenshot.path, m_screenshot.max_path_length);
            ImGui::InputUint("Iterations", &m_screenshot.iterations);
            ImGui::Checkbox("HDR", &m_screenshot.is_HDR);
            ImGui::CheckboxFlags("Depth", &m_screenshot.auxiliary_images, AuxiliaryImages::Depth);
            ImGui::CheckboxFlags("Albedo", &m_screenshot.auxiliary_images, AuxiliaryImages::Albedo);
            ImGui::CheckboxFlags("Tint", &m_screenshot.auxiliary_images, AuxiliaryImages::Tint);
            ImGui::CheckboxFlags("Roughness", &m_screenshot.auxiliary_images, AuxiliaryImages::Roughness);

            if (take_screenshot) {
                auto first_cam_ID = *Cameras::get_iterable().begin();
                Cameras::RequestedContent content = m_screenshot.is_HDR ? Screenshot::Content::ColorHDR : Screenshot::Content::ColorLDR;
                if (m_screenshot.auxiliary_images & AuxiliaryImages::Depth)
                    content |= Screenshot::Content::Depth;
                if (m_screenshot.auxiliary_images & AuxiliaryImages::Albedo)
                    content |= Screenshot::Content::Albedo;
                if (m_screenshot.auxiliary_images & AuxiliaryImages::Tint)
                    content |= Screenshot::Content::Tint;
                if (m_screenshot.auxiliary_images & AuxiliaryImages::Roughness)
                    content |= Screenshot::Content::Roughness;
                Cameras::request_screenshot(first_cam_ID, content, m_screenshot.iterations);
            }
        });
    }

    ImGui::Separator();

    ImGui::PoppedTreeNode("Compositor", [&]() {
        if (ImGui::Button("Toggle V-sync")) {
            bool is_v_sync_enabled = m_compositor->uses_v_sync();
            m_compositor->set_v_sync(!is_v_sync_enabled);
        }
    });

    ImGui::Separator();

    ImGui::PoppedTreeNode("Scene", [&]() {
        SceneRoot scene_root = *SceneRoots::get_iterable().begin();
        RGB environment_tint = scene_root.get_environment_tint();
        if (ImGui::ColorEdit3("Environment tint", &environment_tint.r))
            scene_root.set_environment_tint(environment_tint);

        ImGui::PoppedTreeNode("Camera", [&]() {
            Cameras::UID camera_ID = *Cameras::get_iterable().begin();

            // Translation
            Transform transform = Cameras::get_transform(camera_ID);
            Vector3f translation = transform.translation;
            ImGui::InputFloat3("Translation", translation.begin(), 3, ImGuiInputTextFlags_ReadOnly);

            // Rotation described as vertical and horizontal angle.
            Vector3f forward = transform.rotation.forward();
            float vertical_rotation = std::atan2(forward.x, forward.z);
            float horizontal_rotation = std::asin(forward.y);
            float rotation[2] = { vertical_rotation, horizontal_rotation };
            ImGui::InputFloat2("Rotation", rotation, 3, ImGuiInputTextFlags_ReadOnly);
        });
    });

    ImGui::Separator();

    ImGui::PoppedTreeNode("Materials", [&] {
        for (Material material : Materials::get_iterable()) {
            ImGui::PoppedTreeNode(material.get_name().c_str(), [&] {
                RGB tint = material.get_tint();
                if (ImGui::ColorEdit3("Tint", &tint.r))
                    material.set_tint(tint);

                float roughness = material.get_roughness();
                if (ImGui::SliderFloat("Roughness", &roughness, 0, 1))
                    material.set_roughness(roughness);

                float metallic = material.get_metallic();
                if (ImGui::SliderFloat("Metallic", &metallic, 0, 1))
                    material.set_metallic(metallic);

                float specularity = material.get_specularity();
                if (ImGui::SliderFloat("Specularity", &specularity, 0, 1))
                    material.set_specularity(specularity);

                float coat = material.get_coat();
                if (ImGui::SliderFloat("Coat", &coat, 0, 1))
                    material.set_coat(coat);

                float coat_roughness = material.get_coat_roughness();
                if (ImGui::SliderFloat("Coat roughness", &coat_roughness, 0, 1))
                    material.set_coat_roughness(coat_roughness);
            });
        }
    });

    ImGui::Separator();

    ImGui::PoppedTreeNode("Camera effects", [&]() {
        using namespace Bifrost::Math::CameraEffects;

        Cameras::UID camera_ID = *Cameras::get_iterable().begin();
        auto effects_settings = Cameras::get_effects_settings(camera_ID);
        bool has_changed = false;

        ImGui::PoppedTreeNode("Bloom", [&]() {
            has_changed |= ImGui::InputFloat("Threshold", &effects_settings.bloom.threshold, 0.0f, 0.0f);
            has_changed |= ImGui::SliderFloat("Support", &effects_settings.bloom.support, 0.0f, 1.0f);
        });

        ImGui::PoppedTreeNode("Exposure", [&]() {
            const char* exposure_modes[] = { "Fixed", "LogAverage", "Histogram" };
            int current_exposure_mode = (int)effects_settings.exposure.mode;
            has_changed |= ImGui::Combo("Exposure", &current_exposure_mode, exposure_modes, IM_ARRAYSIZE(exposure_modes));
            effects_settings.exposure.mode = (ExposureMode)current_exposure_mode;

            has_changed |= ImGui::InputFloat("Bias", &effects_settings.exposure.log_lumiance_bias, 0.25f, 1.0f, "%.2f");
        });

        ImGui::PoppedTreeNode("Tonemapping", [&]() {
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

            { // Plot tonemap curve
                const int max_sample_count = 32;
                float intensities[max_sample_count];
                for (int i = 0; i < max_sample_count; ++i) {
                    float c = (i / (max_sample_count - 1.0f)) * 2.0f;
                    switch (effects_settings.tonemapping.mode) {
                    case TonemappingMode::Filmic:
                        intensities[i] = luminance(unreal4(RGB(c), m_state->tonemapping.filmic)); break;
                    case TonemappingMode::Uncharted2:
                        intensities[i] = luminance(uncharted2(RGB(c), m_state->tonemapping.uncharted2)); break;
                    case TonemappingMode::Linear:
                        intensities[i] = c; break;
                    }
                }
                ImGui::PlotLines("", intensities, IM_ARRAYSIZE(intensities), 0, "Intensity [0, 2]", 0.0f, 1.0f, ImVec2(0, 80));
            }

            if (effects_settings.tonemapping.mode == TonemappingMode::Filmic) {
                auto& filmic = effects_settings.tonemapping.filmic;

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

                has_changed |= ImGui::SliderFloat("Black clip", &filmic.black_clip, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Toe", &filmic.toe, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Slope", &filmic.slope, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Shoulder", &filmic.shoulder, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("White clip", &filmic.white_clip, 0.0f, 1.0f);
            }
            if (effects_settings.tonemapping.mode == TonemappingMode::Uncharted2) {
                auto& uncharted2 = effects_settings.tonemapping.uncharted2;
                has_changed |= ImGui::SliderFloat("Shoulder strength", &uncharted2.shoulder_strength, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Linear strength", &uncharted2.linear_strength, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Linear angle", &uncharted2.linear_angle, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Toe strength", &uncharted2.toe_strength, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Toe numerator", &uncharted2.toe_numerator, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Toe denominator", &uncharted2.toe_denominator, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Linear white", &uncharted2.linear_white, 0.0f, 20.0f);

                if (ImGui::Button("Reset")) {
                    has_changed = true;
                    uncharted2 = Uncharted2Settings::default();
                }
            }
        });

        has_changed |= ImGui::InputFloat("Vignette", &effects_settings.vignette, 0.01f, 0.1f, "%.2f");

        if (has_changed)
            Cameras::set_effects_settings(camera_ID, effects_settings);
    });

    if (m_dx_renderer != nullptr) {
        ImGui::Separator();

        ImGui::PoppedTreeNode("DirectX11", [&]() {
            using namespace DX11Renderer;

            { // Settings
                auto settings = m_dx_renderer->get_settings();
                bool has_changed = false;

                has_changed |= ImGui::SliderFloat("G-buffer band scale", &settings.g_buffer_guard_band_scale, 0.0f, 0.99f, "%.2f");

                ImGui::PoppedTreeNode("SSAO", [&]() {
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
                });

                if (has_changed)
                    m_dx_renderer->set_settings(settings);
            }

            // Debug settings
            ImGui::PoppedTreeNode("Debug", [&]() {
                auto settings = m_dx_renderer->get_debug_settings();
                bool has_changed = false;

                const char* display_modes[] = { "Color", "Normals", "Depth", "Scene size", "Ambient occlusion", "Tint", "Roughness", "Metallic", "Coat", "Coat roughness", "Coverage", "UV" };
                int current_display_mode = (int)settings.display_mode;
                has_changed |= ImGui::Combo("Mode", &current_display_mode, display_modes, IM_ARRAYSIZE(display_modes));
                settings.display_mode = (Renderer::DebugSettings::DisplayMode)current_display_mode;

                if (has_changed)
                    m_dx_renderer->set_debug_settings(settings);
            });
        });
    }

#ifdef OPTIX_FOUND
    if (m_optix_renderer != nullptr) {
        ImGui::Separator();

        ImGui::PoppedTreeNode("OptiX", [&]() {
            auto camera_ID = *Cameras::get_iterable().begin();

            const char* backend_modes[] = { "Path tracer", "AI denoised GI", "Depth", "Albedo", "Tint", "Roughness" };
            int current_backend = int(m_optix_renderer->get_backend(camera_ID)) - 1;
            if (ImGui::Combo("Backend", &current_backend, backend_modes, IM_ARRAYSIZE(backend_modes))) {
                auto backend = OptiXRenderer::Backend(current_backend + 1);
                m_optix_renderer->set_backend(camera_ID, backend);
            }

            unsigned int max_bounce_count = m_optix_renderer->get_max_bounce_count(camera_ID);
            if (ImGui::InputUint("Max bounces", &max_bounce_count))
                m_optix_renderer->set_max_bounce_count(camera_ID, max_bounce_count);

            unsigned int max_accumulation_count = m_optix_renderer->get_max_accumulation_count(camera_ID);
            if (ImGui::InputUint("Max accumulations", &max_accumulation_count))
                m_optix_renderer->set_max_accumulation_count(camera_ID, max(max_accumulation_count, 1u));

            float epsilon = m_optix_renderer->get_scene_epsilon(*SceneRoots::get_iterable().begin());
            if (ImGui::InputFloat("Epsilon", &epsilon))
                m_optix_renderer->set_scene_epsilon(*SceneRoots::get_iterable().begin(), epsilon);

            ImGui::PoppedTreeNode("Path regularization", [&] {
                if (ImGui::Checkbox("Enable", &m_state->optix.use_path_regularization)) {
                    if (m_state->optix.use_path_regularization)
                        m_optix_renderer->set_path_regularization_settings(m_state->optix.path_regularization_settings);
                    else {
                        OptiXRenderer::PathRegularizationSettings settings = { 1e37f, 0.0f };
                        m_optix_renderer->set_path_regularization_settings(settings);
                    }
                }

                if (m_state->optix.use_path_regularization) {
                    auto& path_regularization_settings = m_state->optix.path_regularization_settings;
                    bool has_changes = ImGui::InputFloat("Scale", &path_regularization_settings.scale);
                    has_changes |= ImGui::InputFloat("Decay", &path_regularization_settings.decay);
                    if (has_changes)
                        m_optix_renderer->set_path_regularization_settings(path_regularization_settings);
                }
            });

            if (m_optix_renderer->get_backend(camera_ID) == OptiXRenderer::Backend::AIDenoisedPathTracing) {
                ImGui::PoppedTreeNode("AI Denoiser", [&]{
                    using OptiXRenderer::AIDenoiserFlag;

                    auto AI_flags = m_optix_renderer->get_AI_denoiser_flags();
                    bool has_AI_changes = ImGui::CheckboxFlags("Logarithmic update", &AI_flags, AIDenoiserFlag::LogarithmicFeedback);
                    has_AI_changes |= ImGui::CheckboxFlags("Use albedo", &AI_flags, AIDenoiserFlag::Albedo);
                    has_AI_changes |= ImGui::CheckboxFlags("Use normals", &AI_flags, AIDenoiserFlag::Normals);

                    const char* debug_vis_modes[] = { "Denoised image", "Noisy image", "Albedo", "Normals" };

                    int visualization_index = AI_flags & AIDenoiserFlag::VisualizeNormals ? 3 : 0;
                    visualization_index = AI_flags & AIDenoiserFlag::VisualizeAlbedo ? 2 : visualization_index;
                    visualization_index = AI_flags & AIDenoiserFlag::VisualizeNoise ? 1 : visualization_index;
                    if (ImGui::Combo("Visualization", &visualization_index, debug_vis_modes, IM_ARRAYSIZE(debug_vis_modes))) {
                        // Clear visualization modes
                        const OptiXRenderer::AIDenoiserFlags visualization_flags = { AIDenoiserFlag::VisualizeNoise, AIDenoiserFlag::VisualizeAlbedo, AIDenoiserFlag::VisualizeNormals };
                        AI_flags &= ~visualization_flags;

                        if (visualization_index == 1)
                            AI_flags |= AIDenoiserFlag::VisualizeNoise;
                        else if (visualization_index == 2)
                            AI_flags |= AIDenoiserFlag::VisualizeAlbedo;
                        else if (visualization_index == 3)
                            AI_flags |= AIDenoiserFlag::VisualizeNormals;

                        has_AI_changes = true;
                    }

                    if (has_AI_changes)
                        m_optix_renderer->set_AI_denoiser_flags(AI_flags);
                });
            }
        });
    }
#endif

    ImGui::End();
}

} // NS GUI
