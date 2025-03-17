// SimpleViewer rendering GUI.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <GUI/RenderingGUI.h>
#include <CameraHandlers.h>

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>

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

void camera_effects(CameraID camera_ID) {
    ImGui::PoppedTreeNode("Camera effects", [&]() {
        using namespace Bifrost::Math::CameraEffects;

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
            const char* tonemapping_modes[] = { "Linear", "Filmic" };
            int current_tonemapping_mode = (int)effects_settings.tonemapping.mode;
            has_changed |= ImGui::Combo("Tonemapper", &current_tonemapping_mode, tonemapping_modes, IM_ARRAYSIZE(tonemapping_modes));
            effects_settings.tonemapping.mode = (TonemappingMode)current_tonemapping_mode;

            { // Plot tonemap curve
                auto filmic_settings = effects_settings.tonemapping.settings;

                const int max_sample_count = 32;
                float intensities[max_sample_count];
                for (int i = 0; i < max_sample_count; ++i) {
                    float c = (i / (max_sample_count - 1.0f)) * 2.0f;
                    switch (effects_settings.tonemapping.mode) {
                    case TonemappingMode::Filmic:
                        intensities[i] = luminance(filmic(RGB(c), filmic_settings)); break;
                    case TonemappingMode::Linear:
                        intensities[i] = c; break;
                    }
                }
                ImGui::PlotLines("", intensities, IM_ARRAYSIZE(intensities), 0, "Intensity [0, 2]", 0.0f, 1.0f, ImVec2(0, 80));
            }

            if (effects_settings.tonemapping.mode == TonemappingMode::Filmic) {
                auto& filmic = effects_settings.tonemapping.settings;

                const char* filmic_presets[] = { "Select preset", "ACES", "Uncharted2", "HP", "Legacy" };
                int current_preset = 0;
                has_changed |= ImGui::Combo("Preset", &current_preset, filmic_presets, IM_ARRAYSIZE(filmic_presets));
                switch (current_preset) {
                case 1:
                    filmic = TonemappingSettings::ACES(); break;
                case 2:
                    filmic = TonemappingSettings::uncharted2(); break;
                case 3:
                    filmic = TonemappingSettings::HP(); break;
                case 4:
                    filmic = TonemappingSettings::legacy(); break;
                }

                has_changed |= ImGui::SliderFloat("Black clip", &filmic.black_clip, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Toe", &filmic.toe, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Slope", &filmic.slope, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("Shoulder", &filmic.shoulder, 0.0f, 1.0f);
                has_changed |= ImGui::SliderFloat("White clip", &filmic.white_clip, 0.0f, 1.0f);
            }
        });

        has_changed |= ImGui::InputFloat("Vignette", &effects_settings.vignette, 0.01f, 0.1f, "%.2f");

        has_changed |= ImGui::InputFloat("Film grain", &effects_settings.film_grain, 0.0001f, 0.001f, "%.4f");

        if (has_changed)
            Cameras::set_effects_settings(camera_ID, effects_settings);
    });
}

void camera_GUI(CameraID camera_ID, CameraNavigation* navigation) {

    // Viewport
    ImGui::PoppedTreeNode("Viewport", [&]() {
        Rectf viewport = Cameras::get_viewport(camera_ID);
        bool has_changed = ImGui::SliderFloat("X", &viewport.x, 0.0f, 1.0f);
        has_changed |= ImGui::SliderFloat("Y", &viewport.y, 0.0f, 1.0f);
        has_changed |= ImGui::SliderFloat("width", &viewport.width, 0.0f, 1.0f);
        has_changed |= ImGui::SliderFloat("height", &viewport.height, 0.0f, 1.0f);
        if (has_changed)
            Cameras::set_viewport(camera_ID, viewport);
    });

    ImGui::PoppedTreeNode("Movement", [&]() {
        // Camera speed
        if (camera_ID == navigation->get_camera_ID())
        {
            float velocity = navigation->get_velocity();
            if (ImGui::InputFloat("Velocity", &velocity))
                navigation->set_velocity(velocity);
        }

        // Translation
        Transform transform = Cameras::get_transform(camera_ID);
        Vector3f translation = transform.translation;
        ImGui::InputFloat3("Translation", translation.begin(), "%.3f", ImGuiInputTextFlags_ReadOnly);

        // Rotation described as vertical and horizontal angle.
        Vector3f forward = transform.rotation.forward();
        float vertical_rotation = std::atan2(forward.x, forward.z);
        float horizontal_rotation = std::asin(forward.y);
        float rotation[2] = { vertical_rotation, horizontal_rotation };
        ImGui::InputFloat2("Rotation", rotation, "%.3f", ImGuiInputTextFlags_ReadOnly);
    });

    camera_effects(camera_ID);
}

struct RenderingGUI::State {
#ifdef OPTIX_FOUND
    struct {
        bool use_path_regularization = true;
        OptiXRenderer::PathRegularizationSettings path_regularization_settings;
    } optix;
#endif // OPTIX_FOUND
};

RenderingGUI::RenderingGUI(CameraNavigation* navigation, DX11Renderer::Compositor* compositor,
    DX11Renderer::Renderer* dx_renderer, OptiXRenderer::Renderer* optix_renderer)
    : m_navigation(navigation), m_compositor(compositor), m_dx_renderer(dx_renderer), m_optix_renderer(optix_renderer), m_state(new State()) {
    m_screenshot = {};
    strcpy_s(m_screenshot.path, m_screenshot.max_path_length, "c:\\temp\\ss");

#ifdef OPTIX_FOUND
    m_state->optix.path_regularization_settings = m_optix_renderer->get_path_regularization_settings();
#endif
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
                    Image image = Cameras::resolve_screenshot(cam_ID, content, "ss");
                    if (image.exists()) {
                        std::string path = std::string(m_screenshot.path) + file_extension;
                        if (!StbImageWriter::write(image, path))
                            printf("Failed to output screenshot to '%s'\n", path.c_str());
                        image.destroy();
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
            ImGui::CheckboxFlags("HDR", &m_screenshot.screenshot_content, Screenshot::Content::ColorHDR);
            ImGui::CheckboxFlags("Depth", &m_screenshot.screenshot_content, Screenshot::Content::Depth);
            ImGui::CheckboxFlags("Albedo", &m_screenshot.screenshot_content, Screenshot::Content::Albedo);
            ImGui::CheckboxFlags("Tint", &m_screenshot.screenshot_content, Screenshot::Content::Tint);
            ImGui::CheckboxFlags("Roughness", &m_screenshot.screenshot_content, Screenshot::Content::Roughness);

            if (take_screenshot) {
                auto first_cam_ID = *Cameras::get_iterable().begin();
                auto content = m_screenshot.screenshot_content;
                if (content.not_set(Screenshot::Content::ColorHDR))
                    content |= Screenshot::Content::ColorLDR;
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
            int camera_count = 0;
            for (CameraID camera_ID : Cameras::get_iterable()) {
                camera_count++;
                ImGui::PoppedTreeNode(Cameras::get_name(camera_ID).c_str(), [&]() {
                    camera_GUI(camera_ID, m_navigation);
                });
            }

            if (camera_count == 1 && ImGui::Button("Add picture in picture")) {
                // Picture in picture camera option
                CameraID camera_ID = *(Cameras::get_iterable().begin());
                CameraID new_cam_ID = Cameras::create("Picture-in-picture", Cameras::get_scene_ID(camera_ID),
                                                      Cameras::get_projection_matrix(camera_ID), Cameras::get_inverse_projection_matrix(camera_ID));
                Cameras::set_transform(new_cam_ID, Cameras::get_transform(camera_ID));
                Cameras::set_viewport(new_cam_ID, Rectf(0.75f, 0.75f, 0.25f, 0.25));
                Cameras::set_z_index(new_cam_ID, Cameras::get_z_index(camera_ID) + 1);
            }
        });

        ImGui::Separator();

        ImGui::PoppedTreeNode("Lights", [&]() {
            for (LightSource light : LightSources::get_iterable()) {
                SceneNode scene_node = light.get_node();
                ImGui::PoppedTreeNode(scene_node.get_name().c_str(), [&]() {
                    LightSources::Type light_type = light.get_type();
                    
                    if (light_type == LightSources::Type::Sphere) {
                        SphereLight sphere_light = light;

                        float radius = sphere_light.get_radius();
                        if (ImGui::InputFloat("Radius", &radius))
                            sphere_light.set_radius(radius);

                        RGB power = sphere_light.get_power();
                        if (ImGui::InputFloat3("Power", &power.r))
                            sphere_light.set_power(power);
                    } else if (light_type == LightSources::Type::Spot) {
                        SpotLight spot_light = light;
                        
                        float radius = spot_light.get_radius();
                        if (ImGui::InputFloat("Radius", &radius))
                            spot_light.set_radius(radius);

                        float cos_angle = spot_light.get_cos_angle();
                        if (ImGui::SliderFloat("Cos(angle)", &cos_angle, 0, 0.999f))
                            spot_light.set_cos_angle(cos_angle);

                        RGB power = spot_light.get_power();
                        if (ImGui::InputFloat3("Power", &power.r))
                            spot_light.set_power(power);
                    } else if (light_type == LightSources::Type::Directional) {
                        DirectionalLight directional_light = light;

                        RGB radiance = directional_light.get_radiance();
                        if (ImGui::InputFloat3("Radiance", &radiance.r))
                            directional_light.set_radiance(radiance);
                    }

                    if (ImGui::Button("Delete"))
                        light.destroy();
                });
            }

            auto create_new_light_node = [](const std::string& name) -> SceneNode {
                CameraID camera_ID = *Cameras::get_iterable().begin();
                Transform transform = Cameras::get_transform(camera_ID);
                SceneNode root_node = SceneRoot(Cameras::get_scene_ID(camera_ID)).get_root_node();
                SceneNode light_node = SceneNode(name, transform);
                light_node.set_parent(root_node);
                return light_node;
            };

            if (ImGui::Button("Create Directional Light")) {
                SceneNode light_node = create_new_light_node("Directional Light");
                DirectionalLight(light_node, RGB(0.5f));
            }
            if (ImGui::Button("Create Sphere Light")) {
                SceneNode light_node = create_new_light_node("Sphere Light");
                SphereLight(light_node, RGB(1.0f), 0.1f);
            }
        });

        ImGui::Separator();

        ImGui::PoppedTreeNode("Materials", [&] {
            for (Material material : Materials::get_iterable()) {
                ImGui::PoppedTreeNode(material.get_name().c_str(), [&] {
                    const char* shading_model_names[] = { "Default", "Diffuse", "Transmissive" };
                    int current_shading_model = (int)material.get_shading_model();
                    if (ImGui::Combo("Shading model", &current_shading_model, shading_model_names, IM_ARRAYSIZE(shading_model_names)))
                        material.set_shading_model(ShadingModel(current_shading_model));

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
                    if (ImGui::SliderFloat("Specularity", &specularity, 0, 0.08f))
                        material.set_specularity(specularity);

                    float coat = material.get_coat();
                    if (ImGui::SliderFloat("Coat", &coat, 0, 1))
                        material.set_coat(coat);

                    float coat_roughness = material.get_coat_roughness();
                    if (ImGui::SliderFloat("Coat roughness", &coat_roughness, 0, 1))
                        material.set_coat_roughness(coat_roughness);

                    if (material.is_cutout()) {
                        float cutout_threshold = material.get_cutout_threshold();
                        if (ImGui::SliderFloat("Cutout threshold", &cutout_threshold, 0, 1))
                            material.set_cutout_threshold(cutout_threshold);
                    }
                    else {
                        float coverage = material.get_coverage();
                        if (ImGui::SliderFloat("Coverage", &coverage, 0, 1))
                            material.set_coverage(coverage);
                    }

                    Materials::Flags flags = material.get_flags();
                    bool flags_changed = ImGui::CheckboxFlags("Thin walled", &flags, MaterialFlag::ThinWalled);
                    flags_changed |= ImGui::CheckboxFlags("Cutout", &flags, MaterialFlag::Cutout);
                    if (flags_changed)
                        material.set_flags(flags);
                });
            }
        });
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

            static const char* backend_modes[] = { "Uninitialized", "Path tracer", "AI denoised GI", "Depth", "Albedo", "Tint", "Roughness", "Shading normal", "Primitive ID" };
            int current_backend = int(m_optix_renderer->get_backend(camera_ID));
            if (ImGui::Combo("Backend", &current_backend, backend_modes, IM_ARRAYSIZE(backend_modes)) && current_backend != 0) {
                auto backend = OptiXRenderer::Backend(current_backend);
                m_optix_renderer->set_backend(camera_ID, backend);
            }

            unsigned int max_bounce_count = m_optix_renderer->get_max_bounce_count(camera_ID);
            if (ImGui::InputUint("Max bounces", &max_bounce_count))
                m_optix_renderer->set_max_bounce_count(camera_ID, max_bounce_count);

            unsigned int max_accumulation_count = m_optix_renderer->get_max_accumulation_count(camera_ID);
            if (ImGui::InputUint("Max accumulations", &max_accumulation_count))
                m_optix_renderer->set_max_accumulation_count(camera_ID, max(max_accumulation_count, 1u));

            int light_sample_count = m_optix_renderer->get_next_event_sample_count(*SceneRoots::get_iterable().begin());
            if (ImGui::InputInt("Light sample count", &light_sample_count))
                m_optix_renderer->set_next_event_sample_count(*SceneRoots::get_iterable().begin(), light_sample_count);

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
                    bool has_changes = ImGui::InputFloat("PDF scale", &path_regularization_settings.PDF_scale);
                    has_changes |= ImGui::InputFloat("Scale decay", &path_regularization_settings.scale_decay);
                    if (has_changes)
                        m_optix_renderer->set_path_regularization_settings(path_regularization_settings);
                }
            });

            if (m_optix_renderer->get_backend(camera_ID) == OptiXRenderer::Backend::AIDenoisedPathTracing) {
                ImGui::PoppedTreeNode("AI Denoiser", [&]{
                    using OptiXRenderer::AIDenoiserFlag;

                    auto AI_flags = m_optix_renderer->get_AI_denoiser_flags();
                    bool has_AI_changes = ImGui::CheckboxFlags("Logarithmic update", &AI_flags, AIDenoiserFlag::LogarithmicFeedback);

                    const char* debug_vis_modes[] = { "Denoised image", "Noisy image", "Albedo" };

                    int visualization_index = AI_flags.contains(AIDenoiserFlag::VisualizeAlbedo) ? 2 : 0;
                    visualization_index = AI_flags.contains(AIDenoiserFlag::VisualizeNoise) ? 1 : visualization_index;
                    if (ImGui::Combo("Visualization", &visualization_index, debug_vis_modes, IM_ARRAYSIZE(debug_vis_modes))) {
                        // Clear visualization modes
                        const OptiXRenderer::AIDenoiserFlags visualization_flags = { AIDenoiserFlag::VisualizeNoise, AIDenoiserFlag::VisualizeAlbedo };
                        AI_flags &= ~visualization_flags;

                        if (visualization_index == 1)
                            AI_flags |= AIDenoiserFlag::VisualizeNoise;
                        else if (visualization_index == 2)
                            AI_flags |= AIDenoiserFlag::VisualizeAlbedo;

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
