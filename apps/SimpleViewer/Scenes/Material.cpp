// SimpleViewer material scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Scenes/Material.h>

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshCreation.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Assets/Shading/Fittings.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>

#include <ImGui/ImGuiAdaptor.h>

#define NOMINMAX
#ifdef OPTIX_FOUND
#include <OptiXRenderer/Shading/BSDFs/GGX.h>
#endif

using namespace Bifrost::Assets;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace Scenes {

class MaterialGUI final : public ImGui::IImGuiFrame {
public:
    static const int material_count = 9;

    MaterialGUI(){
        m_material0_data = Materials::Data::create_dielectric(RGB(0.02f, 0.27f, 0.33f), 1.0f, 0.02f);
        m_material1_data = Materials::Data::create_metal(gold_tint, 0.02f);

        for (int m = 0; m < material_count; ++m) {
            float lerp_t = m / (material_count - 1.0f);
            Materials::Data material_data = {};
            material_data.flags = MaterialFlag::None;
            material_data.tint = lerp(m_material0_data.tint, m_material1_data.tint, lerp_t);
            material_data.roughness = lerp(m_material0_data.roughness, m_material1_data.roughness, lerp_t);
            material_data.specularity = lerp(m_material0_data.specularity, m_material1_data.specularity, lerp_t);
            material_data.metallic = lerp(m_material0_data.metallic, m_material1_data.metallic, lerp_t);
            material_data.coverage = lerp(m_material0_data.coverage, m_material1_data.coverage, lerp_t);
            m_material_IDs[m] = Materials::create("Lerped material " + std::to_string(m), material_data);
        }

        std::fill_n(m_ggx.sampled_rho, m_ggx.sample_count, 0.0f);
        m_ggx.accumulation_count = 0;
        std::fill_n(m_ggx_with_fresnel.sampled_rho, m_ggx_with_fresnel.sample_count, 0.0f);
        m_ggx_with_fresnel.accumulation_count = 0;
    }

    ~MaterialGUI() {
        for (int m = 0; m < material_count; ++m)
            Materials::destroy(m_material_IDs[m]);
    }

    inline MaterialID get_material_ID(int index) const { return m_material_IDs[index]; }

    void layout_frame() {
        ImGui::Begin("Materials");

        bool updated = false;

        ImGui::PoppedTreeNode("Left material", [&] {
            updated |= ImGui::ColorEdit3("Tint", &m_material0_data.tint.r);
            updated |= ImGui::SliderFloat("Roughness", &m_material0_data.roughness, 0, 1);
            updated |= ImGui::SliderFloat("Metallic", &m_material0_data.metallic, 0, 1);
            updated |= ImGui::SliderFloat("Specularity", &m_material0_data.specularity, 0, 1);
        });

        ImGui::PoppedTreeNode("Right material", [&] {
            updated |= ImGui::ColorEdit3("Tint", &m_material1_data.tint.r);
            updated |= ImGui::SliderFloat("Roughness", &m_material1_data.roughness, 0, 1);
            updated |= ImGui::SliderFloat("Metallic", &m_material1_data.metallic, 0, 1);
            updated |= ImGui::SliderFloat("Specularity", &m_material1_data.specularity, 0, 1);
        });

        if (updated) {
            for (int m = 0; m < material_count; ++m) {
                float lerp_t = m / (material_count - 1.0f);
                Material material = m_material_IDs[m];
                material.set_tint(lerp(m_material0_data.tint, m_material1_data.tint, lerp_t));
                material.set_roughness(lerp(m_material0_data.roughness, m_material1_data.roughness, lerp_t));
                material.set_specularity(lerp(m_material0_data.specularity, m_material1_data.specularity, lerp_t));
                material.set_metallic(lerp(m_material0_data.metallic, m_material1_data.metallic, lerp_t));
                material.set_coverage(lerp(m_material0_data.coverage, m_material1_data.coverage, lerp_t));
            }
        }

#ifdef OPTIX_FOUND
        ImGui::PoppedTreeNode("Shading model properties", [&] {
            updated |= ImGui::SliderFloat("Material lerp", &m_material_lerp, 0, 1);
            float specularity = lerp(m_material0_data.specularity, m_material1_data.specularity, m_material_lerp);
            float roughness = lerp(m_material0_data.roughness, m_material1_data.roughness, m_material_lerp);
            float ggx_alpha = OptiXRenderer::Shading::BSDFs::GGX::alpha_from_roughness(roughness);
            ImGui::LabelText("Roughness", "%f", roughness);
            ImGui::LabelText("Specularity", "%f", specularity);

            ImGui::PoppedTreeNode("GGX", [&] {
                if (updated) {
                    std::fill_n(m_ggx.sampled_rho, m_ggx.sample_count, 0.0f);
                    m_ggx.accumulation_count = 0;
                }

                const float full_specularity = 1.0f;
                float ggx_rho[GGX::sample_count];
                for (int i = 0; i < GGX::sample_count; ++i) {
                    float cos_theta = (i + 0.5f) / GGX::sample_count;
                    ggx_rho[i] = Shading::Rho::sample_GGX(cos_theta, roughness);

                    optix::float3 wo = { sqrtf(1.0f - cos_theta * cos_theta), 0.0f, cos_theta };
                    Vector2f uv = RNG::sample02(m_ggx.accumulation_count, { 0,0 });
                    auto ggx_sample = OptiXRenderer::Shading::BSDFs::GGX_R::sample(ggx_alpha, full_specularity, wo, { uv.x, uv.y });
                    float new_sample_weight = OptiXRenderer::is_PDF_valid(ggx_sample.PDF) ? ggx_sample.reflectance.x * ggx_sample.direction.z / ggx_sample.PDF : 0.0f; // f * ||cos_theta|| / pdf
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
                if (updated) {
                    std::fill_n(m_ggx_with_fresnel.sampled_rho, m_ggx_with_fresnel.sample_count, 0.0f);
                    m_ggx_with_fresnel.accumulation_count = 0;
                }

                float ggx_rho[GGX::sample_count];
                for (int i = 0; i < GGX::sample_count; ++i) {
                    float cos_theta = (i + 0.5f) / GGX::sample_count;
                    float ggx_no_fresnel_rho = Shading::Rho::sample_GGX(cos_theta, roughness);
                    float ggx_full_fresnel_rho = Shading::Rho::sample_GGX_with_fresnel(cos_theta, roughness);
                    float specularity_adjusted_ggx_rho = optix::lerp(ggx_full_fresnel_rho, ggx_no_fresnel_rho, specularity);
                    ggx_rho[i] = specularity_adjusted_ggx_rho;

                    optix::float3 wo = { sqrtf(1.0f - cos_theta * cos_theta), 0.0f, cos_theta };
                    Vector2f uv = RNG::sample02(m_ggx_with_fresnel.accumulation_count, { 0,0 });
                    auto ggx_sample = OptiXRenderer::Shading::BSDFs::GGX_R::sample(ggx_alpha, specularity, wo, { uv.x, uv.y });
                    float new_sample_weight = OptiXRenderer::is_PDF_valid(ggx_sample.PDF) ? ggx_sample.reflectance.x * ggx_sample.direction.z / ggx_sample.PDF : 0.0f; // f * ||cos_theta|| / pdf
                    m_ggx_with_fresnel.sampled_rho[i] = (m_ggx_with_fresnel.accumulation_count * m_ggx_with_fresnel.sampled_rho[i] + new_sample_weight) / (m_ggx_with_fresnel.accumulation_count + 1);
                }
                ++m_ggx_with_fresnel.accumulation_count;

                ImGui::PlotData ggx_plot = { [&](int i) -> float { return ggx_rho[i]; } , GGX::sample_count, IM_COL32(0, 255, 0, 255), "Precomputed GGX * Fresnel" };
                ImGui::PlotData sampled_ggx_plot = { [&](int i) -> float { return m_ggx_with_fresnel.sampled_rho[i]; } , GGX::sample_count, IM_COL32(0, 0, 255, 255), "Sampled GGX * Fresnel" };
                ImGui::PlotData plots[2] = { ggx_plot, sampled_ggx_plot };
                ImGui::PlotLines("", plots, 2, "", 0.0f, 1.0f, ImVec2(0, 80));

                ImGui::Text("%u samples", m_ggx_with_fresnel.accumulation_count);
            });
        });
#endif // OPTIX_FOUND

        ImGui::End();
    }

private:
    Materials::Data m_material0_data;
    Materials::Data m_material1_data;

    MaterialID m_material_IDs[material_count];

    // Shading model visualizations
    float m_material_lerp = 0.0f;
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

void create_material_scene(CameraID camera_ID, SceneNode root_node, ImGui::ImGuiAdaptor* imgui) {

    { // Setup camera transform.
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 3.0f, -17.0f);
        cam_transform.look_at(Vector3f(0, 1.0f, 0.0f));
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Add a directional light.
        Transform light_transform = Transform(Vector3f(20.0f, 20.0f, -20.0f));
        light_transform.look_at(Vector3f::zero());
        SceneNode light_node = SceneNodes::create("light", light_transform);
        light_node.set_parent(root_node);
        LightSources::create_directional_light(light_node.get_ID(), RGB(3.0f, 2.9f, 2.5f));
    }

    { // Create floor.
      // A checker pattern texture would be really nice on the floor.
        unsigned int size = 41;
        ImageID tint_roughness_image_ID = Images::create2D("Floor color", PixelFormat::RGBA32, 2.2f, Vector2ui(size, size));
        Images::set_mipmapable(tint_roughness_image_ID, true);
        unsigned char* tint_roughness_pixels = (unsigned char*)Images::get_pixels(tint_roughness_image_ID);
        for (unsigned int y = 0; y < size; ++y) {
            for (unsigned int x = 0; x < size; ++x) {
                bool is_black = (x & 1) != (y & 1);
                unsigned char* pixel = tint_roughness_pixels + (x + y * size) * 4u;
                unsigned char intensity = is_black ? 1 : 255;
                pixel[0] = pixel[1] = pixel[2] = intensity;
                pixel[3] = is_black ? 6 : 102;
            }
        }

        Materials::Data material_data = Materials::Data::create_dielectric(RGB::white(), 1, 0.04f);
        material_data.tint_roughness_texture_ID = Textures::create2D(tint_roughness_image_ID, MagnificationFilter::None, MinificationFilter::Trilinear);
        material_data.flags = MaterialFlag::ThinWalled;
        MaterialID material_ID = Materials::create("Floor", material_data);

        SceneNode plane_node = SceneNodes::create("Floor", Transform(Vector3f(0.5, -1.0, 0.5), Quaternionf::identity(), float(size)));
        MeshID plane_mesh_ID = MeshCreation::plane(1, { MeshFlag::Position, MeshFlag::Texcoord });
        MeshModels::create(plane_node.get_ID(), plane_mesh_ID, material_ID);
        plane_node.set_parent(root_node);
    }

    // Materials
    auto& materials = imgui->make_frame<MaterialGUI>();

    { // Create material models.
        MeshID cube_mesh_ID = MeshCreation::cube(1);
        Transform cube_transform = Transform(Vector3f(0.0f, -0.25f, 0.0f), Quaternionf::identity(), 1.5f);
        MeshID sphere_mesh_ID = MeshCreation::revolved_sphere(32, 16);
        Transform sphere_transform = Transform(Vector3f(0.0f, 1.0f, 0.0f), Quaternionf::identity(), 1.5f);

        // Mesh combine models.
        MeshID mesh_ID = MeshUtils::combine("MaterialMesh", cube_mesh_ID, cube_transform, sphere_mesh_ID, sphere_transform);
        Meshes::destroy(cube_mesh_ID);
        Meshes::destroy(sphere_mesh_ID);

        for (int m = 0; m < materials.material_count; ++m) {
            Transform transform = Transform(Vector3f(float(m * 2 - 8), 0.0, 0.0f));
            SceneNode node = SceneNodes::create("Model", transform);
            MeshModels::create(node.get_ID(), mesh_ID, materials.get_material_ID(m));
            node.set_parent(root_node);
        }
    }
}

} // NS Scenes