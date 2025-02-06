// SimpleViewer material scene.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include <Scenes/Material.h>
#include <Scenes/Utils.h>

#include <Bifrost/Assets/Mesh.h>
#include <Bifrost/Assets/MeshModel.h>
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>

#include <ImGui/ImGuiAdaptor.h>

using namespace Bifrost::Assets;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace Scenes {

class MaterialGUI final : public ImGui::IImGuiFrame {
public:
    static const int material_count = 7;

    MaterialGUI(){
        m_material0_data = Materials::Data::create_dielectric(RGB(0.02f, 0.27f, 0.33f), 1.0f, 0.04f);
        m_material1_data = Materials::Data::create_metal(gold_tint, 0.02f);
        m_material1_data.specularity = m_material0_data.specularity;

        for (int m = 0; m < material_count; ++m) {
            float lerp_t = m / (material_count - 1.0f);
            Materials::Data material_data = {};
            material_data.flags = MaterialFlag::None;
            material_data.tint = lerp(m_material0_data.tint, m_material1_data.tint, lerp_t);
            material_data.roughness = lerp(m_material0_data.roughness, m_material1_data.roughness, lerp_t);
            material_data.specularity = lerp(m_material0_data.specularity, m_material1_data.specularity, lerp_t);
            material_data.metallic = lerp(m_material0_data.metallic, m_material1_data.metallic, lerp_t);
            material_data.coverage = lerp(m_material0_data.coverage, m_material1_data.coverage, lerp_t);
            m_materials[m] = Material("Lerped material " + std::to_string(m), material_data);
        }

        std::fill_n(m_ggx.sampled_rho, m_ggx.sample_count, 0.0f);
        m_ggx.accumulation_count = 0;
        std::fill_n(m_ggx_with_fresnel.sampled_rho, m_ggx_with_fresnel.sample_count, 0.0f);
        m_ggx_with_fresnel.accumulation_count = 0;
    }

    ~MaterialGUI() {
        for (int m = 0; m < material_count; ++m)
            Materials::destroy(m_materials[m].get_ID());
    }

    inline Material get_material(int index) const { return m_materials[index]; }

    void layout_frame() {
        ImGui::Begin("Materials");

        bool updated = false;

        const char* shading_model_names[] = { "Default", "Diffuse", "Transmissive" };
        int current_shading_model = (int)m_material0_data.shading_model;
        if (ImGui::Combo("Shading model", &current_shading_model, shading_model_names, IM_ARRAYSIZE(shading_model_names))) {
            m_material0_data.shading_model = ShadingModel(current_shading_model);
            m_material1_data.shading_model = ShadingModel(current_shading_model);
            for (int m = 0; m < material_count; ++m) {
                m_materials[m].set_shading_model(ShadingModel(current_shading_model));
            }
        }

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
                Material material = m_materials[m];
                material.set_tint(lerp(m_material0_data.tint, m_material1_data.tint, lerp_t));
                material.set_roughness(lerp(m_material0_data.roughness, m_material1_data.roughness, lerp_t));
                material.set_specularity(lerp(m_material0_data.specularity, m_material1_data.specularity, lerp_t));
                material.set_metallic(lerp(m_material0_data.metallic, m_material1_data.metallic, lerp_t));
                material.set_coverage(lerp(m_material0_data.coverage, m_material1_data.coverage, lerp_t));
            }
        }

        ImGui::End();
    }

private:
    Materials::Data m_material0_data;
    Materials::Data m_material1_data;

    Material m_materials[material_count];

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

SceneNode shallow_clone(SceneNode node) {
    // Clone the node
    SceneNode cloned_node = SceneNodes::create(node.get_name(), node.get_global_transform());

    // Clone the mesh model
    MeshModel mesh_model = MeshModels::get_attached_mesh_model(node.get_ID());
    if (mesh_model.exists())
        MeshModels::create(cloned_node.get_ID(), mesh_model.get_mesh().get_ID(), mesh_model.get_material().get_ID());

    // Recurse over children and attach to cloned node
    for (SceneNode child_node : node.get_children()) {
        SceneNode cloned_child = shallow_clone(child_node);
        cloned_child.set_parent(cloned_node);
    }

    return cloned_node;
}

void create_material_scene(CameraID camera_ID, SceneNode root_node, ImGui::ImGuiAdaptor* imgui, const std::filesystem::path& resource_directory) {

    { // Setup camera transform.
        Transform cam_transform = Cameras::get_transform(camera_ID);
        cam_transform.translation = Vector3f(0, 5.5f, -18.5f);
        cam_transform.look_at(Vector3f(0, 0.5f, 0.0f));
        Cameras::set_transform(camera_ID, cam_transform);
    }

    { // Add a directional light.
        Transform light_transform = Transform(Vector3f(20.0f, 20.0f, -20.0f));
        light_transform.look_at(Vector3f::zero());
        SceneNode light_node = SceneNodes::create("light", light_transform);
        light_node.set_parent(root_node);
        LightSources::create_directional_light(light_node.get_ID(), RGB(3.0f, 2.9f, 2.5f));
    }

    { // Create checkered floor.
        SceneNode floor_node = create_checkered_floor(400, 1);
        floor_node.set_global_transform(Transform(Vector3f(0, -1.0f, 0)));
        floor_node.set_parent(root_node);
    }

    // Materials
    auto& materials = imgui->make_frame<MaterialGUI>();

    { // Create material models.
        const float shader_ball_distance = 1.2f;

        SceneNode shader_ball_node = load_shader_ball(resource_directory, materials.get_material(0));
        Transform transform = Transform(Vector3f::zero(), Quaternionf::identity(), 2.0f);
        shader_ball_node.set_global_transform(transform);
        float shader_ball_pos_x = -shader_ball_distance * 0.5f * (materials.material_count - 1);
        shader_ball_node.apply_delta_transform(Transform(Vector3f(shader_ball_pos_x, 0, 0)));
        shader_ball_node.set_parent(root_node);

        for (int m = 1; m < materials.material_count; ++m) {
            SceneNode shader_ball_node_clone = shallow_clone(shader_ball_node);
            shader_ball_node_clone.apply_delta_transform(Transform(Vector3f(m * shader_ball_distance, 0, 0)));
            shader_ball_node_clone.set_parent(root_node);

            // Set outer surface to tested material
            replace_material(materials.get_material(m), shader_ball_node_clone, "Node5");
        }
    }
}

} // NS Scenes