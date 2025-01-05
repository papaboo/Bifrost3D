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
#include <Bifrost/Scene/Camera.h>
#include <Bifrost/Scene/LightSource.h>
#include <Bifrost/Scene/SceneNode.h>

#include <ImGui/ImGuiAdaptor.h>
#include <glTFLoader/glTFLoader.h>

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

void replace_material(Material material, SceneNode parent_node, const std::string& child_scene_node_name) {
    parent_node.apply_to_children_recursively([=](SceneNode node) {
        if (node.get_name() == child_scene_node_name) {
            MeshModel mesh_model = MeshModels::get_attached_mesh_model(node.get_ID());
            if (mesh_model.exists())
                mesh_model.set_material(material);
        }
    });
}

void create_material_scene(CameraID camera_ID, SceneNode root_node, ImGui::ImGuiAdaptor* imgui, const std::filesystem::path& data_directory) {

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
        const float shader_ball_distance = 300.0f;

        // Load and setup shaderball
        printf("Shader ball curtesy of https://github.com/derkreature/ShaderBall\n");
        auto shader_ball_path = data_directory / "SimpleViewer" / "Shaderball.glb";
        SceneNode shader_ball_node = glTFLoader::load(shader_ball_path.generic_string());
        float shader_ball_pos_x = shader_ball_distance * 0.5f * (materials.material_count - 1);
        Transform transform = Transform(Vector3f::zero(), Quaternionf::from_angle_axis(PI<float>(), Vector3f::up()), 0.01f);
        shader_ball_node.set_global_transform(transform);
        shader_ball_node.apply_delta_transform(Transform(Vector3f(shader_ball_pos_x, -102, 0)));
        shader_ball_node.set_parent(root_node);

        // Set base materials to rubber
        Materials::Data rubber_material_data = Materials::Data::create_dielectric(RGB(0.05f), 1, 0.04f);
        Material rubber_material = Materials::create("Rubber", rubber_material_data);
        for (std::string node_name : { "Node1", "Node4", "Node5" })
            replace_material(rubber_material, shader_ball_node, node_name);

        // Set surrounding surface to tested material
        for (std::string node_name : { "Node2", "Node3", "Node6" })
            replace_material(materials.get_material_ID(0), shader_ball_node, node_name);

        for (int m = 1; m < materials.material_count; ++m) {
            SceneNode shader_ball_node_clone = shallow_clone(shader_ball_node);
            shader_ball_node_clone.apply_delta_transform(Transform(Vector3f(m * -shader_ball_distance, 0, 0)));
            shader_ball_node_clone.set_parent(root_node);

            // Set surrounding surface to tested material
            for (std::string node_name : { "Node2", "Node3", "Node6" })
                replace_material(materials.get_material_ID(m), shader_ball_node_clone, node_name);
        }
    }
}

} // NS Scenes