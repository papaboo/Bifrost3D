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

using namespace Bifrost::Assets;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

namespace Scenes {

class MaterialGUI final : public ImGui::IImGuiFrame {
public:
    static const int material_count = 9;

    MaterialGUI(){
        m_material0_data = Materials::Data::create_dielectric(RGB(0.02f, 0.27f, 0.33f), 1.0f, 0.25f);
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
            m_material_IDs[m] = Materials::create("Lerped material", material_data);
        }
    }

    ~MaterialGUI() {
        for (int m = 0; m < material_count; ++m)
            Materials::destroy(m_material_IDs[m]);
    }

    inline Materials::UID get_material_ID(int index) const { return m_material_IDs[index]; }

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

        ImGui::End();

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
    }

private:
    Materials::Data m_material0_data;
    Materials::Data m_material1_data;

    Materials::UID m_material_IDs[material_count];
};

void create_material_scene(Cameras::UID camera_ID, SceneNode root_node, ImGui::ImGuiAdaptor* imgui) {

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
        const int tile_count_pr_side = 41;
        const int vertices_pr_side = tile_count_pr_side + 1;

        auto is_odd = [](int i) -> bool { return (i % 2) == 1; };

        { // White tiles.
            Transform transform = Transform(Vector3f(0.0f, -1.0f, 0.0f));
            SceneNode node = SceneNodes::create("White tiles", transform);
            int white_tile_count = tile_count_pr_side * tile_count_pr_side + is_odd(tile_count_pr_side);
            int vertex_count = (tile_count_pr_side + 1) * (tile_count_pr_side + 1);
            Meshes::UID tiles_mesh_ID = Meshes::create("White tiles", white_tile_count, vertex_count, MeshFlag::Position);

            Vector3f* positions = Meshes::get_positions(tiles_mesh_ID);
            for (int y = 0; y < vertices_pr_side; ++y)
                for (int x = 0; x < vertices_pr_side; ++x)
                    positions[x + y * vertices_pr_side] = Vector3f(x - 20.0f, 0.0f, y - 20.0f);

            Vector3ui* primitives = Meshes::get_primitives(tiles_mesh_ID);
            for (int y = 0; y < tile_count_pr_side; ++y)
                for (int x = 0; x < tile_count_pr_side; ++x) {
                    if ((x & 1) != (y & 1))
                        continue; // Ignore every other tile.

                    unsigned int base_index = x + y * vertices_pr_side;
                    *primitives = Vector3ui(base_index, base_index + vertices_pr_side, base_index + 1);
                    ++primitives;
                    *primitives = Vector3ui(base_index + 1, base_index + vertices_pr_side, base_index + vertices_pr_side + 1);
                    ++primitives;
                }

            Meshes::compute_bounds(tiles_mesh_ID);

            Materials::Data white_tile_data = Materials::Data::create_dielectric(RGB(0.5f), 0.4f, 0.25f);
            Materials::UID white_tile_material_ID = Materials::create("White tile", white_tile_data);

            MeshModels::create(node.get_ID(), tiles_mesh_ID, white_tile_material_ID);
            node.set_parent(root_node);
        }

        { // Black tiles.
            Transform transform = Transform(Vector3f(0.0f, -1.0f, 0.0f));
            SceneNode node = SceneNodes::create("Black tiles", transform);
            int white_tile_count = tile_count_pr_side * tile_count_pr_side - is_odd(tile_count_pr_side);
            int vertex_count = (tile_count_pr_side + 1) * (tile_count_pr_side + 1);
            Meshes::UID tiles_mesh_ID = Meshes::create("Black tiles", white_tile_count, vertex_count, MeshFlag::Position);

            Vector3f* positions = Meshes::get_positions(tiles_mesh_ID);
            for (int y = 0; y < vertices_pr_side; ++y)
                for (int x = 0; x < vertices_pr_side; ++x)
                    positions[x + y * vertices_pr_side] = Vector3f(x - 20.0f, 0.0f, y - 20.0f);

            Vector3ui* indices = Meshes::get_primitives(tiles_mesh_ID);
            for (int y = 0; y < tile_count_pr_side; ++y)
                for (int x = 0; x < tile_count_pr_side; ++x) {
                    if ((x & 1) == (y & 1))
                        continue; // Ignore every other tile.

                    unsigned int base_index = x + y * vertices_pr_side;
                    *indices = Vector3ui(base_index, base_index + vertices_pr_side, base_index + 1);
                    ++indices;
                    *indices = Vector3ui(base_index + 1, base_index + vertices_pr_side, base_index + vertices_pr_side + 1);
                    ++indices;
                }

            Meshes::compute_bounds(tiles_mesh_ID);

            Materials::Data black_tile_data = Materials::Data::create_dielectric(RGB(0.001f), 0.02f, 0.5f);
            Materials::UID black_tile_material_ID = Materials::create("Black tile", black_tile_data);

            MeshModels::create(node.get_ID(), tiles_mesh_ID, black_tile_material_ID);
            node.set_parent(root_node);
        }
    }

    // Materials
    auto& materials = imgui->make_frame<MaterialGUI>();

    { // Create material models.
        Meshes::UID cube_mesh_ID = MeshCreation::cube(1);
        Transform cube_transform = Transform(Vector3f(0.0f, -0.25f, 0.0f), Quaternionf::identity(), 1.5f);
        Meshes::UID sphere_mesh_ID = MeshCreation::revolved_sphere(32, 16);
        Transform sphere_transform = Transform(Vector3f(0.0f, 1.0f, 0.0f), Quaternionf::identity(), 1.5f);

        // Mesh combine models.
        Meshes::UID mesh_ID = MeshUtils::combine("MaterialMesh", cube_mesh_ID, cube_transform, sphere_mesh_ID, sphere_transform);
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