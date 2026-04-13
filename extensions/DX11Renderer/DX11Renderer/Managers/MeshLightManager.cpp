// DirectX 11 mesh light manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Dx11Renderer/Managers/MeshLightManager.h"

#include "Dx11Renderer/Managers/MaterialManager.h"
#include "Dx11Renderer/Managers/MeshManager.h"
#include "Dx11Renderer/Managers/MeshModelManager.h"
#include "Dx11Renderer/Utils.h"

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/MeshModel.h>

namespace DX11Renderer::Managers {

using namespace Bifrost::Assets;
using namespace Bifrost::Math;

inline RGB get_material_emission(Material material) {
    if (!material.exists())
        return RGB::black();

    if (material.is_cutout())
        printf("DX11Renderer::MeshLightManager warning: Cutout mesh lights not supported.");
    if (material.get_coverage_texture().exists())
        printf("DX11Renderer::MeshLightManager warning: Coverage texture not supported.");

    return material.get_emission() * material.get_coverage();
}

inline bool is_emissive_model(MeshModel model) {
    if (!model.exists() || !model.get_mesh().exists() || !model.get_material().exists())
        return false;

    bool emissive_vertices = model.get_mesh().get_emission() != nullptr;
    bool emissive_material = get_material_emission(model.get_material()) != RGB::black(); // Supports negative emission on purpose.
    return emissive_vertices || emissive_material;
}

void MeshLightManager::handle_updates(ID3D11Device1& device) {
    bool rebuild_combined_buffers = false;

    auto add_mesh_light = [&](MeshModel model) {
        Material material = model.get_material();
        RGB material_emission = get_material_emission(material);
        bool is_thinwalled = material.is_thin_walled();

        Mesh mesh = model.get_mesh();
        bool has_primitive_indices = mesh.get_primitives() != nullptr;
        unsigned int triangle_count = has_primitive_indices ? mesh.get_primitive_count() : (mesh.get_vertex_count() / 3);

        MeshLight light = { model.get_ID(), material.get_ID(), material_emission, is_thinwalled, triangle_count };
        m_lights.insert_or_assign(model.get_ID().get_index(), light);

        // Flag a buffer rebuild.
        rebuild_combined_buffers = true;
    };

    auto update_material = [&](MeshLight& light, Material material) {
        RGB material_emission = get_material_emission(material);
        bool is_thinwalled = material.is_thin_walled();

        if (material_emission != RGB::black()) {
            bool emission_changed = light.material_emission != material_emission || light.is_thinwalled != is_thinwalled;

            light.material_emission = material_emission;
            light.is_thinwalled = is_thinwalled;

            // Flag a buffer rebuild.
            rebuild_combined_buffers |= emission_changed;
        } else {
            // Remove the mesh light as emission is zero.
            m_lights.erase(light.model_ID.get_index()); // TODO use light_itr argument?
            rebuild_combined_buffers = true;
        }
    };

    for (MeshModel model : MeshModels::get_changed_models()) {
        if (model.get_changes().contains(MeshModels::Change::Destroyed)) {
            auto elements_removed = m_lights.erase(model.get_ID().get_index());
            if (elements_removed != 0)
                rebuild_combined_buffers = true;
        }

        if (model.get_changes().contains(MeshModels::Change::Created) && is_emissive_model(model))
            add_mesh_light(model);
        else if (model.get_changes().contains(MeshModels::Change::Material)) {
            auto light_itr = m_lights.find(model.get_ID().get_index());
            if (light_itr != m_lights.end())
                update_material(light_itr->second, model.get_material());
        }
    }

    // Check if any materials have changed.
    for (Material material : Materials::get_changed_materials()) {
        // Ignore material creation. A newly created material isn't interesting and this event is handled by MeshModel creation.
        if (material.get_changes().contains(Materials::Change::Created))
            continue;

        bool is_emissive = get_material_emission(material) != RGB::black();

        // This rarely happens, so for now we just loop over all mesh models
        for (MeshModel model : MeshModels::get_iterable()) {
            if (model.get_material() == material) {
                auto light_itr = m_lights.find(model.get_ID().get_index());
                if (light_itr != m_lights.end())
                    update_material(light_itr->second, model.get_material());
                else if (is_emissive)
                    // If the light is emissive and the model isn't found in the list of lights, then add it.
                    add_mesh_light(model);
            }
        }
    }

    // Check if any meshes have changed.
    for (Mesh mesh : Meshes::get_changed_meshes()) {
        // Ignore mesh creation. A newly created mesh isn't interesting and this event is handled by MeshModel creation.
        if (mesh.get_changes().contains(Meshes::Change::Created))
            continue;

        // This rarely happens, so for now we just loop over all mesh models
        for (MeshModel model : MeshModels::get_iterable()) {
            if (model.get_mesh() == mesh) {
                auto light_itr = m_lights.find(model.get_ID().get_index());

                bool is_emissive = is_emissive_model(model);
                bool model_is_light_source = light_itr != m_lights.end();

                // If the model is currently considered a light source, but isn't emissive, then remove it.
                if (model_is_light_source && !is_emissive) {
                    m_lights.erase(light_itr);
                    rebuild_combined_buffers = true;
                } else if (is_emissive && !model_is_light_source) {
                    // If the light is emissive and the model isn't found in the list of lights, then add it.
                    add_mesh_light(model);
                    rebuild_combined_buffers = true;
                }
            }
        }
    }

    if (rebuild_combined_buffers) {
        // Clear buffer to reset ownership and in case there's no mesh lights.
        m_combined_mesh_lights_SRV = nullptr;

        m_emissive_triangle_count = 0;
        for (auto light_itr : m_lights)
            m_emissive_triangle_count += light_itr.second.triangle_count;

        if (m_emissive_triangle_count > 0) {

            auto combined_mesh_lights = std::vector<MeshLightGPU>(); combined_mesh_lights.reserve(m_emissive_triangle_count);
            for (auto light_itr : m_lights) {
                MeshModel model = light_itr.second.model_ID;

                Mesh mesh = model.get_mesh();
                Vector3f* mesh_positions = mesh.get_positions();
                RGB* mesh_emission = mesh.get_emission();
                Transform local_to_world = model.get_scene_node().get_global_transform();

                Material material = model.get_material();
                RGB emission_scale = get_material_emission(material);
                bool is_thinwalled = material.is_thin_walled();

                // Extract and transform emissive triangles.
                auto add_triangle = [&](unsigned int index0, unsigned int index1, unsigned int index2) {
                    bool triangle_is_emissive = true;
                    RGB emission0 = emission_scale, emission1 = emission_scale, emission2 = emission_scale;
                    if (mesh_emission != nullptr) {
                        emission0 *= mesh_emission[index0];
                        emission1 *= mesh_emission[index1];
                        emission2 *= mesh_emission[index2];
                        triangle_is_emissive = emission0 != RGB::black() || emission1 != RGB::black() || emission2 != RGB::black();
                    }

                    if (triangle_is_emissive) {
                        MeshLightGPU light = {};
                        light.position0 = make_float3(local_to_world * mesh_positions[index0]);
                        light.position1 = make_float3(local_to_world * mesh_positions[index1]);
                        light.position2 = make_float3(local_to_world * mesh_positions[index2]);

                        light.emission0 = make_float3(emission0);
                        light.emission1 = make_float3(emission1);
                        light.emission2 = make_float3(emission2);

                        light.is_thinwalled = is_thinwalled;

                        combined_mesh_lights.push_back(light);
                    }
                };

                bool has_primitive_indices = mesh.get_primitives() != nullptr;
                if (has_primitive_indices)
                    for (Vector3ui indices : mesh.get_primitive_iterable())
                        add_triangle(indices.x, indices.y, indices.z);
                else {
                    unsigned int vertex_count = mesh.get_vertex_count();
                    for (unsigned int i = 0; i < vertex_count; i += 3)
                        add_triangle(i, i + 1, i + 2);
                }
            }

            // Upload combined buffers to the GPU
            create_structured_buffer(device, combined_mesh_lights.data(), (unsigned int)combined_mesh_lights.size(), &m_combined_mesh_lights_SRV);
        }
    }
}

} // NS DX11Renderer::Managers