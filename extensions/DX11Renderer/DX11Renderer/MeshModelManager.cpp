// DirectX 11 mesh model manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Dx11Renderer/MeshModelManager.h"

#include <Bifrost/Assets/Material.h>
#include <Bifrost/Assets/MeshModel.h>

using namespace Bifrost::Assets;

namespace DX11Renderer {

MeshModelManager::MeshModelManager() {
    m_sorted_models = std::vector<Dx11Model>(1);
    m_sorted_models.reserve(MeshModels::capacity());
    m_sorted_models[0] = {};

    m_model_indices = std::vector<unsigned int>(MeshModels::capacity());
}

inline unsigned int model_properties_from_material(Material material) {
    bool uses_coverage = material.get_coverage_texture_ID() != TextureID::invalid_UID() || material.get_coverage() < 1.0f;
    unsigned int coverage_type = material.is_cutout() ? Dx11Model::Properties::Cutout : Dx11Model::Properties::Transparent;
    unsigned int properties = uses_coverage ? coverage_type : Dx11Model::Properties::None;

    properties |= material.is_thin_walled() ? Dx11Model::Properties::ThinWalled : Dx11Model::Properties::None;

    return properties;
}

void MeshModelManager::handle_updates() {
    bool models_updated = false;

    if (!MeshModels::get_changed_models().is_empty()) {
        if (m_sorted_models.size() <= MeshModels::capacity()) {
            m_sorted_models.reserve(MeshModels::capacity());
            int old_size = (int)m_model_indices.size();
            m_model_indices.resize(MeshModels::capacity());
            std::fill(m_model_indices.begin() + old_size, m_model_indices.end(), 0);
        }

        for (MeshModel model : MeshModels::get_changed_models()) {
            unsigned int model_index = m_model_indices[model.get_ID()];

            if (model.get_changes().contains(MeshModels::Change::Destroyed)) {
                Dx11Model dx_model = {};
                m_sorted_models[model_index] = dx_model;

                m_model_indices[model.get_ID()] = 0;
            } else if (model.get_changes().contains(MeshModels::Change::Created)) {
                Dx11Model dx_model;
                dx_model.model_ID = model.get_ID();
                dx_model.material_ID = model.get_material().get_ID();
                dx_model.mesh_ID = model.get_mesh().get_ID();
                dx_model.transform_ID = model.get_scene_node().get_ID();
                dx_model.properties = model_properties_from_material(model.get_material());

                if (model_index == 0) {
                    m_model_indices[model.get_ID()] = (unsigned int)m_sorted_models.size();
                    m_sorted_models.push_back(dx_model);
                } else
                    m_sorted_models[model_index] = dx_model;
            } else if (model.get_changes().contains(MeshModels::Change::Material)) {
                Dx11Model& dx_model = m_sorted_models[model_index];
                dx_model.material_ID = model.get_material().get_ID();
                dx_model.properties = model_properties_from_material(model.get_material());
            }
        }

        models_updated = true;
    }

    // Check if any materials have changed.
    if (!Materials::get_changed_materials().is_empty()) {
        // This rarely happens, so for now we just loop over all models
        for (Iterator model_itr = begin(); model_itr != end(); ++model_itr) {
            // Ignore invalid models.
            if (model_itr->model_ID == 0)
                continue;

            // Update material properties and set models_updated to true in case there was a change.
            Material material = model_itr->material_ID;
            unsigned int model_props = model_properties_from_material(material);
            models_updated |= model_itr->properties != model_props;
            model_itr->properties = model_props;
        }
    }

    if (models_updated) {
        // Sort the models in the order [dummy, opaque backface-culled, opaque thin-walled, cutout, transparent, destroyed]
        // The models to be sorted starts at index 1, because the first model is a dummy model.
        std::sort(m_sorted_models.begin() + 1, m_sorted_models.end(),
            [](Dx11Model lhs, Dx11Model rhs) -> bool {
                return lhs.properties < rhs.properties;
            });

        // Register the models new position and find the transition between model buckets.
        m_begin_opaque_thin_walled_models = m_sorted_models.end();
        m_begin_cutout_models = m_sorted_models.end();
        m_begin_transparent_models = m_sorted_models.end();
        size_t model_count = m_sorted_models.size();

        #pragma omp parallel for
        for (int i = 1; i < m_sorted_models.size(); ++i) {
            Dx11Model& model = m_sorted_models[i];
            m_model_indices[model.model_ID] = i;

            Dx11Model& prevModel = m_sorted_models[i - 1];
            if (prevModel.properties != model.properties) {
                bool thin_walled_transition = !prevModel.is_thin_walled() && model.is_thin_walled();

                if (model.is_opaque() && thin_walled_transition)
                    m_begin_opaque_thin_walled_models = m_sorted_models.begin() + i;
                if (!prevModel.is_cutout() && model.is_cutout())
                    m_begin_cutout_models = m_sorted_models.begin() + i;
                if (!prevModel.is_transparent() && model.is_transparent())
                    m_begin_transparent_models = m_sorted_models.begin() + i;
                if (!prevModel.is_destroyed() && model.is_destroyed())
                    model_count = i;
            }
        }

        // Correct interators in case no transition between buckets was found.
        if (m_begin_transparent_models > m_sorted_models.begin() + model_count)
            m_begin_transparent_models = m_sorted_models.begin() + model_count;
        if (m_begin_cutout_models > m_begin_transparent_models)
            m_begin_cutout_models = m_begin_transparent_models;
        if (m_begin_opaque_thin_walled_models > m_begin_cutout_models)
            m_begin_opaque_thin_walled_models = m_begin_cutout_models;

        // Compact the list of sorted mesh models by removing destroyed models.
        m_sorted_models.resize(model_count);
    }
}

} // NS DX11Renderer