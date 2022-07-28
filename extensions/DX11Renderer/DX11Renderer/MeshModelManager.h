// DirectX 11 mesh model manager.
//-------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_MESH_MODEL_MANAGER_H_
#define _DX11RENDERER_RENDERER_MESH_MODEL_MANAGER_H_

#include "Dx11Renderer/Types.h"

#include <vector>

namespace DX11Renderer {

//-------------------------------------------------------------------------------------------------
// Mesh model manager.
// Holds all models and sorts them according to their properties.
// The models are sorted into the bins opaque, thin-walled opaque, cutouts and transparent models.
//-------------------------------------------------------------------------------------------------
class MeshModelManager {
public:
    typedef std::vector<Dx11Model>::iterator Iterator;
    typedef std::vector<Dx11Model>::const_iterator ConstIterator;

    MeshModelManager();
    MeshModelManager(MeshModelManager&& other) = default;
    MeshModelManager& operator=(MeshModelManager&& rhs) = default;

    void handle_updates();

    inline Iterator begin() { return m_sorted_models.begin() + 1; }
    inline ConstIterator cbegin() const { return m_sorted_models.cbegin() + 1; }
    inline ConstIterator begin() const { cbegin(); }
    inline Iterator end() { return m_sorted_models.end(); }
    inline ConstIterator cend() const { return m_sorted_models.cend(); }
    inline ConstIterator end() const { cend(); }

    inline ConstIterator cbegin_opaque_models() const { return begin(); }
    inline ConstIterator cbegin_opaque_thin_walled_models() const { return m_begin_opaque_thin_walled_models; }
    inline ConstIterator cbegin_cutout_models() const { return m_begin_cutout_models; }
    inline ConstIterator cbegin_transparent_models() const { return m_begin_transparent_models; }

private:
    MeshModelManager(MeshModelManager& other) = delete;
    MeshModelManager& operator=(MeshModelManager& rhs) = delete;

    std::vector<unsigned int> m_model_indices; // The model's entry in the sorted models array.
    std::vector<Dx11Model> m_sorted_models;

    ConstIterator m_begin_opaque_thin_walled_models;
    ConstIterator m_begin_cutout_models;
    ConstIterator m_begin_transparent_models;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_MESH_MODEL_MANAGER_H_