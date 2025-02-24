// DirectX 11 mesh manager.
//-------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
//-------------------------------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_MESH_MANAGER_H_
#define _DX11RENDERER_RENDERER_MESH_MANAGER_H_

#include "Dx11Renderer/Types.h"

#include <vector>

namespace DX11Renderer {

//-------------------------------------------------------------------------------------------------
// Mesh manager.
// Manages the buffers for the meshes, such as constant buffers, index buffers and vertex buffers.
//-------------------------------------------------------------------------------------------------
class MeshManager {
public:
    MeshManager(ID3D11Device1& device);
    MeshManager(MeshManager&& other) = default;
    MeshManager& operator=(MeshManager&& rhs) = default;

    ~MeshManager();

    inline const Dx11Mesh& get_mesh(unsigned int mesh_index) const { return m_meshes[mesh_index]; }
    inline const OBuffer& get_null_buffer() const { return m_null_buffer; }

    void handle_updates(ID3D11Device1& device);

private:
    MeshManager(MeshManager& other) = delete;
    MeshManager& operator=(MeshManager& rhs) = delete;

    OBuffer m_null_buffer;
    std::vector<Dx11Mesh> m_meshes;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_MESH_MANAGER_H_