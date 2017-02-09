// DirectX 11 material manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_MATERIAL_MANAGER_H_
#define _DX11RENDERER_RENDERER_MATERIAL_MANAGER_H_

#include "Dx11Renderer/Types.h"

#include <vector>

namespace DX11Renderer {

//----------------------------------------------------------------------------
// Material manager.
// Uploads and manages a buffer of material parameters.
// Future work:
// * Support a variable number of materials.
// * Upload buffer with changed materials to the GPU and then scatter them
//   to the constant buffer in a compute shader.
//----------------------------------------------------------------------------
class MaterialManager {
private:

    std::vector<Dx11Material> m_materials;

    ID3D11Buffer* m_constant_buffer;
    
public:

    MaterialManager() : m_constant_buffer(nullptr) {}
    MaterialManager(ID3D11Device1& device, ID3D11DeviceContext1& device_context);
    MaterialManager(MaterialManager&& other)
        : m_constant_buffer(other.m_constant_buffer) {
        other.m_constant_buffer = nullptr;
        m_materials = std::move(other.m_materials);
    }
    MaterialManager& operator=(MaterialManager&& rhs) {
        m_constant_buffer = rhs.m_constant_buffer;
        rhs.m_constant_buffer = nullptr;
        m_materials = std::move(rhs.m_materials);
        return *this;
    }
    
    ~MaterialManager();

    Dx11Material& get_material(unsigned int index) { return m_materials[index]; }
    ID3D11Buffer** get_constant_buffer_addr() { return &m_constant_buffer; }

    void handle_updates(ID3D11DeviceContext1& device_context);

private:
    MaterialManager(MaterialManager& other) = delete;
    MaterialManager& operator=(MaterialManager& rhs) = delete;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_ENVIRONMENT_MANAGER_H_