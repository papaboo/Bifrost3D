// DirectX 11 material manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_MANAGERS_MATERIAL_MANAGER_H_
#define _DX11RENDERER_MANAGERS_MATERIAL_MANAGER_H_

#include "Dx11Renderer/ConstantBufferArray.h"
#include "Dx11Renderer/Types.h"

#include <vector>

namespace DX11Renderer::Managers {

//----------------------------------------------------------------------------
// Material manager.
// Uploads and manages a buffer of material parameters.
//----------------------------------------------------------------------------
class MaterialManager {
public:

    MaterialManager() = default;
    MaterialManager(ID3D11Device1& device, ID3D11DeviceContext1& context);
    MaterialManager(MaterialManager&& other) = default;
    MaterialManager& operator=(MaterialManager&& rhs) = default;

    inline ID3D11ShaderResourceView** get_GGX_with_fresnel_rho_srv_addr() { return &m_GGX_with_fresnel_rho_srv; }

    inline Dx11Material& get_material(unsigned int material_index) { return m_materials[material_index]; }
    inline Dx11MaterialTextures& get_material_textures(unsigned int material_index) { return m_material_textures[material_index]; }
    inline void bind_material(ID3D11DeviceContext1& context, unsigned int slot, unsigned int material_index) { 
        context.PSSetConstantBuffers(slot, 1, &m_GPU_materials[material_index]);
    }

    void handle_updates(ID3D11Device1& device, ID3D11DeviceContext1& context);

    static OShaderResourceView create_GGX_with_fresnel_rho_srv(ID3D11Device1& device);

private:
    MaterialManager(MaterialManager& other) = delete;
    MaterialManager& operator=(MaterialManager& rhs) = delete;

    OShaderResourceView m_GGX_with_fresnel_rho_srv;

    std::vector<Dx11Material> m_materials;
    std::vector<Dx11MaterialTextures> m_material_textures;

    std::vector<OBuffer> m_GPU_materials;
};

} // NS DX11Renderer::Managers

#endif // _DX11RENDERER_MANAGERS_ENVIRONMENT_MANAGER_H_