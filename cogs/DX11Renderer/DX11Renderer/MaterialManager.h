// DirectX 11 material manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_MATERIAL_MANAGER_H_
#define _DX11RENDERER_RENDERER_MATERIAL_MANAGER_H_

#include "Dx11Renderer/ConstantBufferArray.h"
#include "Dx11Renderer/Types.h"

#include <vector>

namespace DX11Renderer {

//----------------------------------------------------------------------------
// Material manager.
// Uploads and manages a buffer of material parameters.
// Future work:
// * Support a variable number of materials.
// * Upload buffer with changed materials to the GPU and then scatter them
//   to the constant buffer in a compute shader. Is that possible?
//----------------------------------------------------------------------------
class MaterialManager {
public:

    MaterialManager() = default;
    MaterialManager(ID3D11Device1& device, ID3D11DeviceContext1& context);

    MaterialManager& operator=(MaterialManager&& rhs) = default;

    ID3D11ShaderResourceView** get_GGX_with_fresnel_rho_srv_addr() { return &m_GGX_with_fresnel_rho_srv; }
    ID3D11ShaderResourceView** get_GGX_SPTD_fit_srv_addr() { return &m_GGX_SPTD_fit_srv; }
    ID3D11SamplerState** get_precomputation2D_sampler_addr() { return &m_precomputation2D_sampler; }

    inline Dx11Material& get_material(unsigned int material_index) { return m_materials[material_index]; }
    inline Dx11MaterialTextures& get_material_textures(unsigned int material_index) { return m_material_textures[material_index]; }
    inline ID3D11Buffer** get_constant_buffer_addr() { return m_constant_array.get_buffer_addr(); }
    inline void bind_material(ID3D11DeviceContext1& context, unsigned int slot, unsigned int material_index) { m_constant_array.PS_set(&context, slot, material_index); }

    void handle_updates(ID3D11Device1& device, ID3D11DeviceContext1& context);

private:
    MaterialManager(MaterialManager& other) = delete;
    MaterialManager(MaterialManager&& other) = delete;
    MaterialManager& operator=(MaterialManager& rhs) = delete;

    OID3D11ShaderResourceView m_GGX_with_fresnel_rho_srv;
    OID3D11ShaderResourceView m_GGX_SPTD_fit_srv;
    OID3D11SamplerState m_precomputation2D_sampler;

    std::vector<Dx11Material> m_materials;
    std::vector<Dx11MaterialTextures> m_material_textures;

    ConstantBufferArray<Dx11Material> m_constant_array;
};

} // NS DX11Renderer

#endif // _DX11RENDERER_RENDERER_ENVIRONMENT_MANAGER_H_