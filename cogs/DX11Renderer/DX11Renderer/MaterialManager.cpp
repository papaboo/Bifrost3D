// DirectX 11 material manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Dx11Renderer/MaterialManager.h"
#include "Dx11Renderer/Utils.h"

#include <Cogwheel/Assets/Material.h>
#include <Cogwheel/Assets/Shading/GGXWithFresnelRho.h>

using namespace Cogwheel::Assets;

namespace DX11Renderer {

inline Dx11Material make_dx11material(Material mat) {
    Dx11Material dx11_material;
    dx11_material.tint.x = mat.get_tint().r;
    dx11_material.tint.y = mat.get_tint().g;
    dx11_material.tint.z = mat.get_tint().b;
    dx11_material.tint_texture_index = mat.get_tint_texture_ID();
    dx11_material.roughness = mat.get_roughness();
    dx11_material.specularity = mat.get_specularity();
    dx11_material.metallic = mat.get_metallic();
    dx11_material.coverage = mat.get_coverage();
    dx11_material.coverage_texture_index = mat.get_coverage_texture_ID();
    return dx11_material;
}

MaterialManager::MaterialManager(ID3D11Device1& device, ID3D11DeviceContext1& context) {

    { // Setup GGX with fresnel rho texture.
        using namespace Cogwheel::Assets::Shading;

        D3D11_TEXTURE2D_DESC tex_desc = {};
        tex_desc.Width = GGX_with_fresnel_angle_sample_count;
        tex_desc.Height = GGX_with_fresnel_roughness_sample_count;
        tex_desc.MipLevels = 1;
        tex_desc.ArraySize = 1;
        tex_desc.Format = DXGI_FORMAT_R16_UNORM;
        tex_desc.SampleDesc.Count = 1;
        tex_desc.SampleDesc.Quality = 0;
        tex_desc.Usage = D3D11_USAGE_IMMUTABLE;
        tex_desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

        unsigned short* rho = new unsigned short[tex_desc.Width * tex_desc.Height];
        for (unsigned int i = 0; i < tex_desc.Width * tex_desc.Height; ++i)
            rho[i] = unsigned short(GGX_with_fresnel_rho[i] * 65536);

        D3D11_SUBRESOURCE_DATA resource_data;
        resource_data.pSysMem = rho;
        resource_data.SysMemPitch = sizeof_dx_format(tex_desc.Format) * tex_desc.Width;

        UID3D11Texture2D GGX_with_fresnel_rho_texture;
        HRESULT hr = device.CreateTexture2D(&tex_desc, &resource_data, &GGX_with_fresnel_rho_texture);
        THROW_ON_FAILURE(hr);

        delete[] rho;

        D3D11_SHADER_RESOURCE_VIEW_DESC srv_desc;
        srv_desc.Format = tex_desc.Format;
        srv_desc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        srv_desc.Texture2D.MipLevels = tex_desc.MipLevels;
        srv_desc.Texture2D.MostDetailedMip = 0;
        hr = device.CreateShaderResourceView(GGX_with_fresnel_rho_texture, &srv_desc, &m_GGX_with_fresnel_rho_srv);
        THROW_ON_FAILURE(hr);
    }

    { // Rho sampler
        D3D11_SAMPLER_DESC desc = {};
        desc.Filter = D3D11_FILTER_MIN_POINT_MAG_LINEAR_MIP_POINT;
        desc.AddressU = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressV = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.AddressW = D3D11_TEXTURE_ADDRESS_CLAMP;
        desc.ComparisonFunc = D3D11_COMPARISON_NEVER;
        desc.MinLOD = 0;
        desc.MaxLOD = D3D11_FLOAT32_MAX;

        HRESULT hr = device.CreateSamplerState(&desc, &m_rho_sampler);
        THROW_ON_FAILURE(hr);
    }

    // Default material.
    Dx11Material invalid_mat = make_dx11material(Materials::UID::invalid_UID());

    m_materials.resize(128);
    m_materials[0] = invalid_mat;

    m_constant_array = ConstantBufferArray<Dx11Material>(&device, 128);

    m_constant_array.set(&context, invalid_mat, 0, D3D11_COPY_DISCARD);
}

void MaterialManager::handle_updates(ID3D11DeviceContext1& context) {
    for (Material mat : Materials::get_changed_materials()) {
        // Just ignore deleted materials. They shouldn't be referenced anyway.
        if (!mat.get_changes().is_set(Materials::Change::Destroyed)) {

            Dx11Material dx_mat = make_dx11material(mat);

            unsigned int material_index = mat.get_ID();
            m_materials[material_index] = dx_mat;
            m_constant_array.set(&context, dx_mat, material_index);
        }
    }
}

} // NS DX11Renderer