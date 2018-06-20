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
#include <Cogwheel/Assets/Shading/Fittings.h>

using namespace Cogwheel::Assets;

namespace DX11Renderer {

inline Dx11Material make_dx11material(Material mat) {
    Dx11Material dx11_material;
    dx11_material.tint.x = mat.get_tint().r;
    dx11_material.tint.y = mat.get_tint().g;
    dx11_material.tint.z = mat.get_tint().b;
    dx11_material.roughness = mat.get_roughness();
    dx11_material.specularity = mat.get_specularity();
    dx11_material.metallic = mat.get_metallic();
    dx11_material.coverage = mat.get_coverage();
    dx11_material.textures_bound = mat.get_tint_texture_ID() ? TextureBound::Tint : TextureBound::None;
    dx11_material.textures_bound |= mat.get_coverage_texture_ID() ? TextureBound::Coverage : TextureBound::None;
    return dx11_material;
}

inline Dx11MaterialTextures make_dx11material_textures(Material mat) {
    Dx11MaterialTextures dx11_material_textures;
    dx11_material_textures.tint_index = mat.get_tint_texture_ID();
    dx11_material_textures.coverage_index = mat.get_coverage_texture_ID();
    return dx11_material_textures;
}

MaterialManager::MaterialManager(ID3D11Device1& device, ID3D11DeviceContext1& context) {
    using namespace Cogwheel::Assets::Shading;

    { // Setup GGX with fresnel rho texture.
        const unsigned int width = Rho::GGX_with_fresnel_angle_sample_count;
        const unsigned int height = Rho::GGX_with_fresnel_roughness_sample_count;

        unsigned short* rho = new unsigned short[width * height];
        for (unsigned int i = 0; i < width * height; ++i)
            rho[i] = unsigned short(Rho::GGX_with_fresnel[i] * 65535);

        create_texture_2D(device, DXGI_FORMAT_R16_UNORM, rho, width, height, D3D11_USAGE_IMMUTABLE, &m_GGX_with_fresnel_rho_srv);

        delete[] rho;
    }

    #if SPTD_AREA_LIGHTS
    { // Setup GGX SPTD fit texture.
        const unsigned int width = GGX_SPTD_fit_angular_sample_count;
        const unsigned int height = GGX_SPTD_fit_roughness_sample_count;

        R10G10B10A2_Unorm* pivots = new R10G10B10A2_Unorm[width * height];
        for (unsigned int i = 0; i < width * height; ++i)
            pivots[i] = R10G10B10A2_Unorm(GGX_SPTD_fit[i].x, cosf(GGX_SPTD_fit[i].y), GGX_SPTD_fit[i].z); 
        
        create_texture_2D(device, DXGI_FORMAT_R10G10B10A2_UNORM, pivots, width, height, D3D11_USAGE_IMMUTABLE, &m_GGX_SPTD_fit_srv);

        delete[] pivots;
    }
    #else
        m_GGX_SPTD_fit_srv = nullptr;
    #endif

    m_materials.resize(1);
    m_material_textures.resize(1);
    m_constant_array = ConstantBufferArray<Dx11Material>();

    m_materials[0] = make_dx11material(Materials::UID::invalid_UID());
    m_material_textures[0] = make_dx11material_textures(Materials::UID::invalid_UID());
}

void MaterialManager::handle_updates(ID3D11Device1& device, ID3D11DeviceContext1& context) {
    // Resize buffers if needed.
    if (m_materials.size() < Materials::capacity()) {

        // Fill the host side buffer.
        m_materials.resize(Materials::capacity());
        m_material_textures.resize(Materials::capacity());
        for (Material mat : Materials::get_changed_materials())
            // Just ignore deleted materials. They shouldn't be referenced anyway.
            if (!mat.get_changes().is_set(Materials::Change::Destroyed)) {
                unsigned int material_index = mat.get_ID();
                m_materials[material_index] = make_dx11material(mat);
                m_material_textures[material_index] = make_dx11material_textures(mat);
            }

        // Copy the elements to the constant buffer.
        m_constant_array = ConstantBufferArray<Dx11Material>(device, m_materials.data(), Materials::capacity());
    } else {
        // Upload the changed materials.
        for (Material mat : Materials::get_changed_materials()) {
            // Just ignore deleted materials. They shouldn't be referenced anyway.
            if (!mat.get_changes().is_set(Materials::Change::Destroyed)) {
                unsigned int material_index = mat.get_ID();

                Dx11Material dx_mat = make_dx11material(mat);
                m_materials[material_index] = dx_mat;
                m_constant_array.set(&context, dx_mat, material_index);

                m_material_textures[material_index] = make_dx11material_textures(mat);
            }
        }
    }
}

} // NS DX11Renderer