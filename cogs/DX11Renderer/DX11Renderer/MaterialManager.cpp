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

using namespace Cogwheel::Assets;

namespace DX11Renderer {

inline Dx11Material make_dx11material(Material mat) {
    Dx11Material dx11_material;
    dx11_material.tint.x = mat.get_tint().r;
    dx11_material.tint.y = mat.get_tint().g;
    dx11_material.tint.z = mat.get_tint().b;
    dx11_material.tint_texture_index = mat.get_tint_texture_ID();
    dx11_material.roughness = mat.get_roughness();
    dx11_material.specularity = mat.get_specularity() * 0.08f; // See Physically-Based Shading at Disney bottom of page 8 for why we remap.
    dx11_material.metallic = mat.get_metallic();
    dx11_material.coverage = mat.get_coverage();
    dx11_material.coverage_texture_index = mat.get_coverage_texture_ID();
    return dx11_material;
}

MaterialManager::MaterialManager(ID3D11Device1& device, ID3D11DeviceContext1& device_context) {
    // Default material.
    Dx11Material invalid_mat = make_dx11material(Materials::UID::invalid_UID());

    m_materials.resize(128);
    m_materials[0] = invalid_mat;

    create_constant_buffer(device, sizeof(Dx11Material) * 128, &m_constant_buffer);
    D3D11_BOX box;
    box.left = 0; box.right = sizeof(Dx11Material);
    box.front = box.top = 0;
    box.back = box.bottom = 1;
    device_context.UpdateSubresource1(m_constant_buffer, 0, &box, &invalid_mat, sizeof(Dx11Material), sizeof(Dx11Material), D3D11_COPY_NO_OVERWRITE);
}
    
MaterialManager::~MaterialManager() {
    safe_release(&m_constant_buffer);
}

void MaterialManager::handle_updates(ID3D11DeviceContext1& device_context) {
    for (Material mat : Materials::get_changed_materials()) {
        // Just ignore deleted materials. They shouldn't be referenced anyway.
        if (!mat.get_changes().is_set(Materials::Change::Destroyed)) {
            unsigned int material_index = mat.get_ID();

            Dx11Material dx_mat = make_dx11material(mat);

            m_materials[material_index] = dx_mat;

            D3D11_BOX box;
            box.left = material_index * sizeof(Dx11Material); box.right = box.left + sizeof(Dx11Material);
            box.front = box.top = 0;
            box.back = box.bottom = 1;
            device_context.UpdateSubresource1(m_constant_buffer, 0, &box, &dx_mat, sizeof(Dx11Material), sizeof(Dx11Material), D3D11_COPY_NO_OVERWRITE);
        }
    }
}

} // NS DX11Renderer