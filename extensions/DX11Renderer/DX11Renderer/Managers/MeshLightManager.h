// DirectX 11 mesh light manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_MANAGERS_MESH_LIGHT_MANAGER_H_
#define _DX11RENDERER_MANAGERS_MESH_LIGHT_MANAGER_H_

#include "Dx11Renderer/Types.h"

#include "Bifrost/Math/Color.h"
#include "Bifrost/Math/Vector.h"
#include "Bifrost/Utils/IdDeclarations.h"

#include <map>

namespace DX11Renderer::Managers {

//----------------------------------------------------------------------------
// Mesh light manager.
// Collects the mesh light sources into buffers of active lights and maintains their GPU representation.
//----------------------------------------------------------------------------
class MeshLightManager {
private:
    struct MeshLight {
        Bifrost::Assets::MeshModelID model_ID;
        Bifrost::Assets::MaterialID material_ID;
        Bifrost::Math::RGB material_emission;
        bool is_thinwalled;
        unsigned int triangle_count;

        MeshLight() = default;
        MeshLight(Bifrost::Assets::MeshModelID model_ID, Bifrost::Assets::MaterialID material_ID, Bifrost::Math::RGB material_emission, bool is_thinwalled, unsigned int triangle_count)
            : model_ID(model_ID), material_ID(material_ID), material_emission(material_emission), is_thinwalled(is_thinwalled), triangle_count(triangle_count) { }
    };

    struct MeshLightGPU {
        float3 position0, position1, position2;
        float3 emission0, emission1, emission2;
        unsigned int is_thinwalled;
        unsigned int __128bit_alignment_padding;
    };

private:

    std::map<unsigned int, MeshLight> m_lights; // MeshModel ID index to mesh light

    unsigned int m_emissive_triangle_count;
    OShaderResourceView m_combined_mesh_lights_SRV;

public:
    MeshLightManager() {
        m_lights = std::map<unsigned int, MeshLight>();
        m_emissive_triangle_count = 0;
        m_combined_mesh_lights_SRV = nullptr;
    }

    void handle_updates(ID3D11Device1& device);

    inline unsigned int get_emissive_triangle_count() const { return m_emissive_triangle_count; }
    inline bool has_mesh_light(Bifrost::Assets::MeshModelID model_ID) const { return m_lights.find(model_ID.get_index()) != m_lights.end(); }

    inline const OShaderResourceView& get_combined_mesh_lights_SRV() const { return m_combined_mesh_lights_SRV; }
};

} // NS DX11Renderer::Managers

#endif // _DX11RENDERER_MANAGERS_MESH_LIGHT_MANAGER_H_