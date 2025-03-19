// DirectX 11 light manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_MANAGERS_LIGHT_MANAGER_H_
#define _DX11RENDERER_MANAGERS_LIGHT_MANAGER_H_

#include "Dx11Renderer/Types.h"

#include "Bifrost/Core/Array.h"
#include "Bifrost/Scene/LightSource.h"

namespace DX11Renderer::Managers {

//----------------------------------------------------------------------------
// Light manager.
// Collects the light sources into a tight array of active lights 
// and maintains their GPU representation.
// Future work.
// * Support an arbitrary light count. (Maximum of what fits in a constant buffer would work)
// * Support tiled forward rendering.
//----------------------------------------------------------------------------
class LightManager {
    static const unsigned int MAX_LIGHTS = 12;

private:
    Bifrost::Core::Array<unsigned int> m_ID_to_index;
    Bifrost::Core::Array<Bifrost::Scene::LightSourceID> m_index_to_ID;

    struct LightBuffer {
        int active_count;
        Bifrost::Math::Vector3i _padding;
        Dx11Light lights[MAX_LIGHTS];
    } m_data;
    OBuffer m_lights_buffer;

public:
    LightManager() {
        m_data.active_count = 0u;
        m_lights_buffer = nullptr;
    }

    LightManager(ID3D11Device1& device, unsigned int initial_capacity);

    inline int active_light_count() const { return m_data.active_count; }
    inline ID3D11Buffer** light_buffer_addr() { return &m_lights_buffer; }

    void handle_updates(ID3D11DeviceContext1& device_context);
};

} // NS DX11Renderer::Managers

#endif // _DX11RENDERER_MANAGERS_LIGHT_MANAGER_H_