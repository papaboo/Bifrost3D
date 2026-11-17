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
#include "Bifrost/Math/Vector.h"
#include "Bifrost/Utils/IdDeclarations.h"

namespace DX11Renderer::Managers {

//----------------------------------------------------------------------------
// Light manager.
// Collects the light sources into a tight array of active lights and maintains their GPU representation.
// The buffer splits delta and area lights into their own parts of the buffer,
// such that the delta lights are in the first part of the buffer and the area lights in the last.
//----------------------------------------------------------------------------
class LightManager {
    static const unsigned int MAX_LIGHT_COUNT = 12;

private:
    struct LightBuffer {
        int active_delta_light_count;
        int active_total_light_count;
        Bifrost::Math::Vector2i _padding;
        Dx11Light lights[MAX_LIGHT_COUNT];
    } m_data;
    OBuffer m_lights_buffer;

    Bifrost::Core::Array<Dx11Light> m_lights;

public:
    LightManager() {
        m_data.active_total_light_count = 0u;
        m_lights_buffer = nullptr;
    }

    LightManager(ID3D11Device1& device);

    inline int active_light_count() const { return m_data.active_total_light_count; }
    inline ID3D11Buffer** light_buffer_addr() { return &m_lights_buffer; }

    void handle_updates(ID3D11DeviceContext1& device_context);
};

} // NS DX11Renderer::Managers

#endif // _DX11RENDERER_MANAGERS_LIGHT_MANAGER_H_