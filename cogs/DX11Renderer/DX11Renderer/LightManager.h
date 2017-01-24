// DirectX 11 light manager.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_LIGHT_MANAGER_H_
#define _DX11RENDERER_RENDERER_LIGHT_MANAGER_H_

#include "Dx11Renderer/Types.h"
#include "Dx11Renderer/Utils.h"

#include "Cogwheel/Core/Array.h"
#include "Cogwheel/Scene/LightSource.h"

namespace DX11Renderer {

using namespace Cogwheel::Core;
using namespace Cogwheel::Math;
using namespace Cogwheel::Scene;

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
    Array<unsigned int> m_ID_to_index;
    Array<LightSources::UID> m_index_to_ID;

    struct LightBuffer {
        int m_active_count;
        Vector3i _padding;
        Dx11Light m_lights[MAX_LIGHTS];
    } m_data;
    ID3D11Buffer* m_lights_buffer;

    inline void light_creation(LightSources::UID light_ID, unsigned int light_index, Dx11Light* gpu_lights) {

        Dx11Light& gpu_light = gpu_lights[light_index];
        switch (LightSources::get_type(light_ID)) {
        case LightSources::Type::Sphere: {
            SphereLight host_light = light_ID;

            gpu_light.flags = Dx11Light::Sphere;

            Vector3f position = host_light.get_node().get_global_transform().translation;
            memcpy(&gpu_light.sphere.position, &position, sizeof(gpu_light.sphere.position));

            RGB power = host_light.get_power();
            memcpy(&gpu_light.sphere.power, &power, sizeof(gpu_light.sphere.power));

            gpu_light.sphere.radius = host_light.get_radius();
            break;
        }
        case LightSources::Type::Directional: {
            DirectionalLight host_light = light_ID;

            gpu_light.flags = Dx11Light::Directional;

            Vector3f direction = host_light.get_node().get_global_transform().rotation.forward();
            memcpy(&gpu_light.directional.direction, &direction, sizeof(gpu_light.directional.direction));

            RGB radiance = host_light.get_radiance();
            memcpy(&gpu_light.directional.radiance, &radiance, sizeof(gpu_light.directional.radiance));
            break;
        }
        }
    }

public:
    LightManager() {
        m_data.m_active_count = 0u;
        m_lights_buffer = nullptr;
    }

    LightManager(ID3D11Device& device, unsigned int initial_capacity) {
        initial_capacity = initial_capacity;
        m_ID_to_index = Array<unsigned int>(initial_capacity);
        m_index_to_ID = Array<LightSources::UID>(initial_capacity);
        m_data.m_active_count = 0u;

        D3D11_BUFFER_DESC uniforms_desc = {};
        uniforms_desc.Usage = D3D11_USAGE_DEFAULT;
        uniforms_desc.ByteWidth = sizeof(LightBuffer);
        uniforms_desc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
        uniforms_desc.CPUAccessFlags = 0;
        uniforms_desc.MiscFlags = 0;

        HRESULT hr = device.CreateBuffer(&uniforms_desc, NULL, &m_lights_buffer);
    }

    void release() {
        safe_release(&m_lights_buffer);
    }

    inline ID3D11Buffer* light_buffer() {
        return m_lights_buffer;
    }

    inline ID3D11Buffer** light_buffer_addr() {
        return &m_lights_buffer;
    }

    void handle_updates(ID3D11DeviceContext& device_context) {
        if (!LightSources::get_changed_lights().is_empty()) {
            if (m_ID_to_index.size() < LightSources::capacity()) {
                // Resize the light buffer to hold the new capacity.
                unsigned int new_capacity = LightSources::capacity();
                m_ID_to_index.resize(new_capacity);
                m_index_to_ID.resize(new_capacity);

                // Resizing removes old data, so this as an opportunity to linearize the light data.
                Dx11Light* gpu_lights = m_data.m_lights;
                unsigned int light_index = 0;
                for (LightSources::UID light_ID : LightSources::get_iterable()) {
                    m_ID_to_index[light_ID] = light_index;
                    m_index_to_ID[light_index] = light_ID;

                    light_creation(light_ID, light_index, gpu_lights);
                    ++light_index;
                }

                m_data.m_active_count = light_index;
            } else {

                Dx11Light* gpu_lights = m_data.m_lights;
                LightSources::ChangedIterator created_lights_begin = LightSources::get_changed_lights().begin();
                while (created_lights_begin != LightSources::get_changed_lights().end() &&
                    LightSources::get_changes(*created_lights_begin).not_set(LightSources::Change::Created))
                    ++created_lights_begin;

                // Process destroyed 
                for (LightSources::UID light_ID : LightSources::get_changed_lights()) {
                    if (LightSources::get_changes(light_ID) != LightSources::Change::Destroyed)
                        continue;

                    unsigned int light_index = m_ID_to_index[light_ID];

                    if (created_lights_begin != LightSources::get_changed_lights().end()) {
                        // Replace deleted light by new light source.
                        LightSources::UID new_light_ID = *created_lights_begin;
                        light_creation(new_light_ID, light_index, gpu_lights);
                        m_ID_to_index[new_light_ID] = light_index;
                        m_index_to_ID[light_index] = new_light_ID;

                        // Find next created light.
                        while (created_lights_begin != LightSources::get_changed_lights().end() &&
                            LightSources::get_changes(*created_lights_begin).not_set(LightSources::Change::Created))
                            ++created_lights_begin;
                    } else {
                        // Replace deleted light by light from the end of the array.
                        --m_data.m_active_count;
                        if (light_index != m_data.m_active_count) {
                            memcpy(gpu_lights + light_index, gpu_lights + m_data.m_active_count, sizeof(Dx11Light));

                            // Rewire light ID and index maps.
                            m_index_to_ID[light_index] = m_index_to_ID[m_data.m_active_count];
                            m_ID_to_index[m_index_to_ID[light_index]] = light_index;
                        }
                    }
                }

                // If there are still lights that needs to be created, then append them to the list.
                for (LightSources::UID light_ID : Iterable<LightSources::ChangedIterator>(created_lights_begin, LightSources::get_changed_lights().end())) {
                    if (LightSources::get_changes(light_ID).not_set(LightSources::Change::Created))
                        continue;

                    unsigned int light_index = m_data.m_active_count++;
                    m_ID_to_index[light_ID] = light_index;
                    m_index_to_ID[light_index] = light_ID;

                    light_creation(light_ID, light_index, gpu_lights);
                }
            }

            device_context.UpdateSubresource(m_lights_buffer, 0, NULL, &m_data, 0, 0);
        }
    }
};

} // DX11Renderer

#endif // _DX11RENDERER_RENDERER_LIGHT_MANAGER_H_