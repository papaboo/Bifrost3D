// DirectX 11 light manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11RENDERER_RENDERER_LIGHT_MANAGER_H_
#define _DX11RENDERER_RENDERER_LIGHT_MANAGER_H_

#include "Dx11Renderer/Types.h"
#include "Dx11Renderer/Utils.h"

#include "Bifrost/Core/Array.h"
#include "Bifrost/Scene/LightSource.h"

namespace DX11Renderer {

using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

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
        int active_count;
        Vector3i _padding;
        Dx11Light lights[MAX_LIGHTS];
    } m_data;
    OBuffer m_lights_buffer;

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
        case LightSources::Type::Spot: {
            SpotLight host_light = light_ID;
            Transform global_transform = host_light.get_node().get_global_transform();

            gpu_light.flags = Dx11Light::Spot;

            Vector3f position = global_transform.translation;
            memcpy(&gpu_light.spot.position, &position, sizeof(gpu_light.sphere.position));

            RGB power = host_light.get_power();
            memcpy(&gpu_light.spot.power, &power, sizeof(gpu_light.sphere.power));

            gpu_light.spot.radius = host_light.get_radius();

            gpu_light.spot.cos_angle = host_light.get_cos_angle();

            Vector3f direction = global_transform.rotation.forward();
            memcpy(&gpu_light.spot.direction, &direction, sizeof(gpu_light.spot.direction));

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
        m_data.active_count = 0u;
        m_lights_buffer = nullptr;
    }

    LightManager(ID3D11Device1& device, unsigned int initial_capacity) {
        initial_capacity = initial_capacity;
        m_ID_to_index = Array<unsigned int>(initial_capacity);
        m_index_to_ID = Array<LightSources::UID>(initial_capacity);
        m_data.active_count = 0u;

        THROW_DX11_ERROR(create_constant_buffer(device, sizeof(LightBuffer), &m_lights_buffer));
    }

    inline int active_light_count() const { return m_data.active_count; }
    inline ID3D11Buffer** light_buffer_addr() { return &m_lights_buffer; }

    void handle_updates(ID3D11DeviceContext1& device_context) {
        if (!LightSources::get_changed_lights().is_empty()) {
            if (m_ID_to_index.size() < LightSources::capacity()) {
                // Resize the light buffer to hold the new capacity.
                unsigned int new_capacity = LightSources::capacity();
                m_ID_to_index.resize(new_capacity);
                m_index_to_ID.resize(new_capacity);

                // Resizing removes old data, so this as an opportunity to linearize the light data.
                Dx11Light* gpu_lights = m_data.lights;
                unsigned int light_index = 0;
                for (LightSources::UID light_ID : LightSources::get_iterable()) {
                    m_ID_to_index[light_ID] = light_index;
                    m_index_to_ID[light_index] = light_ID;

                    light_creation(light_ID, light_index, gpu_lights);
                    ++light_index;
                }

                m_data.active_count = light_index;
            } else {

                Dx11Light* gpu_lights = m_data.lights;

                auto destroy_light = [](LightSources::Changes changes) -> bool {
                    return changes.is_set(LightSources::Change::Destroyed) && changes.not_set(LightSources::Change::Created);
                };

                // First process destroyed lights to ensure that we don't allocate lights and then afterwards adds holes to the light array.
                for (LightSources::UID light_ID : LightSources::get_changed_lights()) {
                    if (!destroy_light(LightSources::get_changes(light_ID)))
                        continue;

                    unsigned int light_index = m_ID_to_index[light_ID];
                    // Replace deleted light by light from the end of the array.
                    --m_data.active_count;
                    if (light_index != m_data.active_count) {
                        memcpy(gpu_lights + light_index, gpu_lights + m_data.active_count, sizeof(Dx11Light));

                        // Rewire light ID and index maps.
                        m_index_to_ID[light_index] = m_index_to_ID[m_data.active_count];
                        m_ID_to_index[m_index_to_ID[light_index]] = light_index;
                    }
                }

                // Then update or create the rest of the light sources.
                for (LightSources::UID light_ID : LightSources::get_changed_lights()) {
                    auto light_changes = LightSources::get_changes(light_ID);
                    if (destroy_light(light_changes))
                        continue;

                    if (light_changes.is_set(LightSources::Change::Created)) {
                        unsigned int light_index = m_data.active_count++;
                        m_ID_to_index[light_ID] = light_index;
                        m_index_to_ID[light_index] = light_ID;

                        light_creation(light_ID, light_index, gpu_lights);
                    } else if (light_changes.is_set(LightSources::Change::Updated)) {
                        unsigned int light_index = m_ID_to_index[light_ID];
                        light_creation(light_ID, light_index, gpu_lights);
                    }
                }
            }

            device_context.UpdateSubresource(m_lights_buffer, 0, nullptr, &m_data, 0, 0);
        }
    }
};

} // DX11Renderer

#endif // _DX11RENDERER_RENDERER_LIGHT_MANAGER_H_