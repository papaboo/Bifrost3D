// DirectX 11 light manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Dx11Renderer/Managers/LightManager.h"

#include "Dx11Renderer/Utils.h"

namespace DX11Renderer::Managers {

using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

void light_creation(LightSource light, unsigned int light_index, Dx11Light* gpu_lights) {

    Dx11Light& gpu_light = gpu_lights[light_index];
    switch (light.get_type()) {
    case LightSources::Type::Sphere: {
        SphereLight host_light = light;

        gpu_light.flags = Dx11Light::Sphere;

        Vector3f position = host_light.get_node().get_global_transform().translation;
        memcpy(&gpu_light.sphere.position, &position, sizeof(gpu_light.sphere.position));

        RGB power = host_light.get_power();
        memcpy(&gpu_light.sphere.power, &power, sizeof(gpu_light.sphere.power));

        gpu_light.sphere.radius = host_light.get_radius();
        break;
    }
    case LightSources::Type::Spot: {
        SpotLight host_light = light;
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
        DirectionalLight host_light = light;

        gpu_light.flags = Dx11Light::Directional;

        Vector3f direction = host_light.get_node().get_global_transform().rotation.forward();
        memcpy(&gpu_light.directional.direction, &direction, sizeof(gpu_light.directional.direction));

        RGB radiance = host_light.get_radiance();
        memcpy(&gpu_light.directional.radiance, &radiance, sizeof(gpu_light.directional.radiance));
        break;
    }
    }
}

LightManager::LightManager(ID3D11Device1& device, unsigned int initial_capacity) {
    initial_capacity = initial_capacity;
    m_ID_to_index = Array<unsigned int>(initial_capacity);
    m_index_to_ID = Array<LightSourceID>(initial_capacity);
    m_data.active_count = 0u;

    THROW_DX11_ERROR(create_constant_buffer(device, sizeof(LightBuffer), &m_lights_buffer));
}

void LightManager::handle_updates(ID3D11DeviceContext1& device_context) {
    if (LightSources::get_changed_lights().is_empty())
        return;

    if (m_ID_to_index.size() < LightSources::capacity()) {
        // Resize the light buffer to hold the new capacity.
        unsigned int new_capacity = LightSources::capacity();
        m_ID_to_index.resize(new_capacity);
        m_index_to_ID.resize(new_capacity);

        // Resizing removes old data, so this as an opportunity to linearize the light data.
        Dx11Light* gpu_lights = m_data.lights;
        unsigned int light_index = 0;
        for (LightSourceID light_ID : LightSources::get_iterable()) {
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
        for (LightSourceID light_ID : LightSources::get_changed_lights()) {
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
        for (LightSourceID light_ID : LightSources::get_changed_lights()) {
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

} // NS DX11Renderer::Managers