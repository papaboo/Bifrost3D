// DirectX 11 light manager.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Dx11Renderer/Managers/LightManager.h"

#include "Dx11Renderer/Utils.h"

#include "Bifrost/Scene/LightSource.h"

namespace DX11Renderer::Managers {

using namespace Bifrost::Core;
using namespace Bifrost::Math;
using namespace Bifrost::Scene;

Dx11Light light_creation(LightSource light) {

    Dx11Light gpu_light = {};
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

    return gpu_light;
}

LightManager::LightManager(ID3D11Device1& device) {
    m_data.active_count = 0u;
    m_lights = Array<Dx11Light>(MAX_LIGHT_COUNT);
    for (int i = 0; i < m_lights.size(); ++i)
        m_lights[i] = Dx11Light::none();

    THROW_DX11_ERROR(create_constant_buffer(device, sizeof(LightBuffer), &m_lights_buffer));
}

void LightManager::handle_updates(ID3D11DeviceContext1& device_context) {
    if (LightSources::get_changed_lights().is_empty())
        return;

    int max_active_light_sources = LightSources::capacity() - 1;
    if (m_lights.size() < max_active_light_sources) {
        int previous_size = m_lights.size();
        m_lights.resize(max_active_light_sources);
        for (int i = previous_size; i < m_lights.size(); ++i)
            m_lights[i] = Dx11Light::none();
    }

    for (LightSource light : LightSources::get_changed_lights()) {
        auto light_changes = light.get_changes();

        // Accounts for the 0'th light always being invalid by subtracting 1 from the index.
        unsigned int valid_light_index = light.get_ID().get_index() - 1;

        // Clear destroyed light
        if (light_changes.is_set(LightSources::Change::Destroyed))
            m_lights[valid_light_index] = Dx11Light::none();
        else if (light_changes.any_set(LightSources::Change::Created, LightSources::Change::Updated))
            m_lights[valid_light_index] = light_creation(light);
    }

    // Loop over all lights and gather active lights into the DX11 buffer
    m_data.active_count = 0;
    unsigned int max_active_light_count = min(m_lights.size(), LightSources::capacity());
    for (int i = 0; i < max_active_light_count; ++i) {
        if (m_lights[i].get_type() == Dx11Light::None)
            continue;

        m_data.lights[m_data.active_count] = m_lights[i];
        ++m_data.active_count;
    }

    device_context.UpdateSubresource(m_lights_buffer, 0, nullptr, &m_data, 0, 0);
}

} // NS DX11Renderer::Managers