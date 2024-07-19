// Light sources.
// ---------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_LIGHTSOURCES_H_
#define _DX11_RENDERER_SHADERS_LIGHTSOURCES_H_

#include "Utils.hlsl"

//-----------------------------------------------------------------------------
// Sphere light specialization
//-----------------------------------------------------------------------------

struct SphereLight {
    float3 position;
    float radius;
    float3 power;

    static SphereLight make(float3 position, float radius, float3 power) {
        SphereLight light = { position, radius, power };
        return light;
    }

    // Evaluate the radiance of samples emitted from the sphere light source.
    float3 radiance() {
        float divisor = 4.0 * pow2(PI * radius); // PI * surface area of sphere
        return power * rcp(divisor);
    }
};

//-----------------------------------------------------------------------------
// Light sources data.
//-----------------------------------------------------------------------------

namespace LightType {
    static const float Sphere = 1.0f;
    static const float Directional = 2.0f;
    static const float Spot = 3.0f;
}

struct LightData {
    float4 type_power;
    float4 spatial_softness;
    float4 spatial_softness2;

    float type() { return type_power.x; }

    // Sphere light property accessors.
    float3 sphere_power() { return type_power.yzw; }
    float3 sphere_position() { return spatial_softness.xyz; }
    float sphere_radius() { return spatial_softness.w; }
    SphereLight sphere_light() { return SphereLight::make(sphere_position(), sphere_radius(), sphere_power()); }

    // Directional light property accessors.
    float3 directional_radiance() { return type_power.yzw; }
    float3 directional_direction() { return spatial_softness.xyz; } // Oh what a concise method name.

    // Spot light property accessors.
    float3 spot_power() { return type_power.yzw; }
    float3 spot_position() { return spatial_softness.xyz; }
    float spot_radius() { return spatial_softness.w; }
    float spot_surface_area() { return PI * pow2(spot_radius()); }
    float3 spot_direction() { return spatial_softness2.xyz; }
    float spot_cos_angle() { return spatial_softness2.w; }
};

//-----------------------------------------------------------------------------
// Light sample.
//-----------------------------------------------------------------------------

struct LightSample {
    float3 radiance;
    float angle_subtended;
    float3 direction_to_light;
    float distance;

    static LightSample empty() {
        LightSample light_sample;
        light_sample.radiance = float3(0, 0, 0);
        light_sample.angle_subtended = 0.0;
        light_sample.direction_to_light = float3(0, 1, 0);
        light_sample.distance = 1e30f;
        return light_sample;
    }
};

//-----------------------------------------------------------------------------
// Directional light sampling.
//-----------------------------------------------------------------------------

LightSample sample_directional_light(LightData light) {
    LightSample light_sample;
    light_sample.radiance = light.directional_radiance();
    light_sample.direction_to_light = -light.directional_direction();
    light_sample.distance = 1e30f;
    light_sample.angle_subtended = 0.0;
    return light_sample;
}

//-----------------------------------------------------------------------------
// Sphere light sampling.
//-----------------------------------------------------------------------------

LightSample sample_sphere_light(LightData light, float3 world_position) {
    float3 direction_to_light = light.sphere_position() - world_position;

    LightSample light_sample;
    light_sample.direction_to_light = direction_to_light;
    light_sample.distance = length(light_sample.direction_to_light);
    light_sample.direction_to_light /= light_sample.distance;
    light_sample.radiance = light.sphere_power() / (4.0f * PI * light_sample.distance * light_sample.distance);
    light_sample.distance -= light.sphere_radius(); // Distance to light surface instead of light center.
    light_sample.angle_subtended = light.sphere_radius() / light_sample.distance;
    light_sample.angle_subtended *= light_sample.angle_subtended;
    return light_sample;
}

//-----------------------------------------------------------------------------
// Spot light sampling.
//-----------------------------------------------------------------------------

LightSample sample_spot_light(LightData light, float3 world_position) {
    float3 direction_to_light = light.spot_position() - world_position;

    LightSample light_sample;
    light_sample.direction_to_light = direction_to_light;
    light_sample.distance = length(light_sample.direction_to_light);
    light_sample.direction_to_light /= light_sample.distance;

    float cos_theta = max(0, -dot(light.spot_direction(), light_sample.direction_to_light));
    float normalization = 2 * PI * (1 - light.spot_cos_angle());
    light_sample.radiance = light.spot_power() / (normalization * pow2(light_sample.distance));
    light_sample.radiance *= cos_theta > light.spot_cos_angle() ? 1.0 : 0.0;

    light_sample.angle_subtended = (light.spot_surface_area() * cos_theta) / light_sample.distance;
    return light_sample;
}

//-----------------------------------------------------------------------------
// Generalized light sampling.
//-----------------------------------------------------------------------------

LightSample sample_light(LightData light, float3 world_position) {
    if (light.type() > 2.5)
        return sample_spot_light(light, world_position);
    else if (light.type() > 1.5)
        return sample_directional_light(light);
    else if (light.type() > 0.5)
        return sample_sphere_light(light, world_position);
    else
        return LightSample::empty();
}

#endif // _DX11_RENDERER_SHADERS_LIGHTSOURCES_H_