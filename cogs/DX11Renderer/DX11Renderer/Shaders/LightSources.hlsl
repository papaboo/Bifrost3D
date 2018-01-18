// Light sources.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#ifndef _DX11_RENDERER_SHADERS_LIGHTSOURCES_H_
#define _DX11_RENDERER_SHADERS_LIGHTSOURCES_H_

#include "Utils.hlsl"

//-----------------------------------------------------------------------------
// Light sources data.
//-----------------------------------------------------------------------------

namespace LightType {
    static const float Sphere = 1.0f;
    static const float Directional = 2.0f;
}

struct LightData {
    float4 type_power;
    float4 spatial_softness;

    float type() { return type_power.x; }

    // Sphere light property accessors.
    float3 sphere_power() { return type_power.yzw; }
    float3 sphere_position() { return spatial_softness.xyz; }
    float sphere_radius() { return spatial_softness.w; }

    // Directional light property accessors.
    float3 directional_radiance() { return type_power.yzw; }
    float3 directional_direction() { return spatial_softness.xyz; } // Oh what a concise method name.
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
// Sphere light sampling.
//-----------------------------------------------------------------------------

static const float sphere_light_small_sin_theta_squared = 1e-5f;

// Returns true if the sphere light should be interpreted as a delta light / point light.
// Ideally this should only happen if the radius is zero, but due to floating point 
// imprecission when sampling cones, we draw the line at very tiny subtended angles.
bool is_delta_sphere_light(LightData light, float3 world_position) {
    float3 vector_to_light = light.sphere_position() - world_position;
    float sin_theta_squared = light.sphere_radius() * light.sphere_radius() / dot(vector_to_light, vector_to_light);
    return sin_theta_squared < sphere_light_small_sin_theta_squared;
}

float sphere_surface_area(float radius) {
    return 4.0f * PI * radius * radius;
}

float3 evaluate_sphere_light(LightData light, float3 world_position) {
    float inv_divisor = rcp(is_delta_sphere_light(light, world_position) ? (4.0f * PI) : (PI * sphere_surface_area(light.sphere_radius())));
    return light.sphere_power() * inv_divisor;
}

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
// Generalized light sampling.
//-----------------------------------------------------------------------------

LightSample sample_light(LightData light, float3 world_position) {
    if (light.type() > 1.5)
        return sample_directional_light(light);
    else if (light.type() > 0.5)
        return sample_sphere_light(light, world_position);

    return LightSample::empty();
}

#endif // _DX11_RENDERER_SHADERS_LIGHTSOURCES_H_