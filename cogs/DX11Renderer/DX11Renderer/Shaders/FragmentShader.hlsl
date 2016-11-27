// Simple fragment shader.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

cbuffer scene_variables  : register(b0) {
    float4x4 mvp_matrix;
    float4x4 to_world_matrix;
    float4 color;
};

//-----------------------------------------------------------------------------
// Light sources.
//-----------------------------------------------------------------------------

static const float PI = 3.14159265f;

struct LightData {
    float4 type_power;
    float4 spatial_softness;

    float type() { return type_power.x; }

    // Sphere light property accessors.
    float3 sphere_power() { return type_power.yzw; }
    float3 sphere_position() { return spatial_softness.xyz; }
    float sphere_radius() { return spatial_softness.w; }

    // Directinoal light property accessors.
    float3 directional_radiance() { return type_power.yzw; }
    float3 directional_direction() { return spatial_softness.xyz; } // Oh what a concise method name.
};

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

LightSample sample_directional_light(LightData light) {
    LightSample light_sample;
    light_sample.radiance = light.directional_radiance();
    light_sample.direction_to_light = -light.directional_direction();
    light_sample.distance = 1e30f;
    light_sample.angle_subtended = 0.0;
    return light_sample;
}

LightSample sample_light(LightData light, float3 world_position) {
    if (light.type() > 1.5)
        return sample_directional_light(light);
    else if (light.type() > 0.5)
        return sample_sphere_light(light, world_position);

    return LightSample::empty();
}

cbuffer lights : register(b1) {
    int4 light_count;
    LightData light_data[12];
}

struct PixelInput {
    float4 position : SV_POSITION;
    float4 world_pos : WORLD_POSITION;
    float4 normal : NORMAL;
};

float4 main(PixelInput input) : SV_TARGET{
    float3 normal = normalize(input.normal.xyz);

    float3 radiance = float3(0.0, 0.0, 0.0);
    for (int l = 0; l < light_count.x; ++l) {
        LightSample light_sample = sample_light(light_data[l], input.world_pos.xyz);
        radiance += color.rgb * light_sample.radiance * max(0.0, dot(normal, light_sample.direction_to_light));
    }
    return float4(radiance, 1.0);
}