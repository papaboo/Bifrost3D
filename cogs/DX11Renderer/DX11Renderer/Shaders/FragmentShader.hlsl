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

// TODO Helpful accessors.
struct LightData {
    float4 type_power;
    float4 spatial_softness;
};

struct LightSample {
    float3 radiance;
    float angle_subtended;
    float3 direction_to_light;
    float distance;
};

LightSample sample_sphere_light(LightData light_data, float3 world_position) {
    float radius = light_data.spatial_softness.w;
    float3 direction_to_light = light_data.spatial_softness.xyz - world_position;

    LightSample light_sample;
    light_sample.direction_to_light = direction_to_light;
    light_sample.distance = length(light_sample.direction_to_light);
    light_sample.direction_to_light /= light_sample.distance;
    light_sample.radiance = light_data.type_power.yzw / (4.0f * PI * light_sample.distance * light_sample.distance);
    light_sample.distance -= radius; // Distance to light surface instead of light center.
    light_sample.angle_subtended = radius / light_sample.distance;
    light_sample.angle_subtended *= light_sample.angle_subtended;
    return light_sample;
}

LightSample sample_light(LightData light_data, float3 world_position) {
    if (light_data.type_power.x == 1.0)
        sample_sphere_light(light_data, world_position);

    LightSample light_sample;
    return light_sample;
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
    // return float4(normal * 0.5f + 0.5f, 1.0);
    // return float4(input.world_pos.xyz, 1.0);

    float3 radiance = float3(0.0, 0.0, 0.0);
    for (int l = 0; l < light_count.x; ++l) {
        LightSample light_sample = sample_sphere_light(light_data[l], input.world_pos.xyz);
        radiance += color.rgb * light_sample.radiance * max(0.0, dot(normal, light_sample.direction_to_light));
    }
    return float4(radiance, 1.0);
}