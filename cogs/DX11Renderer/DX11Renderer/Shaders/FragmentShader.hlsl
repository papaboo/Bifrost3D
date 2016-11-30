// Simple fragment shader.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "LightSources.hlsl"

cbuffer scene_variables  : register(b0) {
    float4x4 mvp_matrix;
    float4x3 to_world_matrix;
    float4 color;
    int4 flags;
};

cbuffer lights : register(b1) {
    int4 light_count;
    LightData light_data[12];
}

struct PixelInput {
    float4 position : SV_POSITION;
    float4 world_pos : WORLD_POSITION;
    float4 normal : NORMAL;
    float2 texcoord : TEXCOORD;
};

float4 main(PixelInput input) : SV_TARGET{
    float3 normal = normalize(input.normal.xyz);

    float3 f = flags.x == 1 ? input.texcoord.xyy : color.rgb;

    float3 radiance = float3(0.0, 0.0, 0.0);
    for (int l = 0; l < light_count.x; ++l) {
        LightSample light_sample = sample_light(light_data[l], input.world_pos.xyz);
        radiance += f * light_sample.radiance * max(0.0, dot(normal, light_sample.direction_to_light));
    }
    return float4(radiance, 1.0);
}