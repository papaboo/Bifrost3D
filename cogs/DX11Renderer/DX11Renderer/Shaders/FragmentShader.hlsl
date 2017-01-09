// Simple fragment shader.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "DefaultShading.hlsl"
#include "LightSources.hlsl"

cbuffer scene_variables  : register(b0) {
    float4x4 mvp_matrix;
    float4x3 to_world_matrix;
    float4 camera_position;
};

cbuffer lights : register(b1) {
    int4 light_count;
    LightData light_data[12];
}

cbuffer material : register(b2) {
    DefaultShading material;
}

struct PixelInput {
    float4 position : SV_POSITION;
    float4 world_position : WORLD_POSITION;
    float4 normal : NORMAL;
    float2 texcoord : TEXCOORD;
};

float4 main(PixelInput input) : SV_TARGET {
    // NOTE There may be a performance cost associated with having a potential discard, so we should probably have a separate pixel shader for cutouts.
    float coverage = material.coverage(input.texcoord);
    if (coverage < 0.33f)
        discard;

    float3 normal = normalize(input.normal.xyz);
    float3x3 world_to_shading_tbn = create_tbn(normal);

    float3 wo = camera_position.xyz - input.world_position.xyz;
    float3 wi = -reflect(normalize(wo), normal);

    float3 radiance = material.IBL(normal, wi, input.texcoord);

    wo = normalize(mul(world_to_shading_tbn, wo));
    for (int l = 0; l < light_count.x; ++l) {
        LightSample light_sample = sample_light(light_data[l], input.world_position.xyz);
        float3 wi = mul(world_to_shading_tbn, light_sample.direction_to_light);
        float3 f = material.evaluate(wo, wi, input.texcoord);
        radiance += f * light_sample.radiance * abs(wi.z);
    }

    return float4(radiance, coverage);
}