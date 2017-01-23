// Model fragment shaders.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "DefaultShading.hlsl"
#include "LightSources.hlsl"

cbuffer scene_variables  : register(b0) {
    float4x4 view_projection_matrix;
    float4 camera_position;
    float4 environment_tint; // .w component is 1 if an environment tex is bound, otherwise 0.
};

cbuffer lights : register(b1) {
    int4 light_count;
    LightData light_data[12];
}

cbuffer material : register(b3) {
    DefaultShading material;
}

struct PixelInput {
    float4 position : SV_POSITION;
    float4 world_position : WORLD_POSITION;
    float4 normal : NORMAL;
    float2 texcoord : TEXCOORD;
};

float3 integration(PixelInput input) {
    float3 normal = normalize(input.normal.xyz);
    float3x3 world_to_shading_tbn = create_tbn(normal);

    float3 wo = camera_position.xyz - input.world_position.xyz;
    float3 wi = -reflect(normalize(wo), normal);

    float3 radiance = environment_tint.rgb * material.IBL(normal, wi, input.texcoord);

    wo = normalize(mul(world_to_shading_tbn, wo));
    for (int l = 0; l < light_count.x; ++l) {
        LightSample light_sample = sample_light(light_data[l], input.world_position.xyz);
        float3 wi = mul(world_to_shading_tbn, light_sample.direction_to_light);
        float3 f = material.evaluate(wo, wi, input.texcoord);
        radiance += f * light_sample.radiance * abs(wi.z);
    }

    return radiance;
}

float4 opaque(PixelInput input) : SV_TARGET{
    // NOTE There may be a performance cost associated with having a potential discard, so we should probably have a separate pixel shader for cutouts.
    float coverage = material.coverage(input.texcoord);
    if (coverage < 0.33f)
        discard;

    return float4(integration(input), 1.0f);
}

float4 transparent(PixelInput input) : SV_TARGET{
    float coverage = material.coverage(input.texcoord);
    return float4(integration(input), coverage);
}