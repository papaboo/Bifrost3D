// GBuffer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Utils.hlsl>

// ------------------------------------------------------------------------------------------------
// Constants.
// ------------------------------------------------------------------------------------------------

cbuffer transform : register(b2) {
    float4x3 to_world_matrix;
};

cbuffer material : register(b3) {
    MaterialParams material_params;
}

cbuffer scene_variables : register(b13) {
    SceneVariables scene_vars;
};

// ------------------------------------------------------------------------------------------------
// Opaque geometry
// ------------------------------------------------------------------------------------------------

struct OpaqueVaryings {
    float4 position : SV_POSITION;
    float4 normal_depth : NORMAL;
};

OpaqueVaryings opaque_VS(float4 geometry : GEOMETRY) {
    OpaqueVaryings varyings;
    float3 world_position = mul(float4(geometry.xyz, 1.0f), to_world_matrix).xyz;
    varyings.position = mul(float4(world_position, 1.0f), scene_vars.view_projection_matrix);
    float3 world_normal = mul(float4(decode_octahedral_normal(asint(geometry.w)), 0.0), to_world_matrix);
    varyings.normal_depth.xyz = mul(float4(world_normal, 0), scene_vars.world_to_view_matrix);
    varyings.normal_depth.w = length(varyings.position);
    return varyings;
}

float2 opaque_PS(OpaqueVaryings varyings) : SV_Target {
    float3 view_space_normal = normalize(varyings.normal_depth.xyz);
    return encode_ss_octahedral_normal(view_space_normal);
}

// ------------------------------------------------------------------------------------------------
// Cutout geometry
// ------------------------------------------------------------------------------------------------

Texture2D coverage_tex : register(t2);
SamplerState coverage_sampler : register(s2);

struct CutoutVaryings {
    float4 position : SV_POSITION;
    float4 normal_depth : NORMAL;
    float2 uv : TEXCOORD;
};

CutoutVaryings cutout_VS(float4 geometry : GEOMETRY, float2 uv : TEXCOORD) {
    CutoutVaryings varyings;
    float3 world_position = mul(float4(geometry.xyz, 1.0f), to_world_matrix).xyz;
    varyings.position = mul(float4(world_position, 1.0f), scene_vars.view_projection_matrix);
    float3 world_normal = mul(float4(decode_octahedral_normal(asint(geometry.w)), 0.0), to_world_matrix);
    varyings.normal_depth.xyz = mul(float4(world_normal, 0), scene_vars.world_to_view_matrix);
    varyings.normal_depth.w = length(varyings.position);
    varyings.uv = uv;
    return varyings;
}

float2 cutout_PS(CutoutVaryings varyings) : SV_Target {
    float coverage = material_params.coverage(varyings.uv, coverage_tex, coverage_sampler);
    if (coverage < CUTOFF)
        discard;

    float3 view_space_normal = normalize(varyings.normal_depth.xyz);
    return encode_ss_octahedral_normal(view_space_normal);
}