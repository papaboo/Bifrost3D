// GBuffer.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Utils.hlsl>

// ------------------------------------------------------------------------------------------------
// Vertex shader.
// ------------------------------------------------------------------------------------------------

cbuffer scene_variables : register(b0) {
    SceneVariables scene_vars;
};

cbuffer transform : register(b2) {
    float4x3 to_world_matrix;
};

struct Varyings {
    float4 position : SV_POSITION;
    float4 normal_depth : NORMAL;
};

Varyings main_VS(float4 geometry : GEOMETRY) {
    Varyings varyings;
    float3 world_position = mul(float4(geometry.xyz, 1.0f), to_world_matrix).xyz;
    varyings.position = mul(float4(world_position, 1.0f), scene_vars.view_projection_matrix);
    float3 world_normal = mul(decode_octahedral_normal(asint(geometry.w)), to_world_matrix).xyz;
    varyings.normal_depth.xyz = mul(float4(world_normal, 0), scene_vars.world_to_view_matrix);
    varyings.normal_depth.w = length(varyings.position);
    return varyings;
}

float4 main_PS(Varyings varyings) : SV_Target {
    float3 view_space_normal = normalize(varyings.normal_depth.xyz);
    return float4(encode_octahedral_normal(view_space_normal), 0, 1);
}