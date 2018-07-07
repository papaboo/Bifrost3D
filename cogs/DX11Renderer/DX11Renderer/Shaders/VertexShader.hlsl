// Model vertex shader.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include <Utils.hlsl>

// ------------------------------------------------------------------------------------------------
// Vertex shader.
// ------------------------------------------------------------------------------------------------

cbuffer transform : register(b2) {
    float4x3 to_world_matrix;
};

cbuffer scene_variables : register(b13) {
    SceneVariables scene_vars;
};

struct Output {
    float4 position : SV_POSITION;
    float3 world_position : WORLD_POSITION;
    float3 normal : NORMAL;
    float2 texcoord : TEXCOORD;
};

Output main(float4 geometry : GEOMETRY, float2 texcoord : TEXCOORD) {
    Output output;
    output.world_position.xyz = mul(float4(geometry.xyz, 1.0f), to_world_matrix);
    output.position = mul(float4(output.world_position.xyz, 1.0f), scene_vars.view_projection_matrix);
    output.normal.xyz = mul(float4(decode_octahedral_normal(asint(geometry.w)), 0.0), to_world_matrix);
    output.texcoord = texcoord;
    return output;
}