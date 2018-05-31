// SSAO shaders.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "Utils.hlsl"

// ------------------------------------------------------------------------------------------------
// Scene constants.
// ------------------------------------------------------------------------------------------------

cbuffer scene_variables : register(b0) {
    SceneVariables scene_vars;
};

// ------------------------------------------------------------------------------------------------
// Vertex shader.
// ------------------------------------------------------------------------------------------------

struct Varyings {
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

Varyings main_vs(uint vertex_ID : SV_VertexID) {
    Varyings output;
    // Draw triangle: {-1, -3}, { -1, 1 }, { 3, 1 }
    output.position.x = vertex_ID == 2 ? 3 : -1;
    output.position.y = vertex_ID == 0 ? -3 : 1;
    output.position.zw = float2(1.0f, 1.0f);
    output.texcoord = output.position.xy * 0.5 + 0.5;
    output.texcoord.y = 1.0f - output.texcoord.y;
    return output;
}

// ------------------------------------------------------------------------------------------------
// Alchemy pixel shader.
// ------------------------------------------------------------------------------------------------

Texture2D normal_tex : register(t0);
Texture2D depth_tex : register(t1);
SamplerState point_sampler : register(s14);
SamplerState bilinear_sampler : register(s15); // Always bound since it's generally useful to have a bilinear sampler

// Convertion of depth to view-space position.
// https://mynameismjp.wordpress.com/2009/03/10/reconstructing-position-from-depth/
float3 position_from_depth(float z_over_w, float2 viewport_uv) {
    // Get x/w and y/w from the viewport position
    float x_over_w = viewport_uv.x * 2 - 1;
    float y_over_w = (1 - viewport_uv.y) * 2 - 1;
    float4 projected_position = float4(x_over_w, y_over_w, z_over_w, 1.0f);
    // Transform by the inverse (view?) projection matrix
    float4 projected_world_pos = mul(projected_position, scene_vars.inverted_view_projection_matrix); // TODO Use inverted_projection_matrix??
    // Divide by w to get the view-space position
    return projected_world_pos.xyz / projected_world_pos.w;
}

float4 alchemy_ps(Varyings input) : SV_TARGET {
    float depth = depth_tex.SampleLevel(point_sampler, input.texcoord, 0).r;
    float3 view_normal = normal_tex.SampleLevel(point_sampler, input.texcoord, 0).rgb;
    // return float4(view_normal * 0.5 + 0.5, 1);

    float3 position = position_from_depth(depth, input.texcoord);
    return float4(abs(position), 1);
}