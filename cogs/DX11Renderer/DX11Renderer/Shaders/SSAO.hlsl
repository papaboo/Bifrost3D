// SSAO shaders.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "Utils.hlsl"

float3 project_ray_direction(float2 viewport_pos,
                             float3 camera_position,
                             float4x4 inverted_view_projection_matrix) {

    float4 projected_pos = float4(viewport_pos.x, viewport_pos.y, -1.0f, 1.0f);

    float4 projected_world_pos = mul(projected_pos, inverted_view_projection_matrix);

    float3 ray_origin = projected_world_pos.xyz / projected_world_pos.w;

    return normalize(ray_origin - camera_position);
}

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

float4 alchemy_ps(Varyings input) : SV_TARGET {
    // float2 viewport_pos = input.texcoord;
    // float3 view_dir = project_ray_direction(viewport_pos, scene_vars.camera_position.xyz, scene_vars.inverted_view_projection_matrix);
    // float2 tc = direction_to_latlong_texcoord(view_dir);
    // return float4(scene_vars.environment_tint.rgb * env_tex.SampleLevel(env_sampler, tc, 0).rgb, 1);

    float depth = depth_tex.SampleLevel(point_sampler, input.texcoord, 0).r;
    float3 view_normal = normal_tex.SampleLevel(point_sampler, input.texcoord, 0).rgb;
    return float4(view_normal * 0.5 + 0.5, 1);
}