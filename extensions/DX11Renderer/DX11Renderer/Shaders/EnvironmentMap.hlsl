// Environment map shader.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
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

cbuffer scene_variables : register(b13) {
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
    output.texcoord = output.position.xy;
    return output;
}

// ------------------------------------------------------------------------------------------------
// Pixel shader.
// NOTE We can move the view_dir calculation to the vertex shader, as long as we make sure that 
//      the per pixel view_dir is interpolated over a plane in front of the camera. 
//      If the view dir is normalized in the vertex shader linear interpolation will 
//      give the wrong result.
// ------------------------------------------------------------------------------------------------

Texture2D env_tex : register(t0);
SamplerState env_sampler : register(s0);

float4 main_ps(Varyings input) : SV_TARGET {
    float2 viewport_pos = input.texcoord;
    float3 view_dir = project_ray_direction(viewport_pos, scene_vars.camera_position.xyz, scene_vars.inverted_view_projection_matrix);
    float2 tc = direction_to_latlong_texcoord(view_dir);
    return float4(scene_vars.environment_tint.rgb * env_tex.SampleLevel(env_sampler, tc, 0).rgb, 1);
}