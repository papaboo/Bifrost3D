// Environment map shader.
// ------------------------------------------------------------------------------------------------
// Copyright (C) Bifrost. See AUTHORS.txt for authors.
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "Utils.hlsl"

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
    float3 view_direction : TEXCOORD;
};

Varyings main_vs(uint vertex_ID : SV_VertexID) {
    Varyings output;
    // Draw triangle: {-1, -3}, { -1, 1 }, { 3, 1 }
    output.position.x = vertex_ID == 2 ? 3 : -1;
    output.position.y = vertex_ID == 0 ? -3 : 1;
    output.position.zw = float2(1.0f, 1.0f);

    float2 viewport_pos = output.position.xy;
    float4 projected_pos = float4(viewport_pos.x, viewport_pos.y, 1.0f, 1.0f);
    output.view_direction = mul(projected_pos, scene_vars.inverted_projection_matrix).xyz;
    output.view_direction = mul(output.view_direction, (float3x3)scene_vars.view_to_world_matrix);
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
    float2 tc = direction_to_latlong_texcoord(normalize(input.view_direction));
    return float4(scene_vars.environment_tint.rgb * env_tex.SampleLevel(env_sampler, tc, 0).rgb, 1);
}