// Environment map shader.
// ---------------------------------------------------------------------------
// Copyright (C) 2016, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License. See
// LICENSE.txt for more detail.
// ---------------------------------------------------------------------------

#include "Utils.hlsl"

float3 project_ray_direction(float2 viewport_pos,
                             float3 camera_position,
                             float4x4 inverted_view_projection_matrix) {

    float4 projected_pos = float4(viewport_pos.x, viewport_pos.y, -1.0f, 1.0f);

    float4 projected_world_pos = mul(projected_pos, inverted_view_projection_matrix);

    float3 ray_origin = projected_world_pos.xyz / projected_world_pos.w;

    return normalize(ray_origin - camera_position);
}

// ---------------------------------------------------------------------------
// Vertex shader.
// ---------------------------------------------------------------------------

struct VertexOutput {
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

VertexOutput main_vs(uint vertex_ID : SV_VertexID) {
    VertexOutput output;
    // Draw triangle: {-1, -3}, { -1, 1 }, { 3, 1 }
    output.position.x = vertex_ID == 2 ? 3 : -1;
    output.position.y = vertex_ID == 0 ? -3 : 1;
    output.position.zw = float2(0.99999, 1.0f);
    output.texcoord = output.position.xy * 0.5 + 0.5; // TODO Can I just get the screen position from some built-in instead?
    return output;
}

// ---------------------------------------------------------------------------
// Pixel shader.
// ---------------------------------------------------------------------------

struct PixelInput {
    float4 position : SV_POSITION;
    float2 texcoord : TEXCOORD;
};

// TODO These can be bound as part of the overall scene variables.
cbuffer scene_variables  : register(b0) {
    float4x4 inverted_view_projection_matrix;
    float4 camera_position;
    float4 tint;
};

Texture2D envTex : register(t0);
SamplerState envSampler : register(s0);

float4 main_ps(PixelInput input) : SV_TARGET {
    float2 viewport_pos = input.texcoord * 2 - 1;
    // TODO We can perform most of this projection in the vertex shader and then simply interpolate the result.
    float3 view_dir = project_ray_direction(viewport_pos, camera_position.xyz, inverted_view_projection_matrix);
    float2 tc = direction_to_latlong_texcoord(view_dir);
    return float4(tint.rgb * envTex.Sample(envSampler, tc).rgb, 1);
}