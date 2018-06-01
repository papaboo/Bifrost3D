// SSAO shaders.
// ------------------------------------------------------------------------------------------------
// Copyright (C) 2018, Cogwheel. See AUTHORS.txt for authors
//
// This program is open source and distributed under the New BSD License.
// See LICENSE.txt for more detail.
// ------------------------------------------------------------------------------------------------

#include "RNG.hlsl"
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
    // Transform by the inverse projection matrix
    float4 projected_world_pos = mul(projected_position, scene_vars.inverted_projection_matrix);
    // Divide by w to get the view-space position
    return projected_world_pos.xyz / projected_world_pos.w;
}

float2 uniform_disk_sampling(float2 sample_uv) {
    float r = sqrt(sample_uv.x);
    float theta = TWO_PI * sample_uv.y;
    return r * float2(cos(theta), sin(theta));
}

float4 alchemy_ps(Varyings input) : SV_TARGET {
    // State. Should be in a constant buffer.
    const int sample_count = 64;
    const float radius = 0.3f; // TODO Make depth dependent
    const float intensity_scale = 0.25;
    const float bias = 0.001f;
    const float k = 1.0;

    // Setup sampling
    uint lcg_state = RNG::teschner_hash(input.position.x, input.position.y);

    float3 view_normal = normal_tex.SampleLevel(point_sampler, input.texcoord, 0).rgb;
    float depth = depth_tex.SampleLevel(point_sampler, input.texcoord, 0).r;
    float3 view_position = position_from_depth(depth, input.texcoord);
    // return float4(abs(position), 1);

    float z_scale = -depth;
    float occlusion = 0.0f;
    for (int i = 0; i < sample_count; ++i) {
        float2 uv_offset = uniform_disk_sampling(float2(RNG::lcg_sample(lcg_state), RNG::lcg_sample(lcg_state)));
        float2 uv = saturate(input.texcoord + uv_offset * radius); // TODO Reject when outside the window

        float depth_i = depth_tex.SampleLevel(point_sampler, uv, 0).r;
        float3 view_position_i = position_from_depth(depth_i, uv);
        float3 v_i = view_position_i - view_position;

        // Equation 10
        // occlusion += max(0, dot(v_i, view_normal) + z_scale * bias) / (dot(v_i, v_i) + 0.0001f);
        occlusion += max(0, dot(v_i, view_normal) - bias) / (dot(v_i, v_i) + 0.0001f);
    }

    float a = 1 - (2 * intensity_scale / sample_count) * occlusion;
    a = pow(max(0.0, a), k);

    return float4(a, 1- a, 0, 0);
}