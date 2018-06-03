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

// Transform depth to view-space position.
// https://mynameismjp.wordpress.com/2009/03/10/reconstructing-position-from-depth/
float3 position_from_depth(float z_over_w, float2 viewport_uv) {
    // Get x/w and y/w from the viewport position
    float x_over_w = viewport_uv.x * 2 - 1;
    float y_over_w = (1 - viewport_uv.y) * 2 - 1;
    float4 projected_position = float4(x_over_w, y_over_w, z_over_w, 1.0f);
    // Transform by the inverse projection matrix
    float4 projected_view_pos = mul(projected_position, scene_vars.inverted_projection_matrix);
    // Divide by w to get the view-space position
    return projected_view_pos.xyz / projected_view_pos.w;
}

// Transform view position to uv in screen space.
float2 uv_from_view_position(float3 view_position) {
    float4 _projected_view_pos = mul(view_position, scene_vars.projection_matrix);
    float3 projected_view_pos = _projected_view_pos.xyz / _projected_view_pos.w;

    // Transform from normalized screen space to uv.
    float2 uv = float2(projected_view_pos.x * 0.5 + 0.5, 1 - (projected_view_pos.y + 1) * 0.5);
    return uv;
}

float2 uniform_disk_sampling(float2 sample_uv) {
    float r = sqrt(sample_uv.x);
    float theta = TWO_PI * sample_uv.y;
    return r * float2(cos(theta), sin(theta));
}

float4 alchemy_ps(Varyings input) : SV_TARGET {
    // State. Should be in a constant buffer.
    const int sample_count = 64;
    const float world_radius = 0.5f;
    const float intensity_scale = 0.25;
    const float bias = 0.001f;
    const float k = 1.0;

    // Setup sampling
    uint rng_offset = RNG::teschner_hash(input.position.x, input.position.y);

    float3 view_normal = normal_tex.SampleLevel(point_sampler, input.texcoord, 0).rgb;
    float depth = depth_tex.SampleLevel(point_sampler, input.texcoord, 0).r;
    float3 view_position = position_from_depth(depth, input.texcoord);
    // return float4(abs(position), 1);

    // Compute screen space radius.
    float3 border_view_position = view_position + float3(world_radius, 0, 0);
    float2 border_uv = uv_from_view_position(border_view_position); // TODO We actually only need x.
    float ss_radius = border_uv.x - input.texcoord.x;

    float occlusion = 0.0f;
    for (int i = 0; i < sample_count; ++i) {
        float2 rng_samples = RNG::sample02(i + rng_offset);
        float2 uv_offset = uniform_disk_sampling(rng_samples);
        float2 uv = saturate(input.texcoord + uv_offset * ss_radius);

        float depth_i = depth_tex.SampleLevel(point_sampler, uv, 0).r;
        float3 view_position_i = position_from_depth(depth_i, uv);
        float3 v_i = view_position_i - view_position;

        // Equation 10
        occlusion += max(0, dot(v_i, view_normal) - depth * bias) / (dot(v_i, v_i) + 0.0001f);
        // TODO Fade out based on length(uv_offset)??
    }

    float a = 1 - (2 * intensity_scale / sample_count) * occlusion;
    a = pow(max(0.0, a), k);

    return float4(a, 1- a, 0, 0);
}