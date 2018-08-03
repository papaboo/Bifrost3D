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

cbuffer constants : register(b1) {
    float world_radius;
    float bias;
    float intensity_scale;
    float falloff;
    float recip_double_normal_variance;
    float recip_double_plane_variance;
    int sample_count;
    int filtering_enabled; // CPU side only
    float2 g_buffer_size;
    float2 recip_g_buffer_viewport_size;
    float2 g_buffer_max_uv;
    int2 g_buffer_to_ao_index_offset;
    float2 ao_buffer_size;
    float2 __padding;
};

cbuffer uv_offset_constants : register(b2) {
    float2 uv_offsets[256];
}

cbuffer scene_variables : register(b13) {
    SceneVariables scene_vars;
};

// ------------------------------------------------------------------------------------------------
// Vertex shader.
// ------------------------------------------------------------------------------------------------

struct Varyings {
    float4 position : SV_POSITION;
    float4 uvs : TEXCOORD; // [ao-uv, g-buffer-uv]

    float2 ao_uv() { return uvs.xy; }
    float2 projection_uv() { return uvs.zw; }
};

Varyings main_vs(uint vertex_ID : SV_VertexID) {
    Varyings output;
    // Draw triangle: {-1, -3}, { -1, 1 }, { 3, 1 }
    output.position.x = vertex_ID == 2 ? 3 : -1;
    output.position.y = vertex_ID == 0 ? -3 : 1;
    output.position.zw = float2(1.0f, 1.0f);

    output.uvs.xy = output.position.xy * 0.5 + 0.5;
    output.uvs.y = 1.0f - output.uvs.y;
    output.uvs.zw = (output.uvs.xy * ao_buffer_size - g_buffer_to_ao_index_offset) * recip_g_buffer_viewport_size;
    return output;
}

// ------------------------------------------------------------------------------------------------
// Bilateral box blur.
// ------------------------------------------------------------------------------------------------

namespace BilateralBoxBlur {

Texture2D normal_tex : register(t0);
Texture2D depth_tex : register(t1);
Texture2D ao_tex : register(t2);

cbuffer per_filter_constants : register(b2) {
    float pixel_offset;
}

void sample_ao(int2 g_buffer_index, float3 normal, float plane_d, inout float summed_ao, inout float ao_weight) {
    // Normal weight
    float3 sample_normal = decode_ss_octahedral_normal(normal_tex[g_buffer_index].xy);
    float cos_theta = dot(sample_normal, normal);
    float weight = exp(-pow2(1.0f - cos_theta) * recip_double_normal_variance);

    // Plane fitting weight
    float sample_depth = depth_tex[g_buffer_index].r;
    float2 uv = (g_buffer_index + 0.5f) * recip_g_buffer_viewport_size;
    float3 sample_position = perspective_position_from_depth(sample_depth, uv, scene_vars.inverted_projection_matrix);
    float distance_to_plane = dot(normal, sample_position) + plane_d;
    weight *= exp(-pow2(distance_to_plane) * recip_double_plane_variance);

    weight += 0.00001;

    summed_ao += weight * ao_tex[g_buffer_index + g_buffer_to_ao_index_offset].r;
    ao_weight += weight;
}

float4 filter_ps(Varyings input) : SV_TARGET {
    int2 g_buffer_index = input.position.xy - g_buffer_to_ao_index_offset;
    float3 view_normal = decode_ss_octahedral_normal(normal_tex[g_buffer_index].xy);
    float depth = depth_tex[g_buffer_index].r;

    // No occlusion on the far plane.
    if (depth == 1.0)
        return float4(1, 0, 0, 0);

    float3 view_position = perspective_position_from_depth(depth, input.projection_uv(), scene_vars.inverted_projection_matrix);
    float plane_d = -dot(view_position, view_normal);

    float center_ao = 0.0f;
    float center_weight = 0.0f;
    sample_ao(g_buffer_index, view_normal, plane_d, center_ao, center_weight); // TODO Can be inlined, just need to compute the weight, which should be pow2(exp(-0)) I guess.

    float border_ao = 0.0f;
    float border_weight = 0.0f;

    sample_ao(g_buffer_index + int2(-pixel_offset,  pixel_offset), view_normal, plane_d, border_ao, border_weight);
    sample_ao(g_buffer_index + int2(            0,  pixel_offset), view_normal, plane_d, border_ao, border_weight);
    sample_ao(g_buffer_index + int2( pixel_offset,  pixel_offset), view_normal, plane_d, border_ao, border_weight);
    sample_ao(g_buffer_index + int2(-pixel_offset,             0), view_normal, plane_d, border_ao, border_weight);
    sample_ao(g_buffer_index + int2( pixel_offset,             0), view_normal, plane_d, border_ao, border_weight);
    sample_ao(g_buffer_index + int2(-pixel_offset, -pixel_offset), view_normal, plane_d, border_ao, border_weight);
    sample_ao(g_buffer_index + int2(            0, -pixel_offset), view_normal, plane_d, border_ao, border_weight);
    sample_ao(g_buffer_index + int2( pixel_offset, -pixel_offset), view_normal, plane_d, border_ao, border_weight);

    // Ensure that we perform at least some filtering in areas with high frequency geometry.
    if (border_weight < 2.0 * center_weight) {
        float weight_scale = 2.0 * center_weight * rcp(border_weight);
        border_ao *= weight_scale;
        border_weight = 2.0 * center_weight;
    }

    return float4((center_ao + border_ao) / (center_weight + border_weight), 0, 0, 0);
}

} // NS BilateralBoxBlur

  // ------------------------------------------------------------------------------------------------
// Alchemy pixel shader.
// ------------------------------------------------------------------------------------------------

Texture2D normal_tex : register(t0);
Texture2D depth_tex : register(t1);

// Transform view position to uv in screen space.
float2 uv_from_view_position(float3 view_position) {
    float4 _projected_view_pos = mul(float4(view_position, 1), scene_vars.projection_matrix);
    float3 projected_view_pos = _projected_view_pos.xyz / _projected_view_pos.w;

    // Transform from normalized screen space to uv.
    return float2(projected_view_pos.x * 0.5 + 0.5, 1 - (projected_view_pos.y + 1) * 0.5);
}

// Transform view position to u(v) in screen space.
// Assumes that the projection_matrix is a perspective projection matrix with 0'es in most entries.
float u_coord_from_view_position(float3 view_position) {
    // float x = dot(float4(view_position, 1), scene_vars.projection_matrix._m00_m10_m20_m30);
    // float w = dot(float4(view_position, 1), scene_vars.projection_matrix._m03_m13_m23_m33);
    float x = view_position.x * scene_vars.projection_matrix._m00;
    float w = view_position.z * scene_vars.projection_matrix._m23;
    float projected_view_pos_x = x / w;

    // Transform from normalized screen space to uv.
    return projected_view_pos_x * 0.5 + 0.5;
}

float2 cosine_disk_sampling(float2 sample_uv) {
    float r = sample_uv.x;
    float theta = TWO_PI * sample_uv.y;
    return r * float2(cos(theta), sin(theta));
}

// Returns a position for the tap on a unit disk.
float2 tap_location(int sample_number, int sample_count, float spin_angle) {
    const float spiral_turns = 73856093;
    float alpha = float(sample_number + 0.5) / sample_count;
    float angle = alpha * (spiral_turns * TWO_PI) + spin_angle;
    return float2(cos(angle), sin(angle)) * alpha;
}

float2x2 generate_rotation_matrix(float angle) {
    float cos_angle = cos(angle);
    float sin_angle = sin(angle);
    float2x2 mat = { cos_angle, -sin_angle,   // row 1
                     sin_angle,  cos_angle }; // row 2
    return mat;
}

float4 alchemy_ps(Varyings input) : SV_TARGET {

    // Setup sampling
    uint rng_offset = RNG::evenly_distributed_2D_seed(input.position.xy);
    float sample_pattern_rotation_angle = rng_offset * (TWO_PI / 4294967296.0f); // Scale a uint to the range [0, 2 * PI[
    float2x2 sample_pattern_rotation = generate_rotation_matrix(sample_pattern_rotation_angle);

    float depth = depth_tex[input.position.xy - g_buffer_to_ao_index_offset].r;

    // No occlusion on the far plane.
    if (depth == 1.0)
        return float4(1, 0, 0, 0);

    float3 view_normal = decode_ss_octahedral_normal(normal_tex[input.position.xy - g_buffer_to_ao_index_offset].xy);
    float pixel_bias = depth * bias * (1.0f - pow2(pow2(pow2(view_normal.z))));
    float3 view_position = perspective_position_from_depth(depth, input.projection_uv(), scene_vars.inverted_projection_matrix) + view_normal * pixel_bias;

    // Compute screen space radius.
    float3 border_view_position = view_position + float3(world_radius, 0, 0);
    float border_u = u_coord_from_view_position(border_view_position);
    float ss_radius = border_u - input.projection_uv().x;

    // Determine occlusion
    float occlusion = 0.0f;
    float used_sample_count = 0.0001f;
    for (int i = 0; i < sample_count; ++i) {
        float2 uv_offset = mul(uv_offsets[i] * ss_radius, sample_pattern_rotation);
        float2 sample_uv = input.projection_uv() + uv_offset;

        // Break if sample is outside g-buffer.
        // TODO Resample somehow to avoid wasting samples.
        if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || sample_uv.y < 0.0 || sample_uv.y > 1.0)
            continue;

        float depth_i = depth_tex.SampleLevel(point_sampler, sample_uv * g_buffer_max_uv.x, 0).r;
        float3 view_position_i = perspective_position_from_depth(depth_i, sample_uv, scene_vars.inverted_projection_matrix);
        float3 v_i = view_position_i - view_position;

        // Equation 10
        occlusion += max(0, dot(v_i, view_normal)) / (dot(v_i, v_i) + 0.0001f);
        ++used_sample_count;
    }

    float a = 1 - (2 * intensity_scale / used_sample_count) * occlusion;
    a = pow(max(0.0, a), falloff);

    // Fade out if radius is less than two pixels.
    float pixel_width = g_buffer_size.x * ss_radius;
    a = lerp(1, a, saturate(pixel_width * 0.5f));

    return float4(a, 0, 0, 0);
}