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

cbuffer constants : register(b1) {
    float world_radius;
    float normal_std_dev;
    float depth_std_dev;
    int sample_count;
    int filtering_enabled; // CPU side only
    float3 __padding;
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
// Bilateral box blur.
// ------------------------------------------------------------------------------------------------

namespace BilateralBoxBlur {

Texture2D normal_tex : register(t0);
Texture2D depth_tex : register(t1);
Texture2D ao_tex : register(t2);

cbuffer per_filter_constants : register(b2) {
    float pixel_offset;
}

void sample_ao(float2 uv, float depth, float3 normal, inout float summed_ao, inout float ao_weight) {
    float sample_depth = depth_tex.SampleLevel(point_sampler, uv, 0).r;
    float normalized_depth_delta = abs(depth - sample_depth) / depth;
    float weight = exp(-pow2(normalized_depth_delta) / (2 * depth_std_dev * depth_std_dev));

    float3 sample_normal = normal_tex.SampleLevel(point_sampler, uv, 0).xyz;
    float cos_theta = dot(sample_normal, normal);
    weight *= exp(-pow2(1.0f - cos_theta) / (2 * normal_std_dev * normal_std_dev));

    weight += 0.0001;

    summed_ao += weight * ao_tex.SampleLevel(point_sampler, uv, 0).r;
    ao_weight += weight;
}

float4 filter_ps(Varyings input) : SV_TARGET {
    float width, height;
    ao_tex.GetDimensions(width, height);

    float2 uv = input.texcoord;
    float3 view_normal = normal_tex.SampleLevel(point_sampler, uv, 0).xyz;
    float depth = depth_tex.SampleLevel(point_sampler, uv, 0).r;

    // No occlusion on the far plane.
    if (depth == 1.0)
        return float4(1, 0, 0, 0);

    float center_ao = 0.0f;
    float center_weight = 0.0f;
    sample_ao(uv, depth, view_normal, center_ao, center_weight); // TODO Can be inlined, just need to compute the weight, which should be pow2(exp(-0)) I guess.

    float2 uv_offset = pixel_offset * rcp(float2(width, height));

    float border_ao = 0.0f;
    float border_weight = 0.0f;

    sample_ao(uv + float2(-uv_offset.x,  uv_offset.y), depth, view_normal, border_ao, border_weight);
    sample_ao(uv + float2(           0,  uv_offset.y), depth, view_normal, border_ao, border_weight);
    sample_ao(uv + float2( uv_offset.x,  uv_offset.y), depth, view_normal, border_ao, border_weight);
    sample_ao(uv + float2(-uv_offset.x,            0), depth, view_normal, border_ao, border_weight);
    sample_ao(uv + float2( uv_offset.x,            0), depth, view_normal, border_ao, border_weight);
    sample_ao(uv + float2(-uv_offset.x, -uv_offset.y), depth, view_normal, border_ao, border_weight);
    sample_ao(uv + float2(           0, -uv_offset.y), depth, view_normal, border_ao, border_weight);
    sample_ao(uv + float2( uv_offset.x, -uv_offset.y), depth, view_normal, border_ao, border_weight);

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
    float4 _projected_view_pos = mul(float4(view_position, 1), scene_vars.projection_matrix);
    float3 projected_view_pos = _projected_view_pos.xyz / _projected_view_pos.w;

    // Transform from normalized screen space to uv.
    return float2(projected_view_pos.x * 0.5 + 0.5, 1 - (projected_view_pos.y + 1) * 0.5);
}

// Transform view position to u(v) in screen space.
// Assumes that the projection_matrix is an actual projection matrix with 0'es in most entries.
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
    // State. Should be in a constant buffer.
    const float intensity_scale = 0.25;
    const float bias = 0.001f;
    const float k = 1.0;

    // Setup sampling
    uint rng_offset = RNG::evenly_distributed_2D_seed(input.position.x, input.position.y);
    float sample_pattern_rotation_angle = float((rng_offset >> 8) & 0xffffff) / float(1 << 24) * TWO_PI;
    float2x2 sample_pattern_rotation = generate_rotation_matrix(sample_pattern_rotation_angle);

    float depth = depth_tex.SampleLevel(point_sampler, input.texcoord, 0).r;
    
    // No occlusion on the far plane.
    if (depth == 1.0)
        return float4(1, 0, 0, 0);

    // float2 encoded_normal = normal_tex.SampleLevel(point_sampler, input.texcoord, 0).rg;
    // float3 view_normal = decode_octahedral_normal(encoded_normal);
    float3 view_normal = normal_tex.SampleLevel(point_sampler, input.texcoord, 0).xyz;
    float3 view_position = position_from_depth(depth, input.texcoord);

    // Compute screen space radius.
    float3 border_view_position = view_position + float3(world_radius, 0, 0);
    float border_u = u_coord_from_view_position(border_view_position);
    float ss_radius = border_u - input.texcoord.x;

    float occlusion = 0.0f;
    for (int i = 0; i < sample_count; ++i) {
        float2 rng_samples = RNG::sample02(i);
        float2 uv_offset = mul(cosine_disk_sampling(rng_samples) * ss_radius, sample_pattern_rotation);

        float2 sample_uv = input.texcoord + uv_offset;

        float depth_i = depth_tex.SampleLevel(point_sampler, sample_uv, 0).r;
        float3 view_position_i = position_from_depth(depth_i, sample_uv);
        float3 v_i = view_position_i - view_position;

        // Equation 10
        occlusion += max(0, dot(v_i, view_normal) - depth * bias) / (dot(v_i, v_i) + 0.0001f);
    }

    float a = 1 - (2 * intensity_scale / sample_count) * occlusion;
    a = pow(max(0.0, a), k);

    return float4(a, 0, 0, 0);
}